"""The manifest checks a Series 3 bootloader makes before it flashes an image.

The images here are synthetic, built to the layout the real ones use: a manifest whose
signature covers the tags after it, a content hash over the payload that follows the
manifest, and one memory section per update, bound to the manifest by the digest of its
info tag. `tests/test_gbl4.py` runs the same checks against the local corpus.
"""

from __future__ import annotations

import dataclasses
import hashlib

from cryptography.hazmat.primitives.asymmetric import ec
import pytest

from pygbl import (
    GBL4BundleVersion,
    GBL4Certificate,
    GBL4CompressionScheme,
    GBL4ContentHash,
    GBL4EncryptionScheme,
    GBL4Feature,
    GBL4HashType,
    GBL4Image,
    GBL4Manifest,
    GBL4ManifestFinish,
    GBL4ManifestInfo,
    GBL4MemorySection,
    GBL4MemorySectionBlob,
    GBL4MemorySectionInfo,
    GBL4Pad,
    GBL4Root,
    GBL4SeBlob,
    GBL4Signature,
    GBL4SignatureType,
    GBL4TagBase,
    GBL4UpdateMemorySection,
    GBL4UpdateProcess,
    GBL4UpdateSe,
    HashValue,
    ParseError,
    ValidationError,
)
from pygbl.compression import lz4_compress, lzma_compress
from pygbl.crypto import load_public_key, sign_digest
from pygbl.gbl4 import serialize_tag

PRIVATE_KEY = ec.generate_private_key(ec.SECP256R1())
OTHER_KEY = ec.generate_private_key(ec.SECP256R1())
PRODUCT_ID = bytes(range(16))
APPLICATION = bytes((i * 7) % 251 for i in range(8192))
BOOTLOADER = bytes((i * 11) % 241 for i in range(2048))


def raw_public_key(private_key: ec.EllipticCurvePrivateKey) -> bytes:
    numbers = private_key.public_key().public_numbers()

    return numbers.x.to_bytes(32, "big") + numbers.y.to_bytes(32, "big")


def build_section(
    plain: bytes,
    *,
    compression: GBL4CompressionScheme = GBL4CompressionScheme.LZMA,
    encryption: GBL4EncryptionScheme = GBL4EncryptionScheme.NONE,
) -> GBL4MemorySection:
    blob = lzma_compress(plain) if compression == GBL4CompressionScheme.LZMA else plain

    return GBL4MemorySection(
        children=[
            GBL4MemorySectionInfo(
                compression_scheme=compression,
                encryption_scheme=encryption,
                secure_boot_scheme=0,
                reserved=0,
                sign_block_size=0,
                num_blocks=0,
                nonce=bytes(12),
                # A fixed 64 byte field holding a 32 byte digest, zero padded
                final_image_hash=hashlib.sha256(plain).digest() + bytes(32),
                secure_boot_signature=bytes(128),
            ),
            GBL4MemorySectionBlob(data=blob),
        ]
    )


def build_image(
    payloads: list[bytes],
    *,
    certificate: bool = True,
    se_blob: bytes | None = None,
    pad: bool = False,
    encryption: GBL4EncryptionScheme = GBL4EncryptionScheme.NONE,
) -> GBL4Image:
    """Assemble a signed image whose offsets, digests and signature all agree.

    Every field the manifest cross-references has a fixed width, so the layout can be
    settled first and the values filled in afterwards without moving anything.
    """
    sections = [build_section(p, encryption=encryption) for p in payloads]
    updates = [
        GBL4UpdateMemorySection(
            target_memory=0,
            plain_image_size=len(plain),
            target_address=0x01008000 + index * 0x1000,
            type=1,
            version=1,
            capabilities=0,
            memory_section_position=0,
            hash=HashValue(
                hash_type=GBL4HashType.SHA256,
                value=hashlib.sha256(serialize_tag(section.info)).digest(),
            ),
        )
        for index, (plain, section) in enumerate(zip(payloads, sections))
    ]

    manifest_children: list[GBL4TagBase] = []

    if certificate:
        manifest_children.append(
            GBL4Certificate(
                struct_version=1,
                flags=bytes(3),
                key=raw_public_key(PRIVATE_KEY),
                version=2,
                signature=bytes(64),
            )
        )

    manifest_children += [
        GBL4Signature(
            signature=HashValue(hash_type=GBL4SignatureType.ECDSA_P256, value=bytes(64))
        ),
        GBL4ManifestInfo(version=0x04000000, features=GBL4Feature.COMPRESSION),
        GBL4BundleVersion(product_id=PRODUCT_ID, bundle_version=0, min_version=0),
        GBL4ContentHash(hash=HashValue(hash_type=GBL4HashType.SHA256, value=bytes(32))),
        GBL4UpdateProcess(
            children=(
                ([GBL4UpdateSe(version=0x30307, tlv_position=0)] if se_blob else [])
                + [*updates, GBL4ManifestFinish()]
            )
        ),
    ]

    root_children: list[GBL4TagBase] = [GBL4Manifest(children=manifest_children)]

    if se_blob is not None:
        root_children.append(GBL4SeBlob(data=se_blob))

    root_children += sections

    if pad:
        root_children.append(GBL4Pad(data=b"\xff\xff"))

    image = GBL4Image(root=GBL4Root(children=root_children))

    return sign_image(fill_positions(image))


def fill_positions(image: GBL4Image) -> GBL4Image:
    """Point each update at its section, now that the layout is known."""
    offsets = image.root_child_offsets()
    sections = [(o, c) for c, o in offsets if isinstance(c, GBL4MemorySection)]
    se_offsets = [o for c, o in offsets if isinstance(c, GBL4SeBlob)]
    process = image.get_first_tag(GBL4UpdateProcess)
    children: list[GBL4TagBase] = []
    index = 0

    for child in process.children:
        if isinstance(child, GBL4UpdateMemorySection):
            child = dataclasses.replace(
                child, memory_section_position=sections[index][0]
            )
            index += 1
        elif isinstance(child, GBL4UpdateSe):
            # Unlike a memory section, an SE update points at the blob's payload
            child = dataclasses.replace(child, tlv_position=se_offsets[0] + 8)

        children.append(child)

    return replace_tag(image, process, GBL4UpdateProcess(children=children))


def sign_image(image: GBL4Image) -> GBL4Image:
    """Fill in the content hash, then sign everything the signature covers."""
    content_hash = image.get_first_tag(GBL4ContentHash)
    image = replace_tag(
        image,
        content_hash,
        GBL4ContentHash(
            hash=HashValue(
                hash_type=GBL4HashType.SHA256,
                value=hashlib.sha256(image.content()).digest(),
            )
        ),
    )

    r, s = sign_digest(PRIVATE_KEY, image.signing_digest())
    signature = image.get_first_tag(GBL4Signature)

    return replace_tag(
        image,
        signature,
        GBL4Signature(
            signature=HashValue(hash_type=GBL4SignatureType.ECDSA_P256, value=r + s)
        ),
    )


def replace_tag(image: GBL4Image, old: GBL4TagBase, new: GBL4TagBase) -> GBL4Image:
    """Swap one tag for another of the same size, anywhere in the tree."""

    def walk(tags: list[GBL4TagBase]) -> list[GBL4TagBase]:
        out: list[GBL4TagBase] = []

        for tag in tags:
            if tag is old:
                out.append(new)
            elif isinstance(tag, (GBL4Manifest, GBL4UpdateProcess, GBL4MemorySection)):
                out.append(dataclasses.replace(tag, children=walk(tag.children)))
            else:
                out.append(tag)

        return out

    return GBL4Image(root=GBL4Root(children=walk(image.root.children)))


@pytest.fixture
def image() -> GBL4Image:
    return build_image([APPLICATION])


@pytest.fixture
def bundle() -> GBL4Image:
    """The shape a Hue SiMG301 file has: an SE blob and two memory sections."""
    return build_image(
        [APPLICATION, BOOTLOADER], se_blob=bytes(range(256)) * 4, pad=True
    )


def test_offsets_include_the_root_header(image: GBL4Image) -> None:
    """Positions in the manifest are file offsets, not offsets into the root payload."""
    data = image.serialize()

    for tag, offset in image.root_child_offsets():
        assert data[offset : offset + len(serialize_tag(tag))] == serialize_tag(tag)

    assert image.root_child_offsets()[0][1] == 8


def test_update_points_at_its_section(bundle: GBL4Image) -> None:
    sections = bundle.get_tags(GBL4MemorySection)
    updates = bundle.get_tags(GBL4UpdateMemorySection)

    assert len(sections) == len(updates) == 2

    for update, section in zip(updates, sections):
        assert bundle.memory_section_at(update.memory_section_position) is section


def test_memory_section_at_rejects_a_bad_position(image: GBL4Image) -> None:
    with pytest.raises(KeyError, match="No memory section at offset"):
        image.memory_section_at(0x1234)


def test_verification_passes(image: GBL4Image, bundle: GBL4Image) -> None:
    for subject in (image, bundle):
        assert subject.verify_content_hash()
        assert subject.verify_memory_sections()
        assert subject.verify_signature()


def test_content_is_everything_after_the_manifest(image: GBL4Image) -> None:
    data = image.serialize()
    manifest = serialize_tag(image.get_first_tag(GBL4Manifest))

    assert image.content() == data[8 + len(manifest) :]


def test_content_hash_notices_a_changed_payload(image: GBL4Image) -> None:
    blob = image.get_first_tag(GBL4MemorySectionBlob)
    flipped = bytes([blob.data[0] ^ 0xFF]) + blob.data[1:]
    tampered = replace_tag(image, blob, GBL4MemorySectionBlob(data=flipped))

    assert not tampered.verify_content_hash()


def test_memory_sections_notice_a_changed_info_tag(image: GBL4Image) -> None:
    info = image.get_first_tag(GBL4MemorySectionInfo)
    tampered = replace_tag(
        image, info, dataclasses.replace(info, final_image_hash=bytes(64))
    )

    assert not tampered.verify_memory_sections()


def test_memory_sections_notice_a_moved_section(image: GBL4Image) -> None:
    update = image.get_first_tag(GBL4UpdateMemorySection)
    tampered = replace_tag(
        image, update, dataclasses.replace(update, memory_section_position=0x1234)
    )

    assert not tampered.verify_memory_sections()


def test_memory_sections_notice_a_wrong_plain_size(image: GBL4Image) -> None:
    update = image.get_first_tag(GBL4UpdateMemorySection)
    tampered = replace_tag(
        image, update, dataclasses.replace(update, plain_image_size=1)
    )

    assert not tampered.verify_memory_sections()


def test_signature_covers_the_manifest_after_itself(image: GBL4Image) -> None:
    manifest = image.get_first_tag(GBL4Manifest)
    signature = image.get_first_tag(GBL4Signature)
    after = manifest.children[manifest.children.index(signature) + 1 :]

    assert image.signing_content() == b"".join(serialize_tag(t) for t in after)
    # The certificate sits before the signature, so it is not covered
    assert serialize_tag(image.get_first_tag(GBL4Certificate)) not in image.content()


def test_signature_notices_a_changed_manifest(image: GBL4Image) -> None:
    info = image.get_first_tag(GBL4ManifestInfo)
    tampered = replace_tag(image, info, dataclasses.replace(info, version=0x05000000))

    assert not tampered.verify_signature()


def test_signature_against_an_explicit_key(image: GBL4Image) -> None:
    assert image.verify_signature(PRIVATE_KEY.public_key())
    assert not image.verify_signature(OTHER_KEY.public_key())


def test_signature_without_a_certificate() -> None:
    """A Hue image is signed but carries no certificate, so a key has to be supplied."""
    unsigned_by_us = build_image([APPLICATION], certificate=False)

    with pytest.raises(KeyError, match="no certificate"):
        unsigned_by_us.verify_signature()

    assert unsigned_by_us.verify_signature(PRIVATE_KEY.public_key())


def test_unsigned_image_raises(image: GBL4Image) -> None:
    manifest = image.get_first_tag(GBL4Manifest)
    signature = image.get_first_tag(GBL4Signature)
    stripped = GBL4Image(
        root=GBL4Root(
            children=[
                dataclasses.replace(
                    manifest,
                    children=[c for c in manifest.children if c is not signature],
                ),
                *image.root.children[1:],
            ]
        )
    )

    with pytest.raises(KeyError, match="not signed"):
        stripped.verify_signature()

    with pytest.raises(KeyError, match="not signed"):
        stripped.signing_content()


def test_unknown_signature_type_is_refused(image: GBL4Image) -> None:
    signature = image.get_first_tag(GBL4Signature)
    tampered = replace_tag(
        image,
        signature,
        GBL4Signature(signature=dataclasses.replace(signature.signature, hash_type=9)),
    )

    with pytest.raises(ValidationError, match="Unsupported signature type"):
        tampered.verify_signature()


def test_unknown_hash_type_is_refused(image: GBL4Image) -> None:
    content_hash = image.get_first_tag(GBL4ContentHash)
    tampered = replace_tag(
        image,
        content_hash,
        GBL4ContentHash(hash=dataclasses.replace(content_hash.hash, hash_type=9)),
    )

    with pytest.raises(ValidationError, match="Unsupported hash type"):
        tampered.verify_content_hash()


def test_plain_image_expands_lzma(image: GBL4Image) -> None:
    section = image.get_first_tag(GBL4MemorySection)

    assert image.plain_image(section) == APPLICATION
    assert len(section.blob.data) < len(APPLICATION)


def test_plain_image_passes_through_uncompressed_sections() -> None:
    uncompressed = build_image([APPLICATION])
    section = uncompressed.get_first_tag(GBL4MemorySection)
    swapped = replace_tag(
        uncompressed,
        section,
        GBL4MemorySection(
            children=[
                dataclasses.replace(
                    section.info, compression_scheme=GBL4CompressionScheme.NONE
                ),
                GBL4MemorySectionBlob(data=APPLICATION),
            ]
        ),
    )

    assert swapped.plain_image(swapped.get_first_tag(GBL4MemorySection)) == APPLICATION


def test_plain_image_refuses_an_encrypted_section() -> None:
    encrypted = build_image([APPLICATION], encryption=GBL4EncryptionScheme.AES_CCM)
    section = encrypted.get_first_tag(GBL4MemorySection)

    with pytest.raises(ValidationError, match="encrypted"):
        encrypted.plain_image(section)


def test_encrypted_sections_are_still_bound_to_the_manifest() -> None:
    """What can be checked without the key: the section is the one the manifest names."""
    encrypted = build_image([APPLICATION], encryption=GBL4EncryptionScheme.AES_CCM)

    assert encrypted.verify_memory_sections()
    assert encrypted.verify_content_hash()
    assert encrypted.verify_signature()


def test_plain_image_refuses_an_unknown_compression_scheme(image: GBL4Image) -> None:
    section = image.get_first_tag(GBL4MemorySection)
    tampered = replace_tag(
        image,
        section,
        dataclasses.replace(
            section,
            children=[
                dataclasses.replace(section.info, compression_scheme=9),
                section.blob,
            ],
        ),
    )

    with pytest.raises(ValidationError, match="Unsupported compression scheme"):
        tampered.plain_image(tampered.get_first_tag(GBL4MemorySection))


def test_section_without_an_info_tag_is_refused(image: GBL4Image) -> None:
    section = image.get_first_tag(GBL4MemorySection)
    stripped = GBL4MemorySection(children=[section.blob])

    with pytest.raises(ParseError, match="no info tag"):
        _ = stripped.info

    with pytest.raises(ParseError, match="no blob tag"):
        _ = GBL4MemorySection(children=[section.info]).blob


def rebind(image: GBL4Image, section: GBL4MemorySection) -> GBL4Image:
    """Recompute the update digest for a section that was changed on purpose."""
    update = image.get_first_tag(GBL4UpdateMemorySection)

    return replace_tag(
        image,
        update,
        dataclasses.replace(
            update,
            hash=HashValue(
                hash_type=GBL4HashType.SHA256,
                value=hashlib.sha256(serialize_tag(section.info)).digest(),
            ),
        ),
    )


def test_memory_sections_notice_a_wrong_final_image_hash(image: GBL4Image) -> None:
    """The section still matches its update, but not the image it expands to."""
    section = image.get_first_tag(GBL4MemorySection)
    changed = dataclasses.replace(
        section,
        children=[
            dataclasses.replace(section.info, final_image_hash=bytes(64)),
            section.blob,
        ],
    )

    assert not rebind(
        replace_tag(image, section, changed), changed
    ).verify_memory_sections()


def test_plain_image_expands_lz4() -> None:
    section = build_section(APPLICATION, compression=GBL4CompressionScheme.NONE)
    packed = dataclasses.replace(
        section,
        children=[
            dataclasses.replace(
                section.info, compression_scheme=GBL4CompressionScheme.LZ4
            ),
            GBL4MemorySectionBlob(data=lz4_compress(APPLICATION)),
        ],
    )
    image = build_image([APPLICATION])

    assert image.plain_image(packed) == APPLICATION


def test_content_without_a_manifest(image: GBL4Image) -> None:
    stripped = GBL4Image(
        root=GBL4Root(
            children=[c for c in image.root.children if not isinstance(c, GBL4Manifest)]
        )
    )

    with pytest.raises(KeyError, match="no manifest"):
        stripped.content()


def test_signature_of_the_wrong_length_is_refused(image: GBL4Image) -> None:
    signature = image.get_first_tag(GBL4Signature)
    tampered = replace_tag(
        image,
        signature,
        GBL4Signature(
            signature=dataclasses.replace(signature.signature, value=bytes(32))
        ),
    )

    with pytest.raises(ValidationError, match="must be 64 bytes"):
        tampered.verify_signature()


def test_public_key_must_be_a_raw_point() -> None:
    with pytest.raises(ValueError, match="Key must be 64 bytes"):
        load_public_key(bytes(32))
