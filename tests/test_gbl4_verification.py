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


def sign_image(
    image: GBL4Image, key: ec.EllipticCurvePrivateKey = PRIVATE_KEY
) -> GBL4Image:
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

    r, s = sign_digest(key, image.signing_digest())
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
        assert subject.verify_signature(PRIVATE_KEY.public_key())


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
    certificate = serialize_tag(image.get_first_tag(GBL4Certificate))

    assert certificate not in image.signing_content()


def test_signature_notices_a_changed_manifest(image: GBL4Image) -> None:
    info = image.get_first_tag(GBL4ManifestInfo)
    tampered = replace_tag(image, info, dataclasses.replace(info, version=0x05000000))

    assert not tampered.verify_signature(PRIVATE_KEY.public_key())


def test_signature_against_an_explicit_key(image: GBL4Image) -> None:
    assert image.verify_signature(PRIVATE_KEY.public_key())
    assert not image.verify_signature(OTHER_KEY.public_key())


def test_signature_without_a_certificate() -> None:
    """A Hue image is signed but carries no certificate; the key comes from elsewhere."""
    image = build_image([APPLICATION], certificate=False)

    assert image.verify_signature(PRIVATE_KEY.public_key())


def test_replaced_certificate_does_not_vouch_for_the_image(image: GBL4Image) -> None:
    """Swapping in your own certificate and re-signing is consistent, but not trusted."""
    certificate = image.get_first_tag(GBL4Certificate)
    forged = sign_image(
        replace_tag(
            image,
            certificate,
            dataclasses.replace(certificate, key=raw_public_key(OTHER_KEY)),
        ),
        OTHER_KEY,
    )

    assert forged.verify_signature(forged.get_first_tag(GBL4Certificate).public_key())
    assert not forged.verify_signature(PRIVATE_KEY.public_key())


def fill_certificate_slot(image: GBL4Image, tags: list[GBL4TagBase]) -> GBL4Image:
    """Swap the unsigned certificate for other tags, padded to the same size.

    Nothing after the slot moves, so the signature and every offset stay valid.
    """
    manifest = image.get_first_tag(GBL4Manifest)
    certificate = image.get_first_tag(GBL4Certificate)
    room = len(serialize_tag(certificate)) - sum(len(serialize_tag(t)) for t in tags)
    children: list[GBL4TagBase] = []

    for child in manifest.children:
        if child is certificate:
            children += [*tags, GBL4Pad(data=bytes(room - 8))]
        else:
            children.append(child)

    return replace_tag(
        image, manifest, dataclasses.replace(manifest, children=children)
    )


def content_hash_of(image: GBL4Image) -> GBL4ContentHash:
    return GBL4ContentHash(
        hash=HashValue(
            hash_type=GBL4HashType.SHA256,
            value=hashlib.sha256(image.content()).digest(),
        )
    )


@pytest.mark.parametrize("blob_type", [GBL4MemorySectionBlob, GBL4SeBlob])
def test_unsigned_content_hash_is_refused(
    blob_type: type[GBL4MemorySectionBlob | GBL4SeBlob],
) -> None:
    """A forged hash in the certificate slot must not vouch for a changed payload.

    An encrypted section is only checked as far as its info tag, and nothing but the
    content hash covers the SE blob, so the content hash is all that stands in the way.
    """
    image = build_image(
        [APPLICATION],
        se_blob=bytes(range(256)),
        encryption=GBL4EncryptionScheme.AES_CCM,
    )
    blob = image.get_first_tag(blob_type)
    tampered = replace_tag(image, blob, blob_type(data=bytes(len(blob.data))))
    forged = fill_certificate_slot(tampered, [content_hash_of(tampered)])

    assert not tampered.verify_content_hash()
    assert forged.verify_signature(PRIVATE_KEY.public_key())

    with pytest.raises(ValidationError, match="2 GBL4ContentHash tags"):
        forged.verify_content_hash()


def test_unsigned_update_is_refused(image: GBL4Image) -> None:
    """An appended section, named by an update forged in the certificate slot."""
    extra = build_section(BOOTLOADER)
    grown = GBL4Image(root=GBL4Root(children=[*image.root.children, extra]))
    update = dataclasses.replace(
        image.get_first_tag(GBL4UpdateMemorySection),
        plain_image_size=len(BOOTLOADER),
        memory_section_position=grown.root_child_offsets()[-1][1],
        hash=HashValue(
            hash_type=GBL4HashType.SHA256,
            value=hashlib.sha256(serialize_tag(extra.info)).digest(),
        ),
    )
    forged = fill_certificate_slot(grown, [content_hash_of(grown), update])

    assert forged.verify_signature(PRIVATE_KEY.public_key())

    with pytest.raises(ValidationError, match="update outside the update process"):
        forged.verify_memory_sections()


def test_signature_comes_from_the_manifest(image: GBL4Image) -> None:
    """A valid signature copied ahead of the manifest must not stand in for its own."""
    signature = image.get_first_tag(GBL4Signature)
    corrupted = replace_tag(
        image,
        signature,
        GBL4Signature(
            signature=dataclasses.replace(signature.signature, value=bytes(64))
        ),
    )
    moved = GBL4Image(root=GBL4Root(children=[signature, *corrupted.root.children]))

    with pytest.raises(ValidationError, match="2 GBL4Signature tags"):
        moved.verify_signature(PRIVATE_KEY.public_key())


def test_image_has_to_start_with_its_manifest(image: GBL4Image) -> None:
    shifted = GBL4Image(
        root=GBL4Root(children=[GBL4Pad(data=bytes(4)), *image.root.children])
    )

    for check in (
        shifted.verify_content_hash,
        shifted.verify_memory_sections,
        lambda: shifted.verify_signature(PRIVATE_KEY.public_key()),
    ):
        with pytest.raises(ValidationError, match="does not start with a manifest"):
            check()


def test_memory_sections_of_an_se_only_image() -> None:
    """An SE update has no memory section, so there is nothing to compare."""
    image = build_image([], se_blob=bytes(range(256)))

    assert image.verify_memory_sections()


@pytest.mark.parametrize(
    "blob",
    [lzma_compress(bytes(1024 * 1024)), b"not an LZMA stream"],
    ids=["expands past the update size", "corrupt"],
)
def test_memory_sections_notice_a_blob_that_does_not_expand(
    image: GBL4Image, blob: bytes
) -> None:
    """The info digest this check reads does not cover the blob, the content hash does."""
    swapped = replace_tag(
        image,
        image.get_first_tag(GBL4MemorySectionBlob),
        GBL4MemorySectionBlob(data=blob),
    )

    assert not swapped.verify_memory_sections()


def test_plain_image_stops_at_max_size(image: GBL4Image) -> None:
    section = image.get_first_tag(GBL4MemorySection)

    assert image.plain_image(section, max_size=len(APPLICATION)) == APPLICATION

    with pytest.raises(ValidationError, match="at most 100 bytes"):
        image.plain_image(section, max_size=100)


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
        stripped.verify_signature(PRIVATE_KEY.public_key())

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
        tampered.verify_signature(PRIVATE_KEY.public_key())


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
    assert encrypted.verify_signature(PRIVATE_KEY.public_key())


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


@pytest.mark.parametrize(
    "plain",
    [APPLICATION, APPLICATION * 640],
    # The v4 size field is 24 bits, past the 4 MiB the v3 helper defaults to
    ids=["8 KiB", "5 MiB"],
)
def test_plain_image_expands_lz4(plain: bytes) -> None:
    section = build_section(plain, compression=GBL4CompressionScheme.NONE)
    packed = dataclasses.replace(
        section,
        children=[
            dataclasses.replace(
                section.info, compression_scheme=GBL4CompressionScheme.LZ4
            ),
            GBL4MemorySectionBlob(data=lz4_compress(plain)),
        ],
    )
    image = build_image([APPLICATION])

    assert image.plain_image(packed) == plain


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
        tampered.verify_signature(PRIVATE_KEY.public_key())


def test_public_key_must_be_a_raw_point() -> None:
    with pytest.raises(ValueError, match="Key must be 64 bytes"):
        load_public_key(bytes(32))
