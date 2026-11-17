"""GBLv4, the format used by Series 3 parts.

A redesign rather than an extension of v3: tags nest, the whole image is described by a
signed manifest, and one file can bundle several updates. The framing is the same
little-endian `tag id (4) + length (4)` header, applied recursively.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
from typing import TYPE_CHECKING, Any, Self, Union

from pygbl.compression import DECOMPRESSION_ERRORS, lz4_decompress, lzma_decompress
from pygbl.crypto import load_public_key, verify_digest
from pygbl.types import ParseError, T, ValidationError

if TYPE_CHECKING:
    # Only needed for annotations: signature verification is an optional feature
    from cryptography.hazmat.primitives.asymmetric import ec

# Root tag `0x84A617EB`, from the same `A617EB` family as v3's header but with version
# byte 0x84 instead of 0x03
GBL4_MAGIC = b"\xeb\x17\xa6\x84"

# An update's `plain_image_size` is a 24 bit field
GBL4_MAX_PLAIN_IMAGE_SIZE = 0xFFFFFF


class GBL4TagId(enum.IntEnum):
    """Tag identifiers from the SDK's series 3 parser."""

    V4 = 0x84A617EB
    MANIFEST = 0xAA01012A
    CERTIFICATE = 0x2B01012B
    SIGNATURE = 0x2B02022B
    MANIFEST_INFO = 0x2B03032B
    BUNDLE_VERSION = 0x2B04042B
    CONTENT_HASH = 0x2B05052B
    UPDATE_PROCESS = 0xAB06062B
    UPDATE_SE = 0x2C02022C
    UPDATE_MEMORY_SECTION = 0x2C03032C
    MANIFEST_FINISH = 0x2C04042C
    SE_BLOB = 0x3A02023A
    MEMORY_SECTION = 0xBA01013A
    MEMORY_SECTION_INFO = 0x3B01013B
    MEMORY_SECTION_BLOB = 0x3B02023B
    PAD = 0xEE0101EE


class GBL4Feature(enum.IntFlag, boundary=enum.KEEP):
    NONE = 0x00
    COMPRESSION = 0x01
    ENCRYPTION_AESCCM = 0x02
    SIGNATURE_ECDSA = 0x04


class GBL4TargetMemory(enum.IntEnum):
    SRAM = 0
    OSPI1 = 1
    OSPI2 = 2


class GBL4HashType(enum.IntEnum):
    """The `hash_type` word of a `HashValue` holding a digest."""

    SHA256 = 1


class GBL4SignatureType(enum.IntEnum):
    """The same word, when the `HashValue` holds a signature instead of a digest."""

    ECDSA_P256 = 2


class GBL4CompressionScheme(enum.IntEnum):
    NONE = 0
    LZ4 = 1
    LZMA = 2


class GBL4EncryptionScheme(enum.IntEnum):
    NONE = 0
    AES_CCM = 1


@dataclasses.dataclass(frozen=True, kw_only=True)
class HashValue:
    """A hash or signature: a type word followed by the value.

    The SDK declares the value as a union big enough for SHA-512, but on the wire it is
    only as long as the hash actually used, so a SHA-256 content hash tag is 36 bytes
    rather than 68.
    """

    hash_type: int
    value: bytes

    @classmethod
    def from_bytes(cls, data: bytes) -> HashValue:
        if len(data) < 4:
            raise ParseError(f"Hash value is truncated: {len(data)} bytes")

        return cls(hash_type=int.from_bytes(data[0:4], "little"), value=data[4:])

    def serialize(self) -> bytes:
        return self.hash_type.to_bytes(4, "little") + self.value

    def digest(self, data: bytes) -> bytes:
        """Hash `data` with the algorithm this value declares."""
        if self.hash_type != GBL4HashType.SHA256:
            raise ValidationError(f"Unsupported hash type {self.hash_type:#x}")

        return hashlib.sha256(data).digest()

    def matches(self, data: bytes) -> bool:
        """Whether `data` hashes to this value."""
        return self.digest(data) == self.value


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL4TagBase:
    """A tag id plus a payload encoder, exactly as in v3."""

    tag_id: int

    def serialize_payload(self) -> bytes:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True, kw_only=True)
class _GBL4Container(GBL4TagBase):
    """A tag whose payload is itself a sequence of tags."""

    children: list[GBL4TagBase]

    @classmethod
    def from_payload(cls, payload: bytes) -> Self:
        return cls(tag_id=cls.tag_id, children=parse_tag_stream(payload))

    def serialize_payload(self) -> bytes:
        return b"".join(serialize_tag(child) for child in self.children)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _GBL4OpaqueData(GBL4TagBase):
    data: bytes

    @classmethod
    def from_payload(cls, payload: bytes) -> Self:
        return cls(tag_id=cls.tag_id, data=payload)

    def serialize_payload(self) -> bytes:
        return self.data


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL4Root(_GBL4Container):
    tag_id: int = GBL4TagId.V4


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL4Manifest(_GBL4Container):
    tag_id: int = GBL4TagId.MANIFEST


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL4UpdateProcess(_GBL4Container):
    tag_id: int = GBL4TagId.UPDATE_PROCESS


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL4MemorySection(_GBL4Container):
    """One update payload: a fixed info header followed by the image blob."""

    tag_id: int = GBL4TagId.MEMORY_SECTION

    @property
    def info(self) -> GBL4MemorySectionInfo:
        for child in self.children:
            if isinstance(child, GBL4MemorySectionInfo):
                return child

        raise ParseError("Memory section has no info tag")

    @property
    def blob(self) -> GBL4MemorySectionBlob:
        for child in self.children:
            if isinstance(child, GBL4MemorySectionBlob):
                return child

        raise ParseError("Memory section has no blob tag")


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL4Certificate(GBL4TagBase):
    """Identical layout to v3's `ApplicationCertificate_t`."""

    tag_id: int = GBL4TagId.CERTIFICATE

    struct_version: int
    flags: bytes
    key: bytes
    version: int
    signature: bytes

    @classmethod
    def from_payload(cls, payload: bytes) -> GBL4Certificate:
        if len(payload) != 136:
            raise ParseError(f"GBLv4 certificate must be 136 bytes, got {len(payload)}")

        return cls(
            struct_version=payload[0],
            flags=payload[1:4],
            key=payload[4:68],
            version=int.from_bytes(payload[68:72], "little"),
            signature=payload[72:136],
        )

    def serialize_payload(self) -> bytes:
        return (
            bytes([self.struct_version])
            + self.flags
            + self.key
            + self.version.to_bytes(4, "little")
            + self.signature
        )

    def public_key(self) -> ec.EllipticCurvePublicKey:
        """The certified key, as a `cryptography` object."""
        return load_public_key(self.key)


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL4Signature(GBL4TagBase):
    tag_id: int = GBL4TagId.SIGNATURE

    signature: HashValue

    @classmethod
    def from_payload(cls, payload: bytes) -> GBL4Signature:
        return cls(signature=HashValue.from_bytes(payload))

    def serialize_payload(self) -> bytes:
        return self.signature.serialize()


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL4ManifestInfo(GBL4TagBase):
    tag_id: int = GBL4TagId.MANIFEST_INFO

    version: int
    features: GBL4Feature

    @classmethod
    def from_payload(cls, payload: bytes) -> GBL4ManifestInfo:
        if len(payload) != 8:
            raise ParseError(f"GBLv4 manifest info must be 8 bytes, got {len(payload)}")

        return cls(
            version=int.from_bytes(payload[0:4], "little"),
            features=GBL4Feature(int.from_bytes(payload[4:8], "little")),
        )

    def serialize_payload(self) -> bytes:
        return self.version.to_bytes(4, "little") + int(self.features).to_bytes(
            4, "little"
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL4BundleVersion(GBL4TagBase):
    tag_id: int = GBL4TagId.BUNDLE_VERSION

    product_id: bytes
    bundle_version: int
    min_version: int

    @classmethod
    def from_payload(cls, payload: bytes) -> GBL4BundleVersion:
        if len(payload) != 24:
            raise ParseError(
                f"GBLv4 bundle version must be 24 bytes, got {len(payload)}"
            )

        return cls(
            product_id=payload[0:16],
            bundle_version=int.from_bytes(payload[16:20], "little"),
            min_version=int.from_bytes(payload[20:24], "little"),
        )

    def serialize_payload(self) -> bytes:
        return (
            self.product_id
            + self.bundle_version.to_bytes(4, "little")
            + self.min_version.to_bytes(4, "little")
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL4ContentHash(GBL4TagBase):
    tag_id: int = GBL4TagId.CONTENT_HASH

    hash: HashValue

    @classmethod
    def from_payload(cls, payload: bytes) -> GBL4ContentHash:
        return cls(hash=HashValue.from_bytes(payload))

    def serialize_payload(self) -> bytes:
        return self.hash.serialize()


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL4UpdateSe(GBL4TagBase):
    tag_id: int = GBL4TagId.UPDATE_SE

    version: int
    tlv_position: int

    @classmethod
    def from_payload(cls, payload: bytes) -> GBL4UpdateSe:
        if len(payload) != 8:
            raise ParseError(f"GBLv4 SE update must be 8 bytes, got {len(payload)}")

        return cls(
            version=int.from_bytes(payload[0:4], "little"),
            tlv_position=int.from_bytes(payload[4:8], "little"),
        )

    def serialize_payload(self) -> bytes:
        return self.version.to_bytes(4, "little") + self.tlv_position.to_bytes(
            4, "little"
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL4UpdateMemorySection(GBL4TagBase):
    tag_id: int = GBL4TagId.UPDATE_MEMORY_SECTION

    target_memory: int
    plain_image_size: int
    target_address: int
    type: int
    version: int
    capabilities: int
    memory_section_position: int
    hash: HashValue

    @classmethod
    def from_payload(cls, payload: bytes) -> GBL4UpdateMemorySection:
        if len(payload) < 28:
            raise ParseError(
                f"GBLv4 memory section update is truncated: {len(payload)} bytes"
            )

        return cls(
            target_memory=payload[0],
            # A 24 bit size, which is why the struct is not naturally aligned
            plain_image_size=int.from_bytes(payload[1:4], "little"),
            target_address=int.from_bytes(payload[4:8], "little"),
            type=int.from_bytes(payload[8:12], "little"),
            version=int.from_bytes(payload[12:16], "little"),
            capabilities=int.from_bytes(payload[16:20], "little"),
            memory_section_position=int.from_bytes(payload[20:24], "little"),
            hash=HashValue.from_bytes(payload[24:]),
        )

    def serialize_payload(self) -> bytes:
        return (
            bytes([self.target_memory])
            + self.plain_image_size.to_bytes(3, "little")
            + self.target_address.to_bytes(4, "little")
            + self.type.to_bytes(4, "little")
            + self.version.to_bytes(4, "little")
            + self.capabilities.to_bytes(4, "little")
            + self.memory_section_position.to_bytes(4, "little")
            + self.hash.serialize()
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL4ManifestFinish(GBL4TagBase):
    """A zero-length marker closing the manifest."""

    tag_id: int = GBL4TagId.MANIFEST_FINISH

    @classmethod
    def from_payload(cls, payload: bytes) -> GBL4ManifestFinish:
        if payload:
            raise ParseError(
                f"GBLv4 manifest finish must be empty, got {len(payload)} bytes"
            )

        return cls()

    def serialize_payload(self) -> bytes:
        return b""


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL4MemorySectionInfo(GBL4TagBase):
    """The SDK's `MemSectionPartialInfo_t`, a fixed 212 bytes."""

    tag_id: int = GBL4TagId.MEMORY_SECTION_INFO

    compression_scheme: int
    encryption_scheme: int
    secure_boot_scheme: int
    reserved: int
    sign_block_size: int
    num_blocks: int
    nonce: bytes
    final_image_hash: bytes
    secure_boot_signature: bytes

    @classmethod
    def from_payload(cls, payload: bytes) -> GBL4MemorySectionInfo:
        if len(payload) != 212:
            raise ParseError(
                f"GBLv4 memory section info must be 212 bytes, got {len(payload)}"
            )

        return cls(
            compression_scheme=payload[0],
            encryption_scheme=payload[1],
            secure_boot_scheme=payload[2],
            reserved=payload[3],
            sign_block_size=int.from_bytes(payload[4:6], "little"),
            num_blocks=int.from_bytes(payload[6:8], "little"),
            nonce=payload[8:20],
            final_image_hash=payload[20:84],
            secure_boot_signature=payload[84:212],
        )

    def serialize_payload(self) -> bytes:
        return (
            bytes(
                [
                    self.compression_scheme,
                    self.encryption_scheme,
                    self.secure_boot_scheme,
                    self.reserved,
                ]
            )
            + self.sign_block_size.to_bytes(2, "little")
            + self.num_blocks.to_bytes(2, "little")
            + self.nonce
            + self.final_image_hash
            + self.secure_boot_signature
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL4MemorySectionBlob(_GBL4OpaqueData):
    tag_id: int = GBL4TagId.MEMORY_SECTION_BLOB


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL4SeBlob(_GBL4OpaqueData):
    """A Secure Element upgrade blob, itself in an `A617EB` family container."""

    tag_id: int = GBL4TagId.SE_BLOB


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL4Pad(_GBL4OpaqueData):
    """Alignment padding. Its id has the container bit set, but it holds raw bytes."""

    tag_id: int = GBL4TagId.PAD


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL4UnknownTag(GBL4TagBase):
    tag_id: int
    payload: bytes

    def serialize_payload(self) -> bytes:
        return self.payload


GBL4Tag = Union[
    GBL4Root,
    GBL4Manifest,
    GBL4UpdateProcess,
    GBL4MemorySection,
    GBL4Certificate,
    GBL4Signature,
    GBL4ManifestInfo,
    GBL4BundleVersion,
    GBL4ContentHash,
    GBL4UpdateSe,
    GBL4UpdateMemorySection,
    GBL4ManifestFinish,
    GBL4MemorySectionInfo,
    GBL4MemorySectionBlob,
    GBL4SeBlob,
    GBL4Pad,
    GBL4UnknownTag,
]

GBL4_TAG_CLASSES: dict[GBL4TagId, Any] = {
    GBL4TagId.V4: GBL4Root,
    GBL4TagId.MANIFEST: GBL4Manifest,
    GBL4TagId.UPDATE_PROCESS: GBL4UpdateProcess,
    GBL4TagId.MEMORY_SECTION: GBL4MemorySection,
    GBL4TagId.CERTIFICATE: GBL4Certificate,
    GBL4TagId.SIGNATURE: GBL4Signature,
    GBL4TagId.MANIFEST_INFO: GBL4ManifestInfo,
    GBL4TagId.BUNDLE_VERSION: GBL4BundleVersion,
    GBL4TagId.CONTENT_HASH: GBL4ContentHash,
    GBL4TagId.UPDATE_SE: GBL4UpdateSe,
    GBL4TagId.UPDATE_MEMORY_SECTION: GBL4UpdateMemorySection,
    GBL4TagId.MANIFEST_FINISH: GBL4ManifestFinish,
    GBL4TagId.MEMORY_SECTION_INFO: GBL4MemorySectionInfo,
    GBL4TagId.MEMORY_SECTION_BLOB: GBL4MemorySectionBlob,
    GBL4TagId.SE_BLOB: GBL4SeBlob,
    GBL4TagId.PAD: GBL4Pad,
}


def serialize_tag(tag: GBL4TagBase) -> bytes:
    payload = tag.serialize_payload()

    return (
        int(tag.tag_id).to_bytes(4, "little")
        + len(payload).to_bytes(4, "little")
        + payload
    )


def parse_tag_stream(data: bytes) -> list[GBL4TagBase]:
    """Parse a run of tags to exhaustion. There is no end tag in v4."""
    tags: list[GBL4TagBase] = []
    offset = 0

    while offset < len(data):
        if offset + 8 > len(data):
            raise ParseError("GBLv4 image is truncated: incomplete tag header")

        tag_id = int.from_bytes(data[offset : offset + 4], "little")
        length = int.from_bytes(data[offset + 4 : offset + 8], "little")
        start = offset + 8
        end = start + length

        if end > len(data):
            raise ParseError(
                f"GBLv4 tag {tag_id:#010x} is truncated: expected {length} bytes,"
                f" got {len(data) - start}"
            )

        tags.append(_parse_tag(tag_id, data[start:end]))
        offset = end

    return tags


def _parse_tag(tag_id: int, payload: bytes) -> GBL4TagBase:
    try:
        known = GBL4TagId(tag_id)
    except ValueError:
        return GBL4UnknownTag(tag_id=tag_id, payload=payload)

    parsed: GBL4TagBase = GBL4_TAG_CLASSES[known].from_payload(payload)

    return parsed


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL4Image:
    """A GBLv4 file: one root tag, plus anything following it."""

    root: GBL4Root

    @classmethod
    def deserialize(cls, data: bytes) -> tuple[Self, bytes]:
        """Parse an image, returning it and whatever followed the root tag."""
        data = bytes(data)

        if not data.startswith(GBL4_MAGIC):
            raise ParseError("Data is not a GBLv4 image")

        length = int.from_bytes(data[4:8], "little")
        end = 8 + length

        if end > len(data):
            raise ParseError(
                f"GBLv4 root tag is truncated: expected {length} bytes,"
                f" got {len(data) - 8}"
            )

        return cls(root=GBL4Root.from_payload(data[8:end])), data[end:]

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        """Parse an image, discarding anything that followed the root tag."""
        return cls.deserialize(data)[0]

    def serialize(self) -> bytes:
        return serialize_tag(self.root)

    def get_tags(self, tag_type: type[T], *, allow_missing: bool = False) -> list[T]:
        """Every tag of a type, at any depth."""
        found: list[T] = []

        def walk(tags: list[GBL4TagBase]) -> None:
            for tag in tags:
                if type(tag) is tag_type:
                    found.append(tag)

                if isinstance(tag, _GBL4Container):
                    walk(tag.children)

        walk([self.root])

        if not found and not allow_missing:
            raise KeyError(f"No {tag_type.__name__} tag exists")

        return found

    def get_first_tag(self, tag_type: type[T]) -> T:
        return self.get_tags(tag_type)[0]

    def root_child_offsets(self) -> list[tuple[GBL4TagBase, int]]:
        """Each top-level tag with its offset from the start of the image.

        The manifest addresses payloads by absolute file offset, so the offsets have to
        include the root tag's own 8 byte header.
        """
        offsets: list[tuple[GBL4TagBase, int]] = []
        offset = 8

        for child in self.root.children:
            offsets.append((child, offset))
            offset += 8 + len(child.serialize_payload())

        return offsets

    def memory_section_at(self, position: int) -> GBL4MemorySection:
        """The memory section a `memory_section_position` points at."""
        for child, offset in self.root_child_offsets():
            if offset == position and isinstance(child, GBL4MemorySection):
                return child

        raise KeyError(f"No memory section at offset {position:#x}")

    def content(self) -> bytes:
        """The bytes `CONTENT_HASH` covers: everything after the manifest tag."""
        children = self.root.children

        for index, child in enumerate(children):
            if isinstance(child, GBL4Manifest):
                return b"".join(serialize_tag(t) for t in children[index + 1 :])

        raise KeyError("Image has no manifest")

    def _manifest(self) -> GBL4Manifest:
        """The manifest the checks read from, which has to be the image's first tag."""
        children = self.root.children

        if not children or not isinstance(children[0], GBL4Manifest):
            raise ValidationError("Image does not start with a manifest")

        return children[0]

    def _signed_children(self) -> list[GBL4TagBase]:
        """The manifest tags after `SIGNATURE`, or all of them if the image is unsigned."""
        children = self._manifest().children

        for index, child in enumerate(children):
            if isinstance(child, GBL4Signature):
                return children[index + 1 :]

        return children

    def _signed_tag(self, tag_type: type[T]) -> T:
        """The one tag of a type, which has to sit in the signed part of the manifest.

        The certificate sits before the signature, outside the signed range, so its slot
        can hold forged tags plus padding without moving anything. A tag is therefore only
        taken from the signed range, and a second copy anywhere in the image is refused.
        """
        tags = self.get_tags(tag_type)

        if len(tags) > 1:
            raise ValidationError(f"Image has {len(tags)} {tag_type.__name__} tags")

        if not any(child is tags[0] for child in self._signed_children()):
            raise ValidationError(
                f"{tag_type.__name__} is outside the signed part of the manifest"
            )

        return tags[0]

    def verify_content_hash(self) -> bool:
        """Whether the payload following the manifest hashes to `CONTENT_HASH`."""
        return self._signed_tag(GBL4ContentHash).hash.matches(self.content())

    def verify_memory_sections(self) -> bool:
        """Whether every update in the manifest matches the section it points at.

        Each `UPDATE_MEMORY_SECTION` carries the digest of a whole `MEMORY_SECTION_INFO`
        tag, header included, which in turn carries the digest of the image the section
        expands to. The second half can only be checked when the section is one this
        library can expand, so an encrypted section is verified as far as its info tag.
        A section that does not expand to the update's `plain_image_size`, or that uses a
        compression scheme this library does not know, fails.
        """
        process = self._signed_tag(GBL4UpdateProcess)
        updates = [
            c for c in process.children if isinstance(c, GBL4UpdateMemorySection)
        ]

        if len(self.get_tags(GBL4UpdateMemorySection, allow_missing=True)) != len(
            updates
        ):
            raise ValidationError("Image has an update outside the update process")

        for update in updates:
            try:
                section = self.memory_section_at(update.memory_section_position)
            except KeyError:
                return False

            if not update.hash.matches(serialize_tag(section.info)):
                return False

            if section.info.encryption_scheme != GBL4EncryptionScheme.NONE:
                continue

            try:
                plain = self.plain_image(section, max_size=update.plain_image_size)
            except ValidationError:
                return False

            if len(plain) != update.plain_image_size:
                return False

            # A fixed 64 byte field holding a 32 byte digest, zero padded
            if hashlib.sha256(plain).digest() != section.info.final_image_hash[:32]:
                return False

        return True

    def plain_image(
        self, section: GBL4MemorySection, *, max_size: int = GBL4_MAX_PLAIN_IMAGE_SIZE
    ) -> bytes:
        """Expand a memory section into the image the device will flash.

        Decompression stops at `max_size`, so a small blob cannot expand without bound.
        The default is the largest size an update's 24 bit `plain_image_size` can give.
        """
        info = section.info

        if info.encryption_scheme != GBL4EncryptionScheme.NONE:
            raise ValidationError(
                f"Memory section is encrypted with scheme {info.encryption_scheme:#x},"
                f" which cannot be decrypted without the device's key"
            )

        data = section.blob.data

        if info.compression_scheme == GBL4CompressionScheme.NONE:
            return data

        if info.compression_scheme not in (
            GBL4CompressionScheme.LZMA,
            GBL4CompressionScheme.LZ4,
        ):
            raise ValidationError(
                f"Unsupported compression scheme {info.compression_scheme:#x}"
            )

        try:
            if info.compression_scheme == GBL4CompressionScheme.LZMA:
                return lzma_decompress(data, max_size=max_size)

            return lz4_decompress(data, max_size=max_size)
        except DECOMPRESSION_ERRORS as exc:
            raise ValidationError(
                f"Memory section does not expand to at most {max_size} bytes: {exc}"
            ) from exc

    def signing_content(self) -> bytes:
        """The manifest tags an `ECDSA_P256` signature covers.

        Everything in the manifest after the signature tag itself, which is the manifest
        info, the bundle version, the content hash and the update process. The
        certificate sits before the signature and is not part of the signed range.
        """
        children = self._manifest().children

        for index, child in enumerate(children):
            if isinstance(child, GBL4Signature):
                return b"".join(serialize_tag(t) for t in children[index + 1 :])

        raise KeyError("Image is not signed")

    def signing_digest(self) -> bytes:
        return hashlib.sha256(self.signing_content()).digest()

    def verify_signature(self, public_key: ec.EllipticCurvePublicKey) -> bool:
        """Whether the manifest signature is valid under `public_key`.

        The image's own `CERTIFICATE` tag is not used: anyone who can change the
        manifest can also replace the certificate and re-sign, so only a key obtained
        elsewhere shows who signed the image.
        """
        signatures = self.get_tags(GBL4Signature, allow_missing=True)

        if not signatures:
            raise KeyError("Image is not signed")

        if len(signatures) > 1:
            raise ValidationError(f"Image has {len(signatures)} GBL4Signature tags")

        # Only the manifest's own signature matches the range `signing_content` covers
        if not any(child is signatures[0] for child in self._manifest().children):
            raise ValidationError("GBL4Signature is outside the manifest")

        signature = signatures[0].signature

        if signature.hash_type != GBL4SignatureType.ECDSA_P256:
            raise ValidationError(
                f"Unsupported signature type {signature.hash_type:#x}"
            )

        if len(signature.value) != 64:
            raise ValidationError(
                f"ECDSA P-256 signature must be 64 bytes, got {len(signature.value)}"
            )

        return verify_digest(
            public_key,
            self.signing_digest(),
            signature.value[:32],
            signature.value[32:],
        )
