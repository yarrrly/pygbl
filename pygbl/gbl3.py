from __future__ import annotations

from collections.abc import Callable
import dataclasses
import enum
import hashlib
from typing import TYPE_CHECKING, Any, ClassVar, Self, Union
import zlib

from pygbl.compression import (
    lz4_compress,
    lz4_decompress,
    lzma_compress,
    lzma_decompress,
)
from pygbl.crypto import aes_ctr_crypt, generate_nonce, sign_digest, verify_digest
from pygbl.types import (
    VALID_CRC32,
    ParseError,
    T,
    ValidationError,
    check_length,
    pad_to_multiple,
)

if TYPE_CHECKING:
    # Only needed for annotations: signing is an optional feature
    from cryptography.hazmat.primitives.asymmetric import ec

GBL3_MAGIC = b"\xeb\x17\xa6\x03"

# `commander` pads its output up to a word boundary with 0xFF
GBL3_BLOCK_SIZE = 4


class GBL3TagId(enum.IntEnum):
    """GBL tag identifiers, as decoded from the little-endian 32-bit value on the wire."""

    # First tag in the file. The header tag contains the version number of the GBL file
    # specification, and flags indicating the type of GBL file – whether it is signed
    # or encrypted.
    HEADER = 0x03A617EB
    # Information about the application update image that is contained in this GBL file.
    APP_INFO = 0xF40A0AF4
    # A complete encrypted Secure Element update image. Only applicable on Series 2
    # devices.
    SE_UPGRADE = 0x5EA617EB
    # A complete bootloader update image.
    BOOTLOADER = 0xF50909F5
    # Application data to program at a specific address in main flash. The SDK names
    # these "prog" and "erase&program", but its parser dispatches both to the same
    # handler, so they behave identically. `commander` emits ERASEPROG by default.
    PROG = 0xFE0101FE
    ERASEPROG = 0xFD0303FD
    # The same, LZ4 or LZMA compressed.
    PROG_LZ4 = 0xFD0505FD
    PROG_LZMA = 0xFD0707FD
    # Delta DFU: a patch against an image already on the device rather than a whole one.
    DELTA = 0xF80A0AF8
    DELTA_LZ4 = 0xF80B0BF8
    DELTA_LZMA = 0xF80C0CF8
    # Constraints on the versions already installed, e.g. "only apply me if the running
    # bootloader is at least 2.4".
    VERSION_DEPENDENCY = 0x76A617EB
    # Metadata that the bootloader does not parse, but can be returned to the
    # application through a callback.
    METADATA = 0xF60808F6
    # The ECDSA-P256 signature of all preceding data in the file.
    SIGNATURE = 0xF70A0AF7
    # A public key plus its own signature, letting the bootloader verify the image
    # through a certificate chain instead of against a directly trusted key.
    CERTIFICATE_ECDSA_P256 = 0xF30B0BF3
    # Encrypted counterparts of the above, used when the GBL is AES-CCM encrypted. The
    # plaintext tags are wrapped inside `ENC_GBL_DATA`.
    ENC_HEADER = 0xFB0505FB
    ENC_INIT = 0xFA0606FA
    ENC_GBL_DATA = 0xF90707F9
    # End of the GBL file. It contains a 32-bit CRC for the entire file as an integrity
    # check. The CRC is a non-cryptographic check. This must be the last tag.
    END = 0xFC0404FC


class GBL3Type(enum.IntFlag, boundary=enum.KEEP):
    NONE = 0x00000000
    ENCRYPTION_AESCCM = 0x00000001
    SIGNATURE_ECDSA_P256 = 0x00000100


class GBL3Compression(enum.Enum):
    LZMA = "lzma"
    LZ4 = "lz4"


class GBL3ApplicationType(enum.IntFlag, boundary=enum.KEEP):
    ZIGBEE = 0x01
    THREAD = 0x02
    FLEX = 0x04
    BLUETOOTH = 0x08
    MCU = 0x10
    BLUETOOTH_APP = 0x20
    BOOTLOADER = 0x40
    ZWAVE = 0x80


@dataclasses.dataclass(frozen=True, order=True)
class BootloaderVersion:
    major: int
    minor: int
    customer: int

    @classmethod
    def from_int(cls, value: int) -> BootloaderVersion:
        return cls(
            major=(value >> 24) & 0xFF,
            minor=(value >> 16) & 0xFF,
            customer=value & 0xFFFF,
        )

    def as_int(self) -> int:
        return (self.major << 24) | (self.minor << 16) | self.customer

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.customer}"


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL3TagBase:
    """Everything the container needs from a tag: an identifier and a payload encoder.

    Concrete tags default `tag_id` to their own identifier; unrecognized tags carry
    theirs as ordinary data.
    """

    tag_id: int

    def serialize_payload(self) -> bytes:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL3Header(GBL3TagBase):
    tag_id: int = GBL3TagId.HEADER

    # The spec encodes GBL v3 as `0x03000000`. It is read big-endian so that the value
    # matches the human-readable specification version.
    version: int
    type: GBL3Type

    @classmethod
    def from_payload(cls, payload: bytes) -> GBL3Header:
        check_length(payload, 8, "GBL header")

        return cls(
            version=int.from_bytes(payload[0:4], "big"),
            type=GBL3Type(int.from_bytes(payload[4:8], "little")),
        )

    def serialize_payload(self) -> bytes:
        return self.version.to_bytes(4, "big") + int(self.type).to_bytes(4, "little")


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL3ApplicationInfo(GBL3TagBase):
    tag_id: int = GBL3TagId.APP_INFO

    type: GBL3ApplicationType
    version: int
    capabilities: int
    product_id: bytes

    @classmethod
    def from_payload(cls, payload: bytes) -> GBL3ApplicationInfo:
        check_length(payload, 28, "GBL application info")

        return cls(
            type=GBL3ApplicationType(int.from_bytes(payload[0:4], "little")),
            version=int.from_bytes(payload[4:8], "little"),
            capabilities=int.from_bytes(payload[8:12], "little"),
            product_id=payload[12:28],
        )

    def serialize_payload(self) -> bytes:
        check_length(self.product_id, 16, "GBL product id")

        return (
            int(self.type).to_bytes(4, "little")
            + self.version.to_bytes(4, "little")
            + self.capabilities.to_bytes(4, "little")
            + self.product_id
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL3SeUpgrade(GBL3TagBase):
    tag_id: int = GBL3TagId.SE_UPGRADE

    blob_size: int
    version: int
    data: bytes

    @classmethod
    def from_payload(cls, payload: bytes) -> GBL3SeUpgrade:
        if len(payload) < 8:
            raise ParseError(f"GBL SE upgrade is truncated: {len(payload)} bytes")

        return cls(
            blob_size=int.from_bytes(payload[0:4], "little"),
            version=int.from_bytes(payload[4:8], "little"),
            data=payload[8:],
        )

    def serialize_payload(self) -> bytes:
        return (
            self.blob_size.to_bytes(4, "little")
            + self.version.to_bytes(4, "little")
            + self.data
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL3Bootloader(GBL3TagBase):
    tag_id: int = GBL3TagId.BOOTLOADER

    version: BootloaderVersion
    address: int
    data: bytes

    @classmethod
    def from_payload(cls, payload: bytes) -> GBL3Bootloader:
        if len(payload) < 8:
            raise ParseError(f"GBL bootloader is truncated: {len(payload)} bytes")

        return cls(
            version=BootloaderVersion.from_int(int.from_bytes(payload[0:4], "little")),
            address=int.from_bytes(payload[4:8], "little"),
            data=payload[8:],
        )

    def serialize_payload(self) -> bytes:
        return (
            self.version.as_int().to_bytes(4, "little")
            + self.address.to_bytes(4, "little")
            + self.data
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class _GBL3AddressedData(GBL3TagBase):
    address: int
    data: bytes

    @classmethod
    def from_payload(cls, payload: bytes) -> Self:
        if len(payload) < 4:
            raise ParseError(f"{cls.__name__} is truncated: {len(payload)} bytes")

        return cls(
            tag_id=cls.tag_id,
            address=int.from_bytes(payload[0:4], "little"),
            data=payload[4:],
        )

    def serialize_payload(self) -> bytes:
        return self.address.to_bytes(4, "little") + self.data


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL3Prog(_GBL3AddressedData):
    tag_id: int = GBL3TagId.PROG


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL3EraseProg(_GBL3AddressedData):
    tag_id: int = GBL3TagId.ERASEPROG


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL3ProgLZ4(_GBL3AddressedData):
    tag_id: int = GBL3TagId.PROG_LZ4


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL3ProgLZMA(_GBL3AddressedData):
    tag_id: int = GBL3TagId.PROG_LZMA


@dataclasses.dataclass(frozen=True, kw_only=True)
class _GBL3DeltaData(GBL3TagBase):
    """A delta DFU patch: the target image's CRC and size, then the patch itself.

    The patch stays opaque here because applying it needs the firmware already on the
    device. It is Silicon Labs' own `ddfu` format: a `0D E1 7A` magic and a version
    byte, then variable-length instructions that insert literal bytes from the patch,
    copy a run from the old image, seek the source or output cursor, or copy a run and
    add per-byte corrections to it. Both ends are gated, which is why these tags do not
    appear in the wild: producing one needs Silicon Labs' unpublished `ddfu_diff`
    library, and a bootloader only recognizes the tags when built with the
    `bootloader_gbl_delta_dfu` component, which is not enabled by default.
    """

    new_fw_crc: int
    new_fw_size: int
    address: int
    data: bytes

    @classmethod
    def from_payload(cls, payload: bytes) -> Self:
        if len(payload) < 12:
            raise ParseError(f"{cls.__name__} is truncated: {len(payload)} bytes")

        return cls(
            tag_id=cls.tag_id,
            new_fw_crc=int.from_bytes(payload[0:4], "little"),
            new_fw_size=int.from_bytes(payload[4:8], "little"),
            address=int.from_bytes(payload[8:12], "little"),
            data=payload[12:],
        )

    def serialize_payload(self) -> bytes:
        return (
            self.new_fw_crc.to_bytes(4, "little")
            + self.new_fw_size.to_bytes(4, "little")
            + self.address.to_bytes(4, "little")
            + self.data
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL3Delta(_GBL3DeltaData):
    tag_id: int = GBL3TagId.DELTA


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL3DeltaLZ4(_GBL3DeltaData):
    tag_id: int = GBL3TagId.DELTA_LZ4


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL3DeltaLZMA(_GBL3DeltaData):
    tag_id: int = GBL3TagId.DELTA_LZMA


class GBL3VersionDependencyImageType(enum.IntEnum):
    APPLICATION = 0x01
    BOOTLOADER = 0x02
    SE = 0x03


@dataclasses.dataclass(frozen=True, kw_only=True)
class VersionDependencyStatement:
    """One comparison, e.g. `appVersion > 1.2.3`.

    `statement` packs the comparison operator in its low nibble and the connective
    joining it to the next statement in its high nibble.
    """

    image_type: int
    statement: int
    reserved: int = 0
    version: int

    SIZE: ClassVar[int] = 8

    @classmethod
    def from_bytes(cls, data: bytes) -> VersionDependencyStatement:
        check_length(data, cls.SIZE, "GBL version dependency statement")

        return cls(
            image_type=data[0],
            statement=data[1],
            reserved=int.from_bytes(data[2:4], "little"),
            version=int.from_bytes(data[4:8], "little"),
        )

    def serialize(self) -> bytes:
        return (
            bytes([self.image_type, self.statement])
            + self.reserved.to_bytes(2, "little")
            + self.version.to_bytes(4, "little")
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL3VersionDependency(GBL3TagBase):
    tag_id: int = GBL3TagId.VERSION_DEPENDENCY

    statements: list[VersionDependencyStatement]

    @classmethod
    def from_payload(cls, payload: bytes) -> GBL3VersionDependency:
        size = VersionDependencyStatement.SIZE

        if len(payload) % size != 0:
            raise ParseError(
                f"GBL version dependency must be a multiple of {size} bytes,"
                f" got {len(payload)}"
            )

        return cls(
            statements=[
                VersionDependencyStatement.from_bytes(payload[offset : offset + size])
                for offset in range(0, len(payload), size)
            ]
        )

    def serialize_payload(self) -> bytes:
        return b"".join(statement.serialize() for statement in self.statements)


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL3Metadata(GBL3TagBase):
    tag_id: int = GBL3TagId.METADATA

    metadata: bytes

    @classmethod
    def from_payload(cls, payload: bytes) -> GBL3Metadata:
        return cls(metadata=payload)

    def serialize_payload(self) -> bytes:
        return self.metadata


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL3Signature(GBL3TagBase):
    tag_id: int = GBL3TagId.SIGNATURE

    r: bytes
    s: bytes

    @classmethod
    def from_payload(cls, payload: bytes) -> GBL3Signature:
        check_length(payload, 64, "GBL ECDSA-P256 signature")

        return cls(r=payload[0:32], s=payload[32:64])

    def serialize_payload(self) -> bytes:
        check_length(self.r, 32, "GBL signature r")
        check_length(self.s, 32, "GBL signature s")

        return self.r + self.s


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL3CertificateEcdsaP256(GBL3TagBase):
    """The SDK's `ApplicationCertificate_t`, wrapped in a tag."""

    tag_id: int = GBL3TagId.CERTIFICATE_ECDSA_P256

    struct_version: int
    flags: bytes
    key: bytes
    version: int
    signature: bytes

    @classmethod
    def from_payload(cls, payload: bytes) -> GBL3CertificateEcdsaP256:
        check_length(payload, 136, "GBL ECDSA-P256 certificate")

        return cls(
            struct_version=payload[0],
            flags=payload[1:4],
            key=payload[4:68],
            version=int.from_bytes(payload[68:72], "little"),
            signature=payload[72:136],
        )

    def serialize_payload(self) -> bytes:
        check_length(self.flags, 3, "GBL certificate flags")
        check_length(self.key, 64, "GBL certificate key")
        check_length(self.signature, 64, "GBL certificate signature")

        return (
            bytes([self.struct_version])
            + self.flags
            + self.key
            + self.version.to_bytes(4, "little")
            + self.signature
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL3EncryptionHeader(GBL3TagBase):
    tag_id: int = GBL3TagId.ENC_HEADER

    data: bytes

    @classmethod
    def from_payload(cls, payload: bytes) -> GBL3EncryptionHeader:
        return cls(tag_id=cls.tag_id, data=payload)

    def serialize_payload(self) -> bytes:
        return self.data


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL3EncryptionInitAesCcm(GBL3TagBase):
    tag_id: int = GBL3TagId.ENC_INIT

    msg_len: int
    nonce: bytes

    @classmethod
    def from_payload(cls, payload: bytes) -> GBL3EncryptionInitAesCcm:
        check_length(payload, 16, "GBL AES-CCM encryption init")

        return cls(
            msg_len=int.from_bytes(payload[0:4], "little"),
            nonce=payload[4:16],
        )

    def serialize_payload(self) -> bytes:
        check_length(self.nonce, 12, "GBL AES-CCM nonce")

        return self.msg_len.to_bytes(4, "little") + self.nonce


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL3EncryptedData(GBL3TagBase):
    """AES-CCM ciphertext wrapping the plaintext tags. Not decrypted by this library."""

    tag_id: int = GBL3TagId.ENC_GBL_DATA

    ciphertext: bytes

    @classmethod
    def from_payload(cls, payload: bytes) -> GBL3EncryptedData:
        return cls(ciphertext=payload)

    def serialize_payload(self) -> bytes:
        return self.ciphertext


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL3End(GBL3TagBase):
    tag_id: int = GBL3TagId.END

    crc: int

    @classmethod
    def from_payload(cls, payload: bytes) -> GBL3End:
        check_length(payload, 4, "GBL end tag")

        return cls(crc=int.from_bytes(payload[0:4], "little"))

    def serialize_payload(self) -> bytes:
        return self.crc.to_bytes(4, "little")


@dataclasses.dataclass(frozen=True, kw_only=True)
class GBL3UnknownTag(GBL3TagBase):
    """A tag whose identifier is not recognized. Preserved verbatim."""

    tag_id: int
    payload: bytes

    def serialize_payload(self) -> bytes:
        return self.payload


GBL3Tag = Union[
    GBL3Header,
    GBL3ApplicationInfo,
    GBL3SeUpgrade,
    GBL3Bootloader,
    GBL3Prog,
    GBL3EraseProg,
    GBL3ProgLZ4,
    GBL3ProgLZMA,
    GBL3Delta,
    GBL3DeltaLZ4,
    GBL3DeltaLZMA,
    GBL3VersionDependency,
    GBL3Metadata,
    GBL3Signature,
    GBL3CertificateEcdsaP256,
    GBL3EncryptionHeader,
    GBL3EncryptionInitAesCcm,
    GBL3EncryptedData,
    GBL3End,
    GBL3UnknownTag,
]

GBL3_TAG_CLASSES: dict[GBL3TagId, Any] = {
    GBL3TagId.HEADER: GBL3Header,
    GBL3TagId.APP_INFO: GBL3ApplicationInfo,
    GBL3TagId.SE_UPGRADE: GBL3SeUpgrade,
    GBL3TagId.BOOTLOADER: GBL3Bootloader,
    GBL3TagId.PROG: GBL3Prog,
    GBL3TagId.ERASEPROG: GBL3EraseProg,
    GBL3TagId.PROG_LZ4: GBL3ProgLZ4,
    GBL3TagId.PROG_LZMA: GBL3ProgLZMA,
    GBL3TagId.DELTA: GBL3Delta,
    GBL3TagId.DELTA_LZ4: GBL3DeltaLZ4,
    GBL3TagId.DELTA_LZMA: GBL3DeltaLZMA,
    GBL3TagId.VERSION_DEPENDENCY: GBL3VersionDependency,
    GBL3TagId.METADATA: GBL3Metadata,
    GBL3TagId.SIGNATURE: GBL3Signature,
    GBL3TagId.CERTIFICATE_ECDSA_P256: GBL3CertificateEcdsaP256,
    GBL3TagId.ENC_HEADER: GBL3EncryptionHeader,
    GBL3TagId.ENC_INIT: GBL3EncryptionInitAesCcm,
    GBL3TagId.ENC_GBL_DATA: GBL3EncryptedData,
    GBL3TagId.END: GBL3End,
}

PROGRAM_DATA_TAGS = (
    GBL3Prog,
    GBL3EraseProg,
    GBL3ProgLZ4,
    GBL3ProgLZMA,
)


def _compress_lzma(address: int, data: bytes) -> GBL3ProgLZMA:
    return GBL3ProgLZMA(address=address, data=lzma_compress(data))


def _compress_lz4(address: int, data: bytes) -> GBL3ProgLZ4:
    return GBL3ProgLZ4(address=address, data=lz4_compress(data))


# Keyed by algorithm so that adding a `GBL3Compression` member without a compressor fails
# loudly rather than silently leaving program data uncompressed.
COMPRESSORS: dict[GBL3Compression, Callable[[int, bytes], GBL3TagBase]] = {
    GBL3Compression.LZMA: _compress_lzma,
    GBL3Compression.LZ4: _compress_lz4,
}


def serialize_tag(tag: GBL3TagBase) -> bytes:
    """Serialize a single tag, header included, exactly as it appears on the wire."""
    payload = tag.serialize_payload()

    return (
        int(tag.tag_id).to_bytes(4, "little")
        + len(payload).to_bytes(4, "little")
        + payload
    )


def parse_tag_stream(
    data: bytes, *, stop_at_end: bool
) -> tuple[list[GBL3TagBase], int]:
    """Parse a sequence of tags, returning them and the offset just past the last one.

    Encrypted images wrap a bare tag stream with no end tag, so parsing has to be able
    to run to exhaustion as well as stop at the end tag.
    """
    tags: list[GBL3TagBase] = []
    offset = 0

    while offset < len(data):
        if offset + 8 > len(data):
            raise ParseError("Image is truncated: incomplete tag header")

        tag_id = int.from_bytes(data[offset : offset + 4], "little")
        length = int.from_bytes(data[offset + 4 : offset + 8], "little")
        start = offset + 8
        end = start + length

        if end > len(data):
            raise ParseError(
                f"Tag {tag_id:#010x} is truncated: expected {length} bytes,"
                f" got {len(data) - start}"
            )

        tags.append(_parse_tag(tag_id, data[start:end]))
        offset = end

        # A GBL ends at the first end tag. Everything after it is trailing data.
        if stop_at_end and tag_id == GBL3TagId.END:
            return tags, offset

    if stop_at_end:
        raise ParseError("Image is truncated: no end tag found")

    return tags, offset


@dataclasses.dataclass(frozen=True)
class GBL3Image:
    tags: list[GBL3TagBase]

    @classmethod
    def deserialize(cls, data: bytes, *, validate: bool = True) -> tuple[Self, bytes]:
        """Parse an image, returning it and whatever followed its end tag.

        Real-world files carry `commander`'s word padding, vendor junk or even a second
        firmware payload after the end tag. None of it is covered by the CRC, so what to
        do with it is the caller's decision.
        """
        data = bytes(data)
        tags, offset = parse_tag_stream(data, stop_at_end=True)
        image = cls(tags=tags)

        if validate:
            image.validate()

        return image, data[offset:]

    @classmethod
    def from_bytes(cls, data: bytes, *, validate: bool = True) -> Self:
        """Parse an image, discarding anything that followed its end tag."""
        return cls.deserialize(data, validate=validate)[0]

    def serialize_tags(self) -> bytes:
        """Serialize only the tag region, which is what the CRC covers."""
        return b"".join(serialize_tag(t) for t in self.tags)

    def serialize(
        self, *, block_size: int = GBL3_BLOCK_SIZE, padding: bytes = b"\xff"
    ) -> bytes:
        return pad_to_multiple(self.serialize_tags(), block_size, padding)

    def validate(self) -> None:
        if not self.tags:
            raise ValidationError("Image contains no tags")

        if not isinstance(self.tags[0], GBL3Header):
            raise ValidationError("First tag must be the header")

        if not isinstance(self.tags[-1], GBL3End):
            raise ValidationError("Last tag must be the end tag")

        crc = zlib.crc32(self.serialize_tags())

        if crc != VALID_CRC32:
            raise ValidationError(
                f"Image CRC-32 is invalid: expected {VALID_CRC32:#010x}, got {crc:#010x}"
            )

    def get_tags(self, tag_type: type[T], *, allow_missing: bool = False) -> list[T]:
        tags: list[T] = [t for t in self.tags if type(t) is tag_type]

        if not tags and not allow_missing:
            raise KeyError(f"No {tag_type.__name__} tag exists")

        return tags

    def get_first_tag(self, tag_type: type[T]) -> T:
        return self.get_tags(tag_type)[0]

    def get_metadata(self) -> bytes | None:
        """The metadata tag contents, if the image has one. Opaque to the bootloader."""
        metadata = self.get_tags(GBL3Metadata, allow_missing=True)

        return metadata[0].metadata if metadata else None

    def regenerate_crc(self) -> GBL3Image:
        tags = [t for t in self.tags if not isinstance(t, GBL3End)]
        placeholder = type(self)(tags=[*tags, GBL3End(crc=0)])
        crc = zlib.crc32(placeholder.serialize_tags()[:-4]) & 0xFFFFFFFF

        return type(self)(tags=[*tags, GBL3End(crc=crc)])

    def signing_digest(self) -> bytes:
        """The SHA-256 digest an ECDSA signature covers.

        The bootloader hashes every tag as it appears on the wire, before decryption,
        skipping only the signature and end tags.
        """
        return hashlib.sha256(
            b"".join(
                serialize_tag(t)
                for t in self.tags
                if not isinstance(t, (GBL3Signature, GBL3End))
            )
        ).digest()

    def sign(self, private_key: ec.EllipticCurvePrivateKey) -> GBL3Image:
        """Return a signed copy of this image.

        The header's type flags are part of the signed region, so they are updated
        before the digest is computed.
        """
        tags: list[GBL3TagBase] = [
            t for t in self.tags if not isinstance(t, (GBL3Signature, GBL3End))
        ]
        header = self.get_first_tag(GBL3Header)
        tags[tags.index(header)] = dataclasses.replace(
            header, type=header.type | GBL3Type.SIGNATURE_ECDSA_P256
        )

        unsigned = type(self)(tags=tags)
        r, s = sign_digest(private_key, unsigned.signing_digest())

        return type(self)(tags=[*tags, GBL3Signature(r=r, s=s)]).regenerate_crc()

    def verify_signature(self, public_key: ec.EllipticCurvePublicKey) -> bool:
        signatures = self.get_tags(GBL3Signature, allow_missing=True)

        if not signatures:
            raise KeyError("Image is not signed")

        signature = signatures[0]

        return verify_digest(
            public_key, self.signing_digest(), signature.r, signature.s
        )

    def encrypt(self, key: bytes, *, nonce: bytes | None = None) -> GBL3Image:
        """Return an AES-CTR encrypted copy of this image.

        Every tag except the header and end tag is wrapped into its own encrypted data
        tag. The keystream runs continuously across them, so they cannot be reordered.
        """
        if nonce is None:
            nonce = generate_nonce()

        header = self.get_first_tag(GBL3Header)
        plaintext_tags = [
            t
            for t in self.tags
            if not isinstance(t, (GBL3Header, GBL3End, GBL3Signature))
        ]

        if any(isinstance(t, GBL3EncryptedData) for t in plaintext_tags):
            raise ValueError("Image is already encrypted")

        chunks = [serialize_tag(t) for t in plaintext_tags]
        ciphertext = aes_ctr_crypt(key, nonce, b"".join(chunks))

        encrypted = []
        offset = 0

        for chunk in chunks:
            encrypted.append(
                GBL3EncryptedData(ciphertext=ciphertext[offset : offset + len(chunk)])
            )
            offset += len(chunk)

        return type(self)(
            tags=[
                dataclasses.replace(
                    header, type=header.type | GBL3Type.ENCRYPTION_AESCCM
                ),
                GBL3EncryptionInitAesCcm(msg_len=len(ciphertext), nonce=nonce),
                *encrypted,
            ],
        ).regenerate_crc()

    def decrypt(self, key: bytes) -> GBL3Image:
        """Return a decrypted copy of this image, dropping any signature.

        A signature covers the ciphertext, so it cannot carry over to the plaintext.
        """
        inits = self.get_tags(GBL3EncryptionInitAesCcm, allow_missing=True)

        if not inits:
            raise KeyError("Image is not encrypted")

        init = inits[0]
        ciphertext = b"".join(t.ciphertext for t in self.get_tags(GBL3EncryptedData))
        plaintext = aes_ctr_crypt(key, init.nonce, ciphertext)
        tags, _ = parse_tag_stream(plaintext, stop_at_end=False)
        header = self.get_first_tag(GBL3Header)

        return type(self)(
            tags=[
                dataclasses.replace(
                    header,
                    type=header.type
                    & ~(GBL3Type.ENCRYPTION_AESCCM | GBL3Type.SIGNATURE_ECDSA_P256),
                ),
                *tags,
            ],
        ).regenerate_crc()

    def compress(self, algorithm: GBL3Compression) -> GBL3Image:
        """Return a copy with every uncompressed program data tag compressed."""
        compress = COMPRESSORS[algorithm]
        tags: list[GBL3TagBase] = []

        for tag in self.tags:
            if isinstance(tag, (GBL3Prog, GBL3EraseProg)):
                tags.append(compress(tag.address, tag.data))
            else:
                tags.append(tag)

        return type(self)(tags=tags).regenerate_crc()

    def decompress(self) -> GBL3Image:
        """Return a copy with every compressed program data tag expanded."""
        tags: list[GBL3TagBase] = []

        for tag in self.tags:
            if isinstance(tag, GBL3ProgLZMA):
                tags.append(
                    GBL3EraseProg(address=tag.address, data=lzma_decompress(tag.data))
                )
            elif isinstance(tag, GBL3ProgLZ4):
                tags.append(
                    GBL3EraseProg(address=tag.address, data=lz4_decompress(tag.data))
                )
            else:
                tags.append(tag)

        return type(self)(tags=tags).regenerate_crc()

    def is_combined_bootloader_app(self) -> bool:
        app_infos = self.get_tags(GBL3ApplicationInfo, allow_missing=True)

        if not app_infos or not self.get_tags(GBL3Bootloader, allow_missing=True):
            return False

        app_info = app_infos[0]

        return bool(
            GBL3ApplicationType.BOOTLOADER in app_info.type
            and app_info.type & ~GBL3ApplicationType.BOOTLOADER
        )

    def split_bootloader_app(self) -> list[GBL3Image]:
        """Split a combined bootloader + application image into separate images."""
        if not self.is_combined_bootloader_app():
            return [self]

        header = self.get_first_tag(GBL3Header)

        if header.type != GBL3Type.NONE:
            raise ValueError("Cannot split signed or encrypted images")

        app_info = self.get_first_tag(GBL3ApplicationInfo)
        bootloader = type(self)(tags=[header, self.get_first_tag(GBL3Bootloader)])
        application = type(self)(
            tags=[
                header,
                dataclasses.replace(
                    app_info, type=app_info.type & ~GBL3ApplicationType.BOOTLOADER
                ),
                *[
                    t
                    for t in self.tags
                    if isinstance(t, (*PROGRAM_DATA_TAGS, GBL3Metadata))
                ],
            ]
        )

        return [bootloader.regenerate_crc(), application.regenerate_crc()]

    def combine_bootloader_app(self, other: GBL3Image) -> GBL3Image:
        """Combine a bootloader image and an application image into a single image."""
        ours = self.get_tags(GBL3Bootloader, allow_missing=True)
        theirs = other.get_tags(GBL3Bootloader, allow_missing=True)

        if ours and theirs:
            raise ValueError("Both images contain bootloaders")

        if not ours and not theirs:
            raise ValueError("Neither image contains a bootloader")

        header = self.get_first_tag(GBL3Header)

        if header != other.get_first_tag(GBL3Header):
            raise ValueError("GBL headers do not match")

        if header.type != GBL3Type.NONE:
            raise ValueError("Cannot combine signed or encrypted images")

        bootloader_image = self if ours else other
        application = other if bootloader_image is self else self
        app_info = application.get_first_tag(GBL3ApplicationInfo)

        return type(self)(
            tags=[
                header,
                dataclasses.replace(
                    app_info, type=app_info.type | GBL3ApplicationType.BOOTLOADER
                ),
                bootloader_image.get_first_tag(GBL3Bootloader),
                *[
                    t
                    for t in application.tags
                    if isinstance(t, (*PROGRAM_DATA_TAGS, GBL3Metadata))
                ],
            ]
        ).regenerate_crc()


def _parse_tag(tag_id: int, payload: bytes) -> GBL3TagBase:
    try:
        known_tag_id = GBL3TagId(tag_id)
    except ValueError:
        return GBL3UnknownTag(tag_id=tag_id, payload=payload)

    parsed: GBL3TagBase = GBL3_TAG_CLASSES[known_tag_id].from_payload(payload)

    return parsed
