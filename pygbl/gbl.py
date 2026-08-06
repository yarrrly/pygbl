from __future__ import annotations

import dataclasses
import enum
from typing import Any, ClassVar, Self, Union
import zlib

from pygbl.types import (
    VALID_CRC32,
    ParseError,
    T,
    ValidationError,
    check_length,
    pad_to_multiple,
)

GBL_MAGIC = b"\xeb\x17\xa6\x03"


class GBLTagId(enum.IntEnum):
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
    # Information about what application data to program at a specific address into the
    # main flash memory. The two tags are interchangeable.
    PROGRAM_DATA1 = 0xFE0101FE
    PROGRAM_DATA2 = 0xFD0303FD
    # LZ4 compressed information about what application data to program at a specific
    # address into the main flash memory.
    PROGRAM_DATA_LZ4 = 0xFD0505FD
    # LZMA compressed information about what application data to program at a specific
    # address into the main flash memory.
    PROGRAM_DATA_LZMA = 0xFD0707FD
    # Metadata that the bootloader does not parse, but can be returned to the
    # application through a callback.
    METADATA = 0xF60808F6
    # The ECDSA-P256 signature of all preceding data in the file.
    SIGNATURE = 0xF70A0AF7
    # Encrypted counterparts of the above, used when the GBL is AES-CCM encrypted. The
    # plaintext tags are wrapped inside `ENC_GBL_DATA`.
    ENC_HEADER = 0xFB0505FB
    ENC_INIT = 0xFA0606FA
    ENC_GBL_DATA = 0xF90707F9
    # End of the GBL file. It contains a 32-bit CRC for the entire file as an integrity
    # check. The CRC is a non-cryptographic check. This must be the last tag.
    END = 0xFC0404FC


class GBLType(enum.IntFlag, boundary=enum.KEEP):
    NONE = 0x00000000
    ENCRYPTION_AESCCM = 0x00000001
    SIGNATURE_ECDSA_P256 = 0x00000100


class GBLApplicationType(enum.IntFlag, boundary=enum.KEEP):
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


@dataclasses.dataclass(frozen=True)
class GBLHeader:
    TAG_ID: ClassVar[GBLTagId] = GBLTagId.HEADER

    # The spec encodes GBL v3 as `0x03000000`. It is read big-endian so that the value
    # matches the human-readable specification version.
    version: int
    type: GBLType

    @classmethod
    def from_payload(cls, payload: bytes) -> GBLHeader:
        check_length(payload, 8, "GBL header")

        return cls(
            version=int.from_bytes(payload[0:4], "big"),
            type=GBLType(int.from_bytes(payload[4:8], "little")),
        )

    def serialize_payload(self) -> bytes:
        return self.version.to_bytes(4, "big") + int(self.type).to_bytes(4, "little")


@dataclasses.dataclass(frozen=True)
class GBLApplicationInfo:
    TAG_ID: ClassVar[GBLTagId] = GBLTagId.APP_INFO

    type: GBLApplicationType
    version: int
    capabilities: int
    product_id: bytes

    @classmethod
    def from_payload(cls, payload: bytes) -> GBLApplicationInfo:
        check_length(payload, 28, "GBL application info")

        return cls(
            type=GBLApplicationType(int.from_bytes(payload[0:4], "little")),
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


@dataclasses.dataclass(frozen=True)
class GBLSeUpgrade:
    TAG_ID: ClassVar[GBLTagId] = GBLTagId.SE_UPGRADE

    blob_size: int
    version: int
    data: bytes

    @classmethod
    def from_payload(cls, payload: bytes) -> GBLSeUpgrade:
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


@dataclasses.dataclass(frozen=True)
class GBLBootloader:
    TAG_ID: ClassVar[GBLTagId] = GBLTagId.BOOTLOADER

    version: BootloaderVersion
    address: int
    data: bytes

    @classmethod
    def from_payload(cls, payload: bytes) -> GBLBootloader:
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


@dataclasses.dataclass(frozen=True)
class _GBLAddressedData:
    address: int
    data: bytes

    TAG_ID: ClassVar[GBLTagId]

    @classmethod
    def from_payload(cls, payload: bytes) -> Self:
        if len(payload) < 4:
            raise ParseError(f"{cls.__name__} is truncated: {len(payload)} bytes")

        return cls(
            address=int.from_bytes(payload[0:4], "little"),
            data=payload[4:],
        )

    def serialize_payload(self) -> bytes:
        return self.address.to_bytes(4, "little") + self.data


@dataclasses.dataclass(frozen=True)
class GBLProgramData1(_GBLAddressedData):
    TAG_ID: ClassVar[GBLTagId] = GBLTagId.PROGRAM_DATA1


@dataclasses.dataclass(frozen=True)
class GBLProgramData2(_GBLAddressedData):
    TAG_ID: ClassVar[GBLTagId] = GBLTagId.PROGRAM_DATA2


@dataclasses.dataclass(frozen=True)
class GBLProgramDataLZ4(_GBLAddressedData):
    TAG_ID: ClassVar[GBLTagId] = GBLTagId.PROGRAM_DATA_LZ4


@dataclasses.dataclass(frozen=True)
class GBLProgramDataLZMA(_GBLAddressedData):
    TAG_ID: ClassVar[GBLTagId] = GBLTagId.PROGRAM_DATA_LZMA


@dataclasses.dataclass(frozen=True)
class GBLMetadata:
    TAG_ID: ClassVar[GBLTagId] = GBLTagId.METADATA

    metadata: bytes

    @classmethod
    def from_payload(cls, payload: bytes) -> GBLMetadata:
        return cls(metadata=payload)

    def serialize_payload(self) -> bytes:
        return self.metadata


@dataclasses.dataclass(frozen=True)
class GBLSignature:
    TAG_ID: ClassVar[GBLTagId] = GBLTagId.SIGNATURE

    r: bytes
    s: bytes

    @classmethod
    def from_payload(cls, payload: bytes) -> GBLSignature:
        check_length(payload, 64, "GBL ECDSA-P256 signature")

        return cls(r=payload[0:32], s=payload[32:64])

    def serialize_payload(self) -> bytes:
        check_length(self.r, 32, "GBL signature r")
        check_length(self.s, 32, "GBL signature s")

        return self.r + self.s


@dataclasses.dataclass(frozen=True)
class GBLEncryptionHeader:
    TAG_ID: ClassVar[GBLTagId] = GBLTagId.ENC_HEADER

    data: bytes

    @classmethod
    def from_payload(cls, payload: bytes) -> GBLEncryptionHeader:
        return cls(data=payload)

    def serialize_payload(self) -> bytes:
        return self.data


@dataclasses.dataclass(frozen=True)
class GBLEncryptionInitAesCcm:
    TAG_ID: ClassVar[GBLTagId] = GBLTagId.ENC_INIT

    msg_len: int
    nonce: bytes

    @classmethod
    def from_payload(cls, payload: bytes) -> GBLEncryptionInitAesCcm:
        check_length(payload, 16, "GBL AES-CCM encryption init")

        return cls(
            msg_len=int.from_bytes(payload[0:4], "little"),
            nonce=payload[4:16],
        )

    def serialize_payload(self) -> bytes:
        check_length(self.nonce, 12, "GBL AES-CCM nonce")

        return self.msg_len.to_bytes(4, "little") + self.nonce


@dataclasses.dataclass(frozen=True)
class GBLEncryptedData:
    """AES-CCM ciphertext wrapping the plaintext tags. Not decrypted by this library."""

    TAG_ID: ClassVar[GBLTagId] = GBLTagId.ENC_GBL_DATA

    ciphertext: bytes

    @classmethod
    def from_payload(cls, payload: bytes) -> GBLEncryptedData:
        return cls(ciphertext=payload)

    def serialize_payload(self) -> bytes:
        return self.ciphertext


@dataclasses.dataclass(frozen=True)
class GBLEnd:
    TAG_ID: ClassVar[GBLTagId] = GBLTagId.END

    crc: int

    @classmethod
    def from_payload(cls, payload: bytes) -> GBLEnd:
        check_length(payload, 4, "GBL end tag")

        return cls(crc=int.from_bytes(payload[0:4], "little"))

    def serialize_payload(self) -> bytes:
        return self.crc.to_bytes(4, "little")


@dataclasses.dataclass(frozen=True)
class GBLUnknownTag:
    """A tag whose identifier is not recognized. Preserved verbatim."""

    tag_id: int
    payload: bytes

    def serialize_payload(self) -> bytes:
        return self.payload


GBLTag = Union[
    GBLHeader,
    GBLApplicationInfo,
    GBLSeUpgrade,
    GBLBootloader,
    GBLProgramData1,
    GBLProgramData2,
    GBLProgramDataLZ4,
    GBLProgramDataLZMA,
    GBLMetadata,
    GBLSignature,
    GBLEncryptionHeader,
    GBLEncryptionInitAesCcm,
    GBLEncryptedData,
    GBLEnd,
    GBLUnknownTag,
]

GBL_TAG_CLASSES: dict[GBLTagId, Any] = {
    GBLTagId.HEADER: GBLHeader,
    GBLTagId.APP_INFO: GBLApplicationInfo,
    GBLTagId.SE_UPGRADE: GBLSeUpgrade,
    GBLTagId.BOOTLOADER: GBLBootloader,
    GBLTagId.PROGRAM_DATA1: GBLProgramData1,
    GBLTagId.PROGRAM_DATA2: GBLProgramData2,
    GBLTagId.PROGRAM_DATA_LZ4: GBLProgramDataLZ4,
    GBLTagId.PROGRAM_DATA_LZMA: GBLProgramDataLZMA,
    GBLTagId.METADATA: GBLMetadata,
    GBLTagId.SIGNATURE: GBLSignature,
    GBLTagId.ENC_HEADER: GBLEncryptionHeader,
    GBLTagId.ENC_INIT: GBLEncryptionInitAesCcm,
    GBLTagId.ENC_GBL_DATA: GBLEncryptedData,
    GBLTagId.END: GBLEnd,
}

PROGRAM_DATA_TAGS = (
    GBLProgramData1,
    GBLProgramData2,
    GBLProgramDataLZ4,
    GBLProgramDataLZMA,
)


def gbl_tag_id(tag: GBLTag) -> int:
    if isinstance(tag, GBLUnknownTag):
        return tag.tag_id

    return int(tag.TAG_ID)


@dataclasses.dataclass(frozen=True)
class GBLImage:
    tags: list[GBLTag]

    # Bytes following the end tag. Real-world images pad with `0xFF` or `0x00`, and some
    # vendors append an entire second firmware payload. It is outside of the CRC and is
    # preserved verbatim so that images round-trip byte-for-byte.
    trailing_data: bytes = b""

    @classmethod
    def from_bytes(cls, data: bytes, *, validate: bool = True) -> GBLImage:
        data = bytes(data)
        tags: list[GBLTag] = []
        offset = 0

        while True:
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
            if tag_id == GBLTagId.END:
                break

        image = cls(tags=tags, trailing_data=data[offset:])

        if validate:
            image.validate()

        return image

    def serialize_tags(self) -> bytes:
        """Serialize only the tag region, which is what the CRC covers."""
        chunks = []

        for tag in self.tags:
            payload = tag.serialize_payload()
            chunks.append(
                gbl_tag_id(tag).to_bytes(4, "little")
                + len(payload).to_bytes(4, "little")
                + payload
            )

        return b"".join(chunks)

    def serialize(self, *, block_size: int = 1, padding: bytes = b"\xff") -> bytes:
        return pad_to_multiple(
            self.serialize_tags() + self.trailing_data, block_size, padding
        )

    def validate(self) -> None:
        if not self.tags:
            raise ValidationError("Image contains no tags")

        if not isinstance(self.tags[0], GBLHeader):
            raise ValidationError("First tag must be the header")

        if not isinstance(self.tags[-1], GBLEnd):
            raise ValidationError("Last tag must be the end tag")

        crc = zlib.crc32(self.serialize_tags())

        if crc != VALID_CRC32:
            raise ValidationError(
                f"Image CRC-32 is invalid: expected {VALID_CRC32:#010x}, got {crc:#010x}"
            )

    def get_tags(self, tag_type: type[T]) -> list[T]:
        return [t for t in self.tags if type(t) is tag_type]

    def find_first_tag(self, tag_type: type[T]) -> T | None:
        return next((t for t in self.tags if type(t) is tag_type), None)

    def get_first_tag(self, tag_type: type[T]) -> T:
        tag = self.find_first_tag(tag_type)

        if tag is None:
            raise KeyError(f"No {tag_type.__name__} tag exists")

        return tag

    def has_tag(self, tag_type: type[Any]) -> bool:
        return self.find_first_tag(tag_type) is not None

    def get_metadata(self) -> bytes | None:
        """The metadata tag contents, if the image has one. Opaque to the bootloader."""
        tag = self.find_first_tag(GBLMetadata)

        return None if tag is None else tag.metadata

    def regenerate_crc(self) -> GBLImage:
        tags = [t for t in self.tags if not isinstance(t, GBLEnd)]
        placeholder = type(self)(tags=[*tags, GBLEnd(crc=0)])
        crc = zlib.crc32(placeholder.serialize_tags()[:-4]) & 0xFFFFFFFF

        return type(self)(
            tags=[*tags, GBLEnd(crc=crc)], trailing_data=self.trailing_data
        )

    def is_combined_bootloader_app(self) -> bool:
        app_info = self.find_first_tag(GBLApplicationInfo)

        if app_info is None or not self.has_tag(GBLBootloader):
            return False

        return bool(
            GBLApplicationType.BOOTLOADER in app_info.type
            and app_info.type & ~GBLApplicationType.BOOTLOADER
        )

    def split_bootloader_app(self) -> list[GBLImage]:
        """Split a combined bootloader + application image into separate images."""
        if not self.is_combined_bootloader_app():
            return [self]

        header = self.get_first_tag(GBLHeader)

        if header.type != GBLType.NONE:
            raise ValueError("Cannot split signed or encrypted images")

        app_info = self.get_first_tag(GBLApplicationInfo)
        bootloader = type(self)(tags=[header, self.get_first_tag(GBLBootloader)])
        application = type(self)(
            tags=[
                header,
                dataclasses.replace(
                    app_info, type=app_info.type & ~GBLApplicationType.BOOTLOADER
                ),
                *[
                    t
                    for t in self.tags
                    if isinstance(t, (*PROGRAM_DATA_TAGS, GBLMetadata))
                ],
            ]
        )

        return [bootloader.regenerate_crc(), application.regenerate_crc()]

    def combine_bootloader_app(self, other: GBLImage) -> GBLImage:
        """Combine a bootloader image and an application image into a single image."""
        if self.has_tag(GBLBootloader) and other.has_tag(GBLBootloader):
            raise ValueError("Both images contain bootloaders")

        if not self.has_tag(GBLBootloader) and not other.has_tag(GBLBootloader):
            raise ValueError("Neither image contains a bootloader")

        header = self.get_first_tag(GBLHeader)

        if header != other.get_first_tag(GBLHeader):
            raise ValueError("GBL headers do not match")

        if header.type != GBLType.NONE:
            raise ValueError("Cannot combine signed or encrypted images")

        bootloader_image = self if self.has_tag(GBLBootloader) else other
        application = other if bootloader_image is self else self
        app_info = application.get_first_tag(GBLApplicationInfo)

        return type(self)(
            tags=[
                header,
                dataclasses.replace(
                    app_info, type=app_info.type | GBLApplicationType.BOOTLOADER
                ),
                bootloader_image.get_first_tag(GBLBootloader),
                *[
                    t
                    for t in application.tags
                    if isinstance(t, (*PROGRAM_DATA_TAGS, GBLMetadata))
                ],
            ]
        ).regenerate_crc()


def _parse_tag(tag_id: int, payload: bytes) -> GBLTag:
    try:
        known_tag_id = GBLTagId(tag_id)
    except ValueError:
        return GBLUnknownTag(tag_id=tag_id, payload=payload)

    parsed: GBLTag = GBL_TAG_CLASSES[known_tag_id].from_payload(payload)

    return parsed
