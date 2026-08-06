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

# An EBL always starts with the header tag and its fixed 140 byte length
EBL_MAGIC = b"\x00\x00\x00\x8c"

EBL_BLOCK_SIZE = 64
EBL_HEADER_SIGNATURE = 0xE350


class EBLTagId(enum.IntEnum):
    """EBL tag identifiers, as decoded from the big-endian 16-bit value on the wire."""

    HEADER = 0x0000
    PROG = 0xFE01
    MFGPROG = 0x02FE
    ERASEPROG = 0xFD03
    END = 0xFC04
    ENC_HEADER = 0xFB05
    ENC_INIT = 0xFA06
    ENC_EBL_DATA = 0xF907
    ENC_MAC = 0xF709


@dataclasses.dataclass(frozen=True)
class EBLHeader:
    TAG_ID: ClassVar[EBLTagId] = EBLTagId.HEADER

    version: int
    signature: int
    flash_addr: int
    aat_crc: int
    # The Application Address Table, which the EBL does not describe further
    aat: bytes

    @classmethod
    def from_payload(cls, payload: bytes) -> EBLHeader:
        if len(payload) < 12:
            raise ParseError(f"EBL header is truncated: {len(payload)} bytes")

        return cls(
            version=int.from_bytes(payload[0:2], "big"),
            signature=int.from_bytes(payload[2:4], "big"),
            flash_addr=int.from_bytes(payload[4:8], "big"),
            aat_crc=int.from_bytes(payload[8:12], "big"),
            aat=payload[12:],
        )

    def serialize_payload(self) -> bytes:
        return (
            self.version.to_bytes(2, "big")
            + self.signature.to_bytes(2, "big")
            + self.flash_addr.to_bytes(4, "big")
            + self.aat_crc.to_bytes(4, "big")
            + self.aat
        )


@dataclasses.dataclass(frozen=True)
class _EBLAddressedData:
    address: int
    data: bytes

    TAG_ID: ClassVar[EBLTagId]

    @classmethod
    def from_payload(cls, payload: bytes) -> Self:
        if len(payload) < 4:
            raise ParseError(f"{cls.__name__} is truncated: {len(payload)} bytes")

        return cls(
            address=int.from_bytes(payload[0:4], "big"),
            data=payload[4:],
        )

    def serialize_payload(self) -> bytes:
        return self.address.to_bytes(4, "big") + self.data


@dataclasses.dataclass(frozen=True)
class EBLProgram(_EBLAddressedData):
    TAG_ID: ClassVar[EBLTagId] = EBLTagId.PROG


@dataclasses.dataclass(frozen=True)
class EBLManufacturingProgram(_EBLAddressedData):
    TAG_ID: ClassVar[EBLTagId] = EBLTagId.MFGPROG


@dataclasses.dataclass(frozen=True)
class EBLEraseProgram(_EBLAddressedData):
    TAG_ID: ClassVar[EBLTagId] = EBLTagId.ERASEPROG


@dataclasses.dataclass(frozen=True)
class _EBLOpaqueData:
    data: bytes

    TAG_ID: ClassVar[EBLTagId]

    @classmethod
    def from_payload(cls, payload: bytes) -> Self:
        return cls(data=payload)

    def serialize_payload(self) -> bytes:
        return self.data


@dataclasses.dataclass(frozen=True)
class EBLEncryptionHeader(_EBLOpaqueData):
    TAG_ID: ClassVar[EBLTagId] = EBLTagId.ENC_HEADER


@dataclasses.dataclass(frozen=True)
class EBLEncryptionInit(_EBLOpaqueData):
    TAG_ID: ClassVar[EBLTagId] = EBLTagId.ENC_INIT


@dataclasses.dataclass(frozen=True)
class EBLEncryptedData(_EBLOpaqueData):
    TAG_ID: ClassVar[EBLTagId] = EBLTagId.ENC_EBL_DATA


@dataclasses.dataclass(frozen=True)
class EBLEncryptionMac(_EBLOpaqueData):
    TAG_ID: ClassVar[EBLTagId] = EBLTagId.ENC_MAC


@dataclasses.dataclass(frozen=True)
class EBLEnd:
    TAG_ID: ClassVar[EBLTagId] = EBLTagId.END

    # Unlike every other EBL field, the end tag's CRC is little-endian
    crc: int

    @classmethod
    def from_payload(cls, payload: bytes) -> EBLEnd:
        check_length(payload, 4, "EBL end tag")

        return cls(crc=int.from_bytes(payload[0:4], "little"))

    def serialize_payload(self) -> bytes:
        return self.crc.to_bytes(4, "little")


@dataclasses.dataclass(frozen=True)
class EBLUnknownTag:
    """A tag whose identifier is not recognized. Preserved verbatim."""

    tag_id: int
    payload: bytes

    def serialize_payload(self) -> bytes:
        return self.payload


EBLTag = Union[
    EBLHeader,
    EBLProgram,
    EBLManufacturingProgram,
    EBLEraseProgram,
    EBLEncryptionHeader,
    EBLEncryptionInit,
    EBLEncryptedData,
    EBLEncryptionMac,
    EBLEnd,
    EBLUnknownTag,
]

EBL_TAG_CLASSES: dict[EBLTagId, Any] = {
    EBLTagId.HEADER: EBLHeader,
    EBLTagId.PROG: EBLProgram,
    EBLTagId.MFGPROG: EBLManufacturingProgram,
    EBLTagId.ERASEPROG: EBLEraseProgram,
    EBLTagId.ENC_HEADER: EBLEncryptionHeader,
    EBLTagId.ENC_INIT: EBLEncryptionInit,
    EBLTagId.ENC_EBL_DATA: EBLEncryptedData,
    EBLTagId.ENC_MAC: EBLEncryptionMac,
    EBLTagId.END: EBLEnd,
}

PROGRAM_DATA_TAGS = (EBLProgram, EBLManufacturingProgram, EBLEraseProgram)


def ebl_tag_id(tag: EBLTag) -> int:
    if isinstance(tag, EBLUnknownTag):
        return tag.tag_id

    return int(tag.TAG_ID)


@dataclasses.dataclass(frozen=True)
class EBLImage:
    tags: list[EBLTag]

    # Bytes following the end tag: the padding that aligns the image to a 64 byte
    # boundary, and for some vendors an entire second firmware payload appended after
    # it. Outside of the CRC and preserved verbatim so that images round-trip
    # byte-for-byte.
    trailing_data: bytes = b""

    @classmethod
    def from_bytes(cls, data: bytes, *, validate: bool = True) -> EBLImage:
        data = bytes(data)
        tags: list[EBLTag] = []
        offset = 0

        while True:
            if offset + 4 > len(data):
                raise ParseError("Image is truncated: incomplete tag header")

            tag_id = int.from_bytes(data[offset : offset + 2], "big")
            length = int.from_bytes(data[offset + 2 : offset + 4], "big")
            start = offset + 4
            end = start + length

            if end > len(data):
                raise ParseError(
                    f"Tag {tag_id:#06x} is truncated: expected {length} bytes,"
                    f" got {len(data) - start}"
                )

            tags.append(_parse_tag(tag_id, data[start:end]))
            offset = end

            # An EBL ends at the first end tag. Everything after it is trailing data.
            if tag_id == EBLTagId.END:
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
                ebl_tag_id(tag).to_bytes(2, "big")
                + len(payload).to_bytes(2, "big")
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

        if not isinstance(self.tags[0], EBLHeader):
            raise ValidationError("First tag must be the header")

        if not isinstance(self.tags[-1], EBLEnd):
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

    def regenerate_crc(self) -> EBLImage:
        tags = [t for t in self.tags if not isinstance(t, EBLEnd)]
        placeholder = type(self)(tags=[*tags, EBLEnd(crc=0)])
        crc = zlib.crc32(placeholder.serialize_tags()[:-4]) & 0xFFFFFFFF

        return type(self)(
            tags=[*tags, EBLEnd(crc=crc)], trailing_data=self.trailing_data
        )


def _parse_tag(tag_id: int, payload: bytes) -> EBLTag:
    try:
        known_tag_id = EBLTagId(tag_id)
    except ValueError:
        return EBLUnknownTag(tag_id=tag_id, payload=payload)

    parsed: EBLTag = EBL_TAG_CLASSES[known_tag_id].from_payload(payload)

    return parsed
