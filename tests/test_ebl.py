from __future__ import annotations

import zlib

import pytest

from pygbl import (
    EBL_BLOCK_SIZE,
    VALID_CRC32,
    EBLEnd,
    EBLEraseProgram,
    EBLHeader,
    EBLImage,
    EBLProgram,
    EBLTagId,
    EBLUnknownTag,
    ParseError,
    ValidationError,
    parse_firmware_image,
)

# Field values taken from a real EM357 image
HEADER = EBLHeader(
    version=0x0202,
    signature=0xE350,
    flash_addr=0x08004000,
    aat_crc=0x110F5679,
    aat=bytes(range(128)),
)


def build(*tags, trailing_data: bytes = b"") -> EBLImage:
    return EBLImage(tags=[HEADER, *tags], trailing_data=trailing_data).regenerate_crc()


def test_build_and_reparse():
    image = build(EBLEraseProgram(address=0x08004080, data=b"\xab" * 32))
    data = image.serialize()

    assert EBLImage.from_bytes(data) == image
    assert zlib.crc32(data) == VALID_CRC32


def test_magic_matches_header_tag_and_length():
    data = build().serialize()

    # tag 0x0000 followed by the fixed 140 byte header length, both big-endian
    assert data[0:4] == b"\x00\x00\x00\x8c"


def test_tag_framing_is_big_endian():
    image = build(EBLEraseProgram(address=0x08004080, data=b"\x01\x02"))
    data = image.serialize()
    offset = 4 + 140

    assert data[offset : offset + 2] == int(EBLTagId.ERASEPROG).to_bytes(2, "big")
    assert data[offset + 2 : offset + 4] == (6).to_bytes(2, "big")
    assert data[offset + 4 : offset + 8] == (0x08004080).to_bytes(4, "big")


def test_header_fields_are_big_endian():
    payload = HEADER.serialize_payload()

    assert len(payload) == 140
    assert payload[0:2] == b"\x02\x02"
    assert payload[2:4] == b"\xe3\x50"
    assert payload[4:8] == b"\x08\x00\x40\x00"
    assert EBLHeader.from_payload(payload) == HEADER


def test_end_tag_crc_is_little_endian():
    """Every other EBL field is big-endian, but the end tag's CRC is not."""
    image = build(EBLEraseProgram(address=0, data=b"\x00" * 8))
    data = image.serialize()
    crc = image.get_first_tag(EBLEnd).crc

    assert data[-4:] == crc.to_bytes(4, "little")
    assert EBLEnd.from_payload(data[-4:]).crc == crc


def test_eraseprogram_addresses_are_contiguous():
    """Mirrors how real images chain program tags."""
    first = EBLEraseProgram(address=0x08004080, data=b"\x00" * 0x780)
    second = EBLEraseProgram(address=0x08004800, data=b"\x00" * 0x800)
    image = build(first, second)

    tags = image.get_tags(EBLEraseProgram)

    assert tags[0].address + len(tags[0].data) == tags[1].address


def test_program_and_eraseprogram_are_distinct_tags():
    image = build(
        EBLProgram(address=0, data=b"\x01"),
        EBLEraseProgram(address=4, data=b"\x02"),
    )
    reparsed = EBLImage.from_bytes(image.serialize())

    assert len(reparsed.get_tags(EBLProgram)) == 1
    assert len(reparsed.get_tags(EBLEraseProgram)) == 1


@pytest.mark.parametrize(
    "trailing", [b"", b"\xff" * 20, b"\x00" * 8, b"vendor payload"]
)
def test_trailing_data_roundtrips(trailing: bytes):
    image = build(EBLEraseProgram(address=0, data=b"\x01"), trailing_data=trailing)
    data = image.serialize()

    assert data.endswith(trailing)
    assert EBLImage.from_bytes(data) == image
    assert EBLImage.from_bytes(data).trailing_data == trailing


def test_unaligned_image_is_accepted():
    """Real images append vendor payloads that break 64 byte alignment."""
    image = build(EBLEraseProgram(address=0, data=b"\x01"), trailing_data=b"\xa5" * 3)
    data = image.serialize()

    assert len(data) % EBL_BLOCK_SIZE != 0

    EBLImage.from_bytes(data).validate()


def test_block_padding_to_ebl_alignment():
    image = build(EBLEraseProgram(address=0, data=b"\x01"))
    padded = image.serialize(block_size=EBL_BLOCK_SIZE)

    assert len(padded) % EBL_BLOCK_SIZE == 0
    assert padded.startswith(image.serialize())


def test_unknown_tag_is_preserved():
    unknown = EBLUnknownTag(tag_id=0xABCD, payload=b"\x01\x02")
    image = build(unknown)
    data = image.serialize()

    assert EBLImage.from_bytes(data).get_first_tag(EBLUnknownTag) == unknown
    assert EBLImage.from_bytes(data).serialize() == data


def test_bad_crc_rejected():
    data = bytearray(build(EBLEraseProgram(address=0, data=b"\x01")).serialize())
    data[-1] ^= 0xFF

    with pytest.raises(ValidationError, match="CRC-32 is invalid"):
        EBLImage.from_bytes(bytes(data))

    EBLImage.from_bytes(bytes(data), validate=False)


def test_truncated_tag_payload():
    with pytest.raises(ParseError, match="is truncated"):
        EBLImage.from_bytes(b"\x00\x00\x00\x8c" + b"\x00" * 4)


def test_missing_end_tag_rejected():
    image = EBLImage(tags=[HEADER])

    with pytest.raises(ValidationError, match="Last tag must be the end tag"):
        image.validate()


def test_parse_firmware_image_dispatches():
    data = build(EBLEraseProgram(address=0, data=b"\x01")).serialize()

    assert isinstance(parse_firmware_image(data), EBLImage)
