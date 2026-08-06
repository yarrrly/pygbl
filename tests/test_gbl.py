from __future__ import annotations

import zlib

import pytest

from pygbl import (
    VALID_CRC32,
    BootloaderVersion,
    GBLApplicationInfo,
    GBLApplicationType,
    GBLEncryptedData,
    GBLEncryptionInitAesCcm,
    GBLEnd,
    GBLEraseProg,
    GBLHeader,
    GBLImage,
    GBLMetadata,
    GBLSignature,
    GBLTagBase,
    GBLTagId,
    GBLType,
    GBLUnknownTag,
    ParseError,
    ValidationError,
    parse_firmware_image,
)

PRODUCT_ID = bytes(range(16))


def build(
    *tags: GBLTagBase, trailing_data: bytes = b"", type: GBLType = GBLType.NONE
) -> GBLImage:
    return GBLImage(
        tags=[GBLHeader(version=3, type=type), *tags],
        trailing_data=trailing_data,
    ).regenerate_crc()


def app_info(app_type: GBLApplicationType) -> GBLApplicationInfo:
    return GBLApplicationInfo(
        type=app_type, version=1, capabilities=0, product_id=PRODUCT_ID
    )


def test_build_and_reparse() -> None:
    image = build(
        app_info(GBLApplicationType.ZIGBEE),
        GBLEraseProg(address=0x1000, data=b"\xab" * 16),
    )
    data = image.serialize()

    assert GBLImage.from_bytes(data) == image
    assert zlib.crc32(data) == VALID_CRC32


def test_header_endianness() -> None:
    """The version is big-endian while the type flags are little-endian."""
    image = build()
    payload = image.serialize()[8:16]

    assert payload[0:4] == b"\x00\x00\x00\x03"
    assert GBLHeader.from_payload(payload).version == 3


def test_tag_framing() -> None:
    image = build(GBLMetadata(metadata=b"hello"))
    data = image.serialize()

    # tag id and length are both little-endian 32-bit
    assert data[0:4] == b"\xeb\x17\xa6\x03"
    assert data[4:8] == (8).to_bytes(4, "little")
    assert data[16:20] == int(GBLTagId.METADATA).to_bytes(4, "little")
    assert data[20:24] == (5).to_bytes(4, "little")


def test_signature_is_64_bytes() -> None:
    signature = GBLSignature(r=bytes(range(32)), s=bytes(range(32, 64)))
    payload = signature.serialize_payload()

    assert len(payload) == 64
    assert GBLSignature.from_payload(payload) == signature

    with pytest.raises(ParseError, match="must be 64 bytes"):
        GBLSignature.from_payload(payload[:8])


def test_encryption_init_is_structured() -> None:
    tag = GBLEncryptionInitAesCcm(msg_len=0x1234, nonce=bytes(range(12)))
    payload = tag.serialize_payload()

    assert len(payload) == 16
    assert payload[0:4] == (0x1234).to_bytes(4, "little")
    assert GBLEncryptionInitAesCcm.from_payload(payload) == tag


def test_encrypted_payload_stays_opaque() -> None:
    image = build(
        GBLEncryptionInitAesCcm(msg_len=4, nonce=bytes(12)),
        GBLEncryptedData(ciphertext=b"\xde\xad\xbe\xef"),
        type=GBLType.ENCRYPTION_AESCCM,
    )

    reparsed = GBLImage.from_bytes(image.serialize())

    assert reparsed.get_first_tag(GBLEncryptedData).ciphertext == b"\xde\xad\xbe\xef"
    assert reparsed.get_first_tag(GBLHeader).type == GBLType.ENCRYPTION_AESCCM


def test_unknown_tag_is_preserved() -> None:
    unknown = GBLUnknownTag(tag_id=0x12345678, payload=b"\x01\x02\x03")
    image = build(unknown)
    data = image.serialize()
    reparsed = GBLImage.from_bytes(data)

    assert reparsed.get_first_tag(GBLUnknownTag) == unknown
    assert reparsed.serialize() == data


def test_unknown_header_flags_are_preserved() -> None:
    image = build(type=GBLType(0x00008001))
    data = image.serialize()

    assert GBLImage.from_bytes(data).get_first_tag(GBLHeader).type == 0x00008001
    assert GBLImage.from_bytes(data).serialize() == data


@pytest.mark.parametrize(
    "trailing", [b"", b"\xff", b"\xff\xff\xff", b"\x00\x00", b"vendor payload"]
)
def test_trailing_data_roundtrips(trailing: bytes) -> None:
    image = build(GBLMetadata(metadata=b"{}"), trailing_data=trailing)
    data = image.serialize()

    assert data.endswith(trailing)
    assert GBLImage.from_bytes(data) == image
    assert GBLImage.from_bytes(data).trailing_data == trailing


def test_trailing_data_is_outside_the_crc() -> None:
    tags = [GBLHeader(version=3, type=GBLType.NONE), GBLMetadata(metadata=b"{}")]
    without = GBLImage(tags=tags).regenerate_crc()
    with_trailing = GBLImage(tags=tags, trailing_data=b"\xff" * 64).regenerate_crc()

    assert without.get_first_tag(GBLEnd) == with_trailing.get_first_tag(GBLEnd)
    with_trailing.validate()


def test_block_size_padding() -> None:
    image = build(GBLEraseProg(address=0, data=b"\x00" * 3))

    assert len(image.serialize(block_size=128)) % 128 == 0
    assert image.serialize(block_size=128).startswith(image.serialize())
    assert image.serialize(block_size=128).endswith(b"\xff")
    assert image.serialize(block_size=128, padding=b"\x00").endswith(b"\x00")


def test_bad_crc_rejected() -> None:
    data = bytearray(build(GBLMetadata(metadata=b"{}")).serialize())
    data[-1] ^= 0xFF

    with pytest.raises(ValidationError, match="CRC-32 is invalid"):
        GBLImage.from_bytes(bytes(data))

    # ...but parsing without validation still works
    GBLImage.from_bytes(bytes(data), validate=False)


def test_truncated_tag_header() -> None:
    with pytest.raises(ParseError, match="incomplete tag header"):
        GBLImage.from_bytes(b"\xeb\x17\xa6\x03\x08\x00")


def test_truncated_tag_payload() -> None:
    with pytest.raises(ParseError, match="is truncated"):
        GBLImage.from_bytes(b"\xeb\x17\xa6\x03" + (999).to_bytes(4, "little") + b"\x00")


def test_missing_header_rejected() -> None:
    image = GBLImage(tags=[GBLMetadata(metadata=b"{}")]).regenerate_crc()

    with pytest.raises(ValidationError, match="First tag must be the header"):
        image.validate()


def test_metadata_is_opaque_bytes() -> None:
    image = build(GBLMetadata(metadata=b'{"not": "parsed"}'))

    assert image.get_metadata() == b'{"not": "parsed"}'
    assert build().get_metadata() is None


def test_tag_lookup_helpers() -> None:
    first = GBLEraseProg(address=0, data=b"\x01")
    second = GBLEraseProg(address=4, data=b"\x02")
    image = build(first, second)

    assert image.get_tags(GBLEraseProg) == [first, second]
    assert image.get_first_tag(GBLEraseProg) == first
    assert image.find_first_tag(GBLMetadata) is None
    assert image.has_tag(GBLEraseProg)
    assert not image.has_tag(GBLMetadata)

    with pytest.raises(KeyError, match="No GBLMetadata tag exists"):
        image.get_first_tag(GBLMetadata)


def test_bootloader_version() -> None:
    version = BootloaderVersion.from_int(0x02040008)

    assert (version.major, version.minor, version.customer) == (2, 4, 8)
    assert str(version) == "2.4.8"
    assert version.as_int() == 0x02040008


def test_parse_firmware_image_rejects_unknown() -> None:
    with pytest.raises(ParseError, match="neither a GBL nor an EBL"):
        parse_firmware_image(b"not a firmware image")


def test_parse_firmware_image_dispatches() -> None:
    data = build(GBLMetadata(metadata=b"{}")).serialize()

    assert isinstance(parse_firmware_image(data), GBLImage)
