from __future__ import annotations

import zlib

import pytest

from pygbl import (
    VALID_CRC32,
    BootloaderVersion,
    GBL3ApplicationInfo,
    GBL3ApplicationType,
    GBL3EncryptedData,
    GBL3EncryptionInitAesCcm,
    GBL3End,
    GBL3EraseProg,
    GBL3Header,
    GBL3Image,
    GBL3Metadata,
    GBL3Signature,
    GBL3TagBase,
    GBL3TagId,
    GBL3Type,
    GBL3UnknownTag,
    ParseError,
    ValidationError,
    parse_firmware_image,
)

PRODUCT_ID = bytes(range(16))


def build(
    *tags: GBL3TagBase, trailing_data: bytes = b"", type: GBL3Type = GBL3Type.NONE
) -> GBL3Image:
    return GBL3Image(
        tags=[GBL3Header(version=3, type=type), *tags],
        trailing_data=trailing_data,
    ).regenerate_crc()


def app_info(app_type: GBL3ApplicationType) -> GBL3ApplicationInfo:
    return GBL3ApplicationInfo(
        type=app_type, version=1, capabilities=0, product_id=PRODUCT_ID
    )


def test_build_and_reparse() -> None:
    image = build(
        app_info(GBL3ApplicationType.ZIGBEE),
        GBL3EraseProg(address=0x1000, data=b"\xab" * 16),
    )
    data = image.serialize()

    assert GBL3Image.from_bytes(data) == image
    assert zlib.crc32(data) == VALID_CRC32


def test_header_endianness() -> None:
    """The version is big-endian while the type flags are little-endian."""
    image = build()
    payload = image.serialize()[8:16]

    assert payload[0:4] == b"\x00\x00\x00\x03"
    assert GBL3Header.from_payload(payload).version == 3


def test_tag_framing() -> None:
    image = build(GBL3Metadata(metadata=b"hello"))
    data = image.serialize()

    # tag id and length are both little-endian 32-bit
    assert data[0:4] == b"\xeb\x17\xa6\x03"
    assert data[4:8] == (8).to_bytes(4, "little")
    assert data[16:20] == int(GBL3TagId.METADATA).to_bytes(4, "little")
    assert data[20:24] == (5).to_bytes(4, "little")


def test_signature_is_64_bytes() -> None:
    signature = GBL3Signature(r=bytes(range(32)), s=bytes(range(32, 64)))
    payload = signature.serialize_payload()

    assert len(payload) == 64
    assert GBL3Signature.from_payload(payload) == signature

    with pytest.raises(ParseError, match="must be 64 bytes"):
        GBL3Signature.from_payload(payload[:8])


def test_encryption_init_is_structured() -> None:
    tag = GBL3EncryptionInitAesCcm(msg_len=0x1234, nonce=bytes(range(12)))
    payload = tag.serialize_payload()

    assert len(payload) == 16
    assert payload[0:4] == (0x1234).to_bytes(4, "little")
    assert GBL3EncryptionInitAesCcm.from_payload(payload) == tag


def test_encrypted_payload_stays_opaque() -> None:
    image = build(
        GBL3EncryptionInitAesCcm(msg_len=4, nonce=bytes(12)),
        GBL3EncryptedData(ciphertext=b"\xde\xad\xbe\xef"),
        type=GBL3Type.ENCRYPTION_AESCCM,
    )

    reparsed = GBL3Image.from_bytes(image.serialize())

    assert reparsed.get_first_tag(GBL3EncryptedData).ciphertext == b"\xde\xad\xbe\xef"
    assert reparsed.get_first_tag(GBL3Header).type == GBL3Type.ENCRYPTION_AESCCM


def test_unknown_tag_is_preserved() -> None:
    unknown = GBL3UnknownTag(tag_id=0x12345678, payload=b"\x01\x02\x03")
    image = build(unknown)
    data = image.serialize()
    reparsed = GBL3Image.from_bytes(data)

    assert reparsed.get_first_tag(GBL3UnknownTag) == unknown
    assert reparsed.serialize() == data


def test_unknown_header_flags_are_preserved() -> None:
    image = build(type=GBL3Type(0x00008001))
    data = image.serialize()

    assert GBL3Image.from_bytes(data).get_first_tag(GBL3Header).type == 0x00008001
    assert GBL3Image.from_bytes(data).serialize() == data


@pytest.mark.parametrize(
    "trailing", [b"", b"\xff", b"\xff\xff\xff", b"\x00\x00", b"vendor payload"]
)
def test_trailing_data_roundtrips(trailing: bytes) -> None:
    image = build(GBL3Metadata(metadata=b"{}"), trailing_data=trailing)
    data = image.serialize()

    assert data.endswith(trailing)
    assert GBL3Image.from_bytes(data) == image
    assert GBL3Image.from_bytes(data).trailing_data == trailing


def test_trailing_data_is_outside_the_crc() -> None:
    tags = [GBL3Header(version=3, type=GBL3Type.NONE), GBL3Metadata(metadata=b"{}")]
    without = GBL3Image(tags=tags).regenerate_crc()
    with_trailing = GBL3Image(tags=tags, trailing_data=b"\xff" * 64).regenerate_crc()

    assert without.get_first_tag(GBL3End) == with_trailing.get_first_tag(GBL3End)
    with_trailing.validate()


def test_block_size_padding() -> None:
    image = build(GBL3EraseProg(address=0, data=b"\x00" * 3))

    assert len(image.serialize(block_size=128)) % 128 == 0
    assert image.serialize(block_size=128).startswith(image.serialize())
    assert image.serialize(block_size=128).endswith(b"\xff")
    assert image.serialize(block_size=128, padding=b"\x00").endswith(b"\x00")


def test_bad_crc_rejected() -> None:
    data = bytearray(build(GBL3Metadata(metadata=b"{}")).serialize())
    data[-1] ^= 0xFF

    with pytest.raises(ValidationError, match="CRC-32 is invalid"):
        GBL3Image.from_bytes(bytes(data))

    # ...but parsing without validation still works
    GBL3Image.from_bytes(bytes(data), validate=False)


def test_truncated_tag_header() -> None:
    with pytest.raises(ParseError, match="incomplete tag header"):
        GBL3Image.from_bytes(b"\xeb\x17\xa6\x03\x08\x00")


def test_truncated_tag_payload() -> None:
    with pytest.raises(ParseError, match="is truncated"):
        GBL3Image.from_bytes(
            b"\xeb\x17\xa6\x03" + (999).to_bytes(4, "little") + b"\x00"
        )


def test_missing_header_rejected() -> None:
    image = GBL3Image(tags=[GBL3Metadata(metadata=b"{}")]).regenerate_crc()

    with pytest.raises(ValidationError, match="First tag must be the header"):
        image.validate()


def test_metadata_is_opaque_bytes() -> None:
    image = build(GBL3Metadata(metadata=b'{"not": "parsed"}'))

    assert image.get_metadata() == b'{"not": "parsed"}'
    assert build().get_metadata() is None


def test_tag_lookup_helpers() -> None:
    first = GBL3EraseProg(address=0, data=b"\x01")
    second = GBL3EraseProg(address=4, data=b"\x02")
    image = build(first, second)

    assert image.get_tags(GBL3EraseProg) == [first, second]
    assert image.get_first_tag(GBL3EraseProg) == first
    assert image.find_first_tag(GBL3Metadata) is None
    assert image.has_tag(GBL3EraseProg)
    assert not image.has_tag(GBL3Metadata)

    with pytest.raises(KeyError, match="No GBL3Metadata tag exists"):
        image.get_first_tag(GBL3Metadata)


def test_bootloader_version() -> None:
    version = BootloaderVersion.from_int(0x02040008)

    assert (version.major, version.minor, version.customer) == (2, 4, 8)
    assert str(version) == "2.4.8"
    assert version.as_int() == 0x02040008


def test_parse_firmware_image_rejects_unknown() -> None:
    with pytest.raises(ParseError, match="not a GBL or EBL image"):
        parse_firmware_image(b"not a firmware image")


def test_parse_firmware_image_dispatches() -> None:
    data = build(GBL3Metadata(metadata=b"{}")).serialize()

    assert isinstance(parse_firmware_image(data), GBL3Image)
