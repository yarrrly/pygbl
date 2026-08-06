"""Tags defined by the SDK that no image in the wild uses.

None of these appeared in a survey of 772 real images, so they are covered here from
the SDK's struct definitions rather than from a fixture.
"""

from __future__ import annotations

import pytest

from pygbl import (
    GBLCertificateEcdsaP256,
    GBLDelta,
    GBLDeltaLZ4,
    GBLDeltaLZMA,
    GBLEraseProg,
    GBLHeader,
    GBLImage,
    GBLProg,
    GBLType,
    GBLVersionDependency,
    GBLVersionDependencyImageType,
    ParseError,
    VersionDependencyStatement,
)


def build(*tags: object) -> GBLImage:
    return GBLImage(
        tags=[GBLHeader(version=3, type=GBLType.NONE), *tags]  # type: ignore[list-item]
    ).regenerate_crc()


def test_version_dependency_statement_layout() -> None:
    """imageType (1), statement (1), reserved (2), version (4)."""
    statement = VersionDependencyStatement(
        image_type=GBLVersionDependencyImageType.BOOTLOADER,
        statement=0x06,  # GEQ
        version=0x02040003,
    )
    data = statement.serialize()

    assert len(data) == VersionDependencyStatement.SIZE == 8
    assert data == bytes([0x02, 0x06, 0x00, 0x00]) + (0x02040003).to_bytes(4, "little")
    assert VersionDependencyStatement.from_bytes(data) == statement


def test_version_dependency_holds_several_statements() -> None:
    """The tag length divided by the struct size gives the statement count."""
    statements = [
        VersionDependencyStatement(
            image_type=GBLVersionDependencyImageType.APPLICATION,
            statement=0x08,
            version=1,
        ),
        VersionDependencyStatement(
            image_type=GBLVersionDependencyImageType.SE, statement=0x04, version=2
        ),
    ]
    tag = GBLVersionDependency(statements=statements)
    payload = tag.serialize_payload()

    assert len(payload) == 16
    assert GBLVersionDependency.from_payload(payload) == tag


def test_version_dependency_rejects_partial_statements() -> None:
    with pytest.raises(ParseError, match="multiple of 8 bytes"):
        GBLVersionDependency.from_payload(bytes(12))


def test_version_dependency_roundtrips_in_an_image() -> None:
    tag = GBLVersionDependency(
        statements=[
            VersionDependencyStatement(
                image_type=GBLVersionDependencyImageType.BOOTLOADER,
                statement=0x06,
                version=0x02040000,
            )
        ]
    )
    image = build(tag)
    data = image.serialize()

    assert GBLImage.from_bytes(data) == image
    assert GBLImage.from_bytes(data).get_first_tag(GBLVersionDependency) == tag


def test_certificate_layout() -> None:
    """structVersion (1), flags (3), key (64), version (4), signature (64)."""
    certificate = GBLCertificateEcdsaP256(
        struct_version=1,
        flags=b"\x00\x00\x00",
        key=bytes(range(64)),
        version=7,
        signature=bytes(range(64, 128)),
    )
    payload = certificate.serialize_payload()

    assert len(payload) == 136
    assert payload[0] == 1
    assert payload[68:72] == (7).to_bytes(4, "little")
    assert GBLCertificateEcdsaP256.from_payload(payload) == certificate


def test_certificate_rejects_wrong_length() -> None:
    with pytest.raises(ParseError, match="must be 136 bytes"):
        GBLCertificateEcdsaP256.from_payload(bytes(64))


def test_certificate_roundtrips_in_an_image() -> None:
    certificate = GBLCertificateEcdsaP256(
        struct_version=1,
        flags=bytes(3),
        key=bytes(64),
        version=1,
        signature=bytes(64),
    )
    image = build(certificate)

    assert GBLImage.from_bytes(image.serialize()) == image


@pytest.mark.parametrize("tag_type", [GBLDelta, GBLDeltaLZ4, GBLDeltaLZMA])
def test_delta_layout(tag_type: type[GBLDelta]) -> None:
    """A delta patch is prefixed with the target image's CRC and size."""
    tag = tag_type(
        new_fw_crc=0xDEADBEEF, new_fw_size=0x1234, address=0x8006000, data=b"patch"
    )
    payload = tag.serialize_payload()

    assert payload[0:4] == (0xDEADBEEF).to_bytes(4, "little")
    assert payload[4:8] == (0x1234).to_bytes(4, "little")
    assert payload[8:12] == (0x8006000).to_bytes(4, "little")
    assert payload[12:] == b"patch"
    assert tag_type.from_payload(payload) == tag


@pytest.mark.parametrize("tag_type", [GBLDelta, GBLDeltaLZ4, GBLDeltaLZMA])
def test_delta_rejects_truncated_prefix(tag_type: type[GBLDelta]) -> None:
    with pytest.raises(ParseError, match="is truncated"):
        tag_type.from_payload(bytes(8))


@pytest.mark.parametrize("tag_type", [GBLDelta, GBLDeltaLZ4, GBLDeltaLZMA])
def test_delta_roundtrips_in_an_image(tag_type: type[GBLDelta]) -> None:
    tag = tag_type(new_fw_crc=1, new_fw_size=2, address=0x8006000, data=b"\xaa" * 32)
    image = build(tag)

    assert GBLImage.from_bytes(image.serialize()) == image


def test_prog_and_eraseprog_are_distinct_tags() -> None:
    """Different tag ids, though the bootloader dispatches both to one handler."""
    prog = GBLProg(address=0x8006000, data=b"\x01")
    erase_prog = GBLEraseProg(address=0x8006000, data=b"\x01")

    assert int(prog.tag_id) == 0xFE0101FE
    assert int(erase_prog.tag_id) == 0xFD0303FD
    assert prog.tag_id != erase_prog.tag_id

    # Only the tag id differs: the payload layout is identical
    assert prog.serialize_payload() == erase_prog.serialize_payload()


def test_prog_survives_a_roundtrip() -> None:
    """`PROG` never appears in the wild, so make sure we can still read one."""
    image = build(GBLProg(address=0x8006000, data=b"\xab" * 16))
    data = image.serialize()
    reparsed = GBLImage.from_bytes(data)

    assert reparsed == image
    assert isinstance(reparsed.get_first_tag(GBLProg), GBLProg)
    assert reparsed.find_first_tag(GBLEraseProg) is None
