"""Tags defined by the SDK that no image in the wild uses.

None of these appeared in a survey of 772 real images, so they are covered here from
the SDK's struct definitions rather than from a fixture.
"""

from __future__ import annotations

import pytest

from pygbl import (
    GBL3CertificateEcdsaP256,
    GBL3Delta,
    GBL3DeltaLZ4,
    GBL3DeltaLZMA,
    GBL3EraseProg,
    GBL3Header,
    GBL3Image,
    GBL3Prog,
    GBL3Type,
    GBL3VersionDependency,
    GBL3VersionDependencyImageType,
    ParseError,
    VersionDependencyStatement,
)


def build(*tags: object) -> GBL3Image:
    return GBL3Image(
        tags=[GBL3Header(version=3, type=GBL3Type.NONE), *tags]  # type: ignore[list-item]
    ).regenerate_crc()


def test_version_dependency_statement_layout() -> None:
    """imageType (1), statement (1), reserved (2), version (4)."""
    statement = VersionDependencyStatement(
        image_type=GBL3VersionDependencyImageType.BOOTLOADER,
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
            image_type=GBL3VersionDependencyImageType.APPLICATION,
            statement=0x08,
            version=1,
        ),
        VersionDependencyStatement(
            image_type=GBL3VersionDependencyImageType.SE, statement=0x04, version=2
        ),
    ]
    tag = GBL3VersionDependency(statements=statements)
    payload = tag.serialize_payload()

    assert len(payload) == 16
    assert GBL3VersionDependency.from_payload(payload) == tag


def test_version_dependency_rejects_partial_statements() -> None:
    with pytest.raises(ParseError, match="multiple of 8 bytes"):
        GBL3VersionDependency.from_payload(bytes(12))


def test_version_dependency_roundtrips_in_an_image() -> None:
    tag = GBL3VersionDependency(
        statements=[
            VersionDependencyStatement(
                image_type=GBL3VersionDependencyImageType.BOOTLOADER,
                statement=0x06,
                version=0x02040000,
            )
        ]
    )
    image = build(tag)
    data = image.serialize()

    assert GBL3Image.from_bytes(data) == image
    assert GBL3Image.from_bytes(data).get_first_tag(GBL3VersionDependency) == tag


def test_certificate_layout() -> None:
    """structVersion (1), flags (3), key (64), version (4), signature (64)."""
    certificate = GBL3CertificateEcdsaP256(
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
    assert GBL3CertificateEcdsaP256.from_payload(payload) == certificate


def test_certificate_rejects_wrong_length() -> None:
    with pytest.raises(ParseError, match="must be 136 bytes"):
        GBL3CertificateEcdsaP256.from_payload(bytes(64))


def test_certificate_roundtrips_in_an_image() -> None:
    certificate = GBL3CertificateEcdsaP256(
        struct_version=1,
        flags=bytes(3),
        key=bytes(64),
        version=1,
        signature=bytes(64),
    )
    image = build(certificate)

    assert GBL3Image.from_bytes(image.serialize()) == image


@pytest.mark.parametrize("tag_type", [GBL3Delta, GBL3DeltaLZ4, GBL3DeltaLZMA])
def test_delta_layout(tag_type: type[GBL3Delta]) -> None:
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


@pytest.mark.parametrize("tag_type", [GBL3Delta, GBL3DeltaLZ4, GBL3DeltaLZMA])
def test_delta_rejects_truncated_prefix(tag_type: type[GBL3Delta]) -> None:
    with pytest.raises(ParseError, match="is truncated"):
        tag_type.from_payload(bytes(8))


@pytest.mark.parametrize("tag_type", [GBL3Delta, GBL3DeltaLZ4, GBL3DeltaLZMA])
def test_delta_roundtrips_in_an_image(tag_type: type[GBL3Delta]) -> None:
    tag = tag_type(new_fw_crc=1, new_fw_size=2, address=0x8006000, data=b"\xaa" * 32)
    image = build(tag)

    assert GBL3Image.from_bytes(image.serialize()) == image


def test_prog_and_eraseprog_are_distinct_tags() -> None:
    """Different tag ids, though the bootloader dispatches both to one handler."""
    prog = GBL3Prog(address=0x8006000, data=b"\x01")
    erase_prog = GBL3EraseProg(address=0x8006000, data=b"\x01")

    assert int(prog.tag_id) == 0xFE0101FE
    assert int(erase_prog.tag_id) == 0xFD0303FD
    assert prog.tag_id != erase_prog.tag_id

    # Only the tag id differs: the payload layout is identical
    assert prog.serialize_payload() == erase_prog.serialize_payload()


def test_prog_survives_a_roundtrip() -> None:
    """`PROG` never appears in the wild, so make sure we can still read one."""
    image = build(GBL3Prog(address=0x8006000, data=b"\xab" * 16))
    data = image.serialize()
    reparsed = GBL3Image.from_bytes(data)

    assert reparsed == image
    assert isinstance(reparsed.get_first_tag(GBL3Prog), GBL3Prog)
    assert reparsed.get_tags(GBL3EraseProg, allow_missing=True) == []
