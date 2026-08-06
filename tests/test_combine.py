from __future__ import annotations

import pytest

from pygbl import (
    BootloaderVersion,
    GBL3ApplicationInfo,
    GBL3ApplicationType,
    GBL3EraseProg,
    GBL3Header,
    GBL3Image,
    GBL3Metadata,
    GBL3Type,
)
from pygbl.gbl3 import GBL3Bootloader

PRODUCT_ID = bytes(range(16))
BOOTLOADER = GBL3Bootloader(
    version=BootloaderVersion(major=2, minor=4, customer=3),
    address=0x0,
    data=b"\xbb" * 64,
)
PROGRAM_DATA = GBL3EraseProg(address=0x6000, data=b"\xaa" * 128)
METADATA = GBL3Metadata(metadata=b'{"fw_type": "zigbee_ncp"}')


def header(type: GBL3Type = GBL3Type.NONE) -> GBL3Header:
    return GBL3Header(version=3, type=type)


def app_info(app_type: GBL3ApplicationType) -> GBL3ApplicationInfo:
    return GBL3ApplicationInfo(
        type=app_type, version=1, capabilities=0, product_id=PRODUCT_ID
    )


def bootloader_image() -> GBL3Image:
    return GBL3Image(tags=[header(), BOOTLOADER]).regenerate_crc()


def application_image() -> GBL3Image:
    return GBL3Image(
        tags=[header(), app_info(GBL3ApplicationType.ZIGBEE), PROGRAM_DATA, METADATA]
    ).regenerate_crc()


def test_combine() -> None:
    combined = application_image().combine_bootloader_app(bootloader_image())

    combined.validate()

    assert combined.get_first_tag(GBL3ApplicationInfo).type == (
        GBL3ApplicationType.ZIGBEE | GBL3ApplicationType.BOOTLOADER
    )
    assert combined.get_first_tag(GBL3Bootloader) == BOOTLOADER
    assert combined.get_first_tag(GBL3EraseProg) == PROGRAM_DATA
    assert combined.get_first_tag(GBL3Metadata) == METADATA
    assert combined.is_combined_bootloader_app()


def test_combine_is_order_independent() -> None:
    assert application_image().combine_bootloader_app(
        bootloader_image()
    ) == bootloader_image().combine_bootloader_app(application_image())


def test_split_roundtrips_combine() -> None:
    combined = application_image().combine_bootloader_app(bootloader_image())

    assert combined.split_bootloader_app() == [bootloader_image(), application_image()]


def test_split_reparses() -> None:
    combined = application_image().combine_bootloader_app(bootloader_image())

    for image in combined.split_bootloader_app():
        assert GBL3Image.from_bytes(image.serialize()) == image


def test_application_only_image_is_not_split() -> None:
    image = application_image()

    assert not image.is_combined_bootloader_app()
    assert image.split_bootloader_app() == [image]


def test_bootloader_only_image_is_not_split() -> None:
    image = GBL3Image(
        tags=[header(), app_info(GBL3ApplicationType.BOOTLOADER), BOOTLOADER]
    ).regenerate_crc()

    assert not image.is_combined_bootloader_app()
    assert image.split_bootloader_app() == [image]


def test_two_bootloaders_rejected() -> None:
    image = bootloader_image()

    with pytest.raises(ValueError, match="Both images contain bootloaders"):
        image.combine_bootloader_app(image)


def test_no_bootloader_rejected() -> None:
    image = application_image()

    with pytest.raises(ValueError, match="Neither image contains a bootloader"):
        image.combine_bootloader_app(image)


def test_mismatched_headers_rejected() -> None:
    other = GBL3Image(tags=[GBL3Header(version=2, type=GBL3Type.NONE), BOOTLOADER])

    with pytest.raises(ValueError, match="GBL headers do not match"):
        application_image().combine_bootloader_app(other)


def test_signed_images_cannot_be_combined() -> None:
    signed_bootloader = GBL3Image(
        tags=[header(GBL3Type.SIGNATURE_ECDSA_P256), BOOTLOADER]
    ).regenerate_crc()
    signed_app = GBL3Image(
        tags=[
            header(GBL3Type.SIGNATURE_ECDSA_P256),
            app_info(GBL3ApplicationType.ZIGBEE),
        ]
    ).regenerate_crc()

    with pytest.raises(ValueError, match="Cannot combine signed or encrypted images"):
        signed_app.combine_bootloader_app(signed_bootloader)


def test_signed_combined_image_cannot_be_split() -> None:
    image = GBL3Image(
        tags=[
            header(GBL3Type.SIGNATURE_ECDSA_P256),
            app_info(GBL3ApplicationType.ZIGBEE | GBL3ApplicationType.BOOTLOADER),
            BOOTLOADER,
            PROGRAM_DATA,
        ]
    ).regenerate_crc()

    with pytest.raises(ValueError, match="Cannot split signed or encrypted images"):
        image.split_bootloader_app()
