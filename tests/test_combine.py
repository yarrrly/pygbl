from __future__ import annotations

import pytest

from pygbl import (
    BootloaderVersion,
    GBLApplicationInfo,
    GBLApplicationType,
    GBLEraseProg,
    GBLHeader,
    GBLImage,
    GBLMetadata,
    GBLType,
)
from pygbl.gbl3 import GBLBootloader

PRODUCT_ID = bytes(range(16))
BOOTLOADER = GBLBootloader(
    version=BootloaderVersion(major=2, minor=4, customer=3),
    address=0x0,
    data=b"\xbb" * 64,
)
PROGRAM_DATA = GBLEraseProg(address=0x6000, data=b"\xaa" * 128)
METADATA = GBLMetadata(metadata=b'{"fw_type": "zigbee_ncp"}')


def header(type: GBLType = GBLType.NONE) -> GBLHeader:
    return GBLHeader(version=3, type=type)


def app_info(app_type: GBLApplicationType) -> GBLApplicationInfo:
    return GBLApplicationInfo(
        type=app_type, version=1, capabilities=0, product_id=PRODUCT_ID
    )


def bootloader_image() -> GBLImage:
    return GBLImage(tags=[header(), BOOTLOADER]).regenerate_crc()


def application_image() -> GBLImage:
    return GBLImage(
        tags=[header(), app_info(GBLApplicationType.ZIGBEE), PROGRAM_DATA, METADATA]
    ).regenerate_crc()


def test_combine() -> None:
    combined = application_image().combine_bootloader_app(bootloader_image())

    combined.validate()

    assert combined.get_first_tag(GBLApplicationInfo).type == (
        GBLApplicationType.ZIGBEE | GBLApplicationType.BOOTLOADER
    )
    assert combined.get_first_tag(GBLBootloader) == BOOTLOADER
    assert combined.get_first_tag(GBLEraseProg) == PROGRAM_DATA
    assert combined.get_first_tag(GBLMetadata) == METADATA
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
        assert GBLImage.from_bytes(image.serialize()) == image


def test_application_only_image_is_not_split() -> None:
    image = application_image()

    assert not image.is_combined_bootloader_app()
    assert image.split_bootloader_app() == [image]


def test_bootloader_only_image_is_not_split() -> None:
    image = GBLImage(
        tags=[header(), app_info(GBLApplicationType.BOOTLOADER), BOOTLOADER]
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
    other = GBLImage(tags=[GBLHeader(version=2, type=GBLType.NONE), BOOTLOADER])

    with pytest.raises(ValueError, match="GBL headers do not match"):
        application_image().combine_bootloader_app(other)


def test_signed_images_cannot_be_combined() -> None:
    signed_bootloader = GBLImage(
        tags=[header(GBLType.SIGNATURE_ECDSA_P256), BOOTLOADER]
    ).regenerate_crc()
    signed_app = GBLImage(
        tags=[header(GBLType.SIGNATURE_ECDSA_P256), app_info(GBLApplicationType.ZIGBEE)]
    ).regenerate_crc()

    with pytest.raises(ValueError, match="Cannot combine signed or encrypted images"):
        signed_app.combine_bootloader_app(signed_bootloader)


def test_signed_combined_image_cannot_be_split() -> None:
    image = GBLImage(
        tags=[
            header(GBLType.SIGNATURE_ECDSA_P256),
            app_info(GBLApplicationType.ZIGBEE | GBLApplicationType.BOOTLOADER),
            BOOTLOADER,
            PROGRAM_DATA,
        ]
    ).regenerate_crc()

    with pytest.raises(ValueError, match="Cannot split signed or encrypted images"):
        image.split_bootloader_app()
