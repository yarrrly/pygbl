from __future__ import annotations

from typing import Union

from pygbl.ebl import (
    EBL_BLOCK_SIZE,
    EBL_MAGIC,
    EBLEncryptedData,
    EBLEncryptionHeader,
    EBLEncryptionInit,
    EBLEncryptionMac,
    EBLEnd,
    EBLEraseProgram,
    EBLHeader,
    EBLImage,
    EBLManufacturingProgram,
    EBLProgram,
    EBLTag,
    EBLTagBase,
    EBLTagId,
    EBLUnknownTag,
)
from pygbl.elf import (
    LoadableSegment,
    build_application_gbl,
    build_bootloader_gbl,
    read_application_info,
    read_application_properties,
    read_bootloader_version,
    read_loadable_segments,
)
from pygbl.gbl import (
    GBL_MAGIC,
    BootloaderVersion,
    GBLApplicationInfo,
    GBLApplicationType,
    GBLBootloader,
    GBLCertificateEcdsaP256,
    GBLCompression,
    GBLDelta,
    GBLDeltaLZ4,
    GBLDeltaLZMA,
    GBLEncryptedData,
    GBLEncryptionHeader,
    GBLEncryptionInitAesCcm,
    GBLEnd,
    GBLEraseProg,
    GBLHeader,
    GBLImage,
    GBLMetadata,
    GBLProg,
    GBLProgLZ4,
    GBLProgLZMA,
    GBLSeUpgrade,
    GBLSignature,
    GBLTag,
    GBLTagBase,
    GBLTagId,
    GBLType,
    GBLUnknownTag,
    GBLVersionDependency,
    GBLVersionDependencyImageType,
    VersionDependencyStatement,
)
from pygbl.types import (
    VALID_CRC32,
    GBLError,
    MissingDependencyError,
    ParseError,
    ValidationError,
    pad_to_multiple,
)

FirmwareImage = Union[GBLImage, EBLImage]


def parse_firmware_image(data: bytes, *, validate: bool = True) -> FirmwareImage:
    """Parse a Silicon Labs GBL or EBL firmware image."""
    data = bytes(data)

    if data.startswith(GBL_MAGIC):
        return GBLImage.from_bytes(data, validate=validate)

    if data.startswith(EBL_MAGIC):
        return EBLImage.from_bytes(data, validate=validate)

    raise ParseError("Data is neither a GBL nor an EBL image")


__all__ = [
    "EBL_BLOCK_SIZE",
    "EBL_MAGIC",
    "GBL_MAGIC",
    "VALID_CRC32",
    "BootloaderVersion",
    "EBLEncryptedData",
    "EBLEncryptionHeader",
    "EBLEncryptionInit",
    "EBLEncryptionMac",
    "EBLEnd",
    "EBLEraseProgram",
    "EBLHeader",
    "EBLImage",
    "EBLManufacturingProgram",
    "EBLProgram",
    "EBLTag",
    "EBLTagBase",
    "EBLTagId",
    "EBLUnknownTag",
    "FirmwareImage",
    "GBLApplicationInfo",
    "GBLApplicationType",
    "GBLBootloader",
    "GBLCompression",
    "GBLEncryptedData",
    "GBLEncryptionHeader",
    "GBLEncryptionInitAesCcm",
    "GBLEnd",
    "GBLError",
    "GBLHeader",
    "GBLImage",
    "GBLMetadata",
    "GBLProg",
    "GBLEraseProg",
    "GBLProgLZ4",
    "GBLProgLZMA",
    "GBLCertificateEcdsaP256",
    "GBLDelta",
    "GBLDeltaLZ4",
    "GBLDeltaLZMA",
    "GBLVersionDependency",
    "GBLVersionDependencyImageType",
    "VersionDependencyStatement",
    "GBLSeUpgrade",
    "GBLSignature",
    "GBLTag",
    "GBLTagBase",
    "GBLTagId",
    "GBLType",
    "GBLUnknownTag",
    "LoadableSegment",
    "MissingDependencyError",
    "ParseError",
    "ValidationError",
    "build_application_gbl",
    "build_bootloader_gbl",
    "pad_to_multiple",
    "parse_firmware_image",
    "read_application_info",
    "read_application_properties",
    "read_bootloader_version",
    "read_loadable_segments",
]
