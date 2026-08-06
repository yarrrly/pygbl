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
    EBLTagId,
    EBLUnknownTag,
)
from pygbl.gbl import (
    GBL_MAGIC,
    BootloaderVersion,
    GBLApplicationInfo,
    GBLApplicationType,
    GBLEncryptedData,
    GBLEncryptionHeader,
    GBLEncryptionInitAesCcm,
    GBLEnd,
    GBLHeader,
    GBLImage,
    GBLMetadata,
    GBLProgramData1,
    GBLProgramData2,
    GBLProgramDataLZ4,
    GBLProgramDataLZMA,
    GBLSeUpgrade,
    GBLSignature,
    GBLTag,
    GBLTagId,
    GBLType,
    GBLUnknownTag,
)
from pygbl.types import (
    VALID_CRC32,
    GBLError,
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
    "EBLTagId",
    "EBLUnknownTag",
    "FirmwareImage",
    "GBLApplicationInfo",
    "GBLApplicationType",
    "GBLEncryptedData",
    "GBLEncryptionHeader",
    "GBLEncryptionInitAesCcm",
    "GBLEnd",
    "GBLError",
    "GBLHeader",
    "GBLImage",
    "GBLMetadata",
    "GBLProgramData1",
    "GBLProgramData2",
    "GBLProgramDataLZ4",
    "GBLProgramDataLZMA",
    "GBLSeUpgrade",
    "GBLSignature",
    "GBLTag",
    "GBLTagId",
    "GBLType",
    "GBLUnknownTag",
    "ParseError",
    "ValidationError",
    "pad_to_multiple",
    "parse_firmware_image",
]
