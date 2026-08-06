from __future__ import annotations

import dataclasses
from typing import BinaryIO
import zlib

from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

from pygbl.gbl3 import (
    BootloaderVersion,
    GBL3ApplicationInfo,
    GBL3Bootloader,
    GBL3EraseProg,
    GBL3Header,
    GBL3Image,
    GBL3Metadata,
    GBL3TagBase,
    GBL3Type,
)
from pygbl.types import ParseError

# The SDK's `application_properties_t`: a magic GUID, then structVersion, signatureType
# and signatureLocation, and then the `ApplicationData_t` block that becomes the
# application info tag verbatim.
APP_PROPERTIES_MAGIC = bytes.fromhex("13b779fac925ddb7adf3cfe0f1b614b8")
APP_PROPERTIES_SYMBOLS = ("sl_app_properties", "application_properties")
APP_INFO_OFFSET = 28
APP_INFO_SIZE = 28

GBL3_SPEC_VERSION = 3


@dataclasses.dataclass(frozen=True)
class LoadableSegment:
    address: int
    data: bytes


def _loadable_segments(elf: ELFFile) -> list[LoadableSegment]:
    raw = sorted(
        (
            (segment["p_paddr"], segment.data())
            for segment in elf.iter_segments()
            if segment["p_type"] == "PT_LOAD" and segment["p_filesz"] > 0
        ),
        key=lambda item: item[0],
    )
    merged: list[LoadableSegment] = []

    for address, data in raw:
        if merged and merged[-1].address + len(merged[-1].data) == address:
            merged[-1] = LoadableSegment(
                address=merged[-1].address, data=merged[-1].data + data
            )
        else:
            merged.append(LoadableSegment(address=address, data=data))

    if not merged:
        raise ParseError("ELF contains no loadable segments")

    return merged


def _application_properties(elf: ELFFile) -> bytes:
    symtab = elf.get_section_by_name(".symtab")

    if not isinstance(symtab, SymbolTableSection):
        raise ParseError("ELF has no symbol table")

    for name in APP_PROPERTIES_SYMBOLS:
        symbols = symtab.get_symbol_by_name(name)

        if symbols:
            symbol = symbols[0]
            break
    else:
        raise ParseError(f"ELF has none of the symbols {APP_PROPERTIES_SYMBOLS}")

    address = symbol["st_value"]
    size = symbol["st_size"]

    for segment in _loadable_segments(elf):
        offset = address - segment.address

        if 0 <= offset < len(segment.data):
            properties = segment.data[offset : offset + size]
            break
    else:
        raise ParseError(f"{name} at {address:#010x} is not in a loadable segment")

    if not properties.startswith(APP_PROPERTIES_MAGIC):
        raise ParseError(f"Bad application properties magic: {properties[:16].hex()}")

    return properties


def _application_info(elf: ELFFile) -> GBL3ApplicationInfo:
    properties = _application_properties(elf)

    return GBL3ApplicationInfo.from_payload(
        properties[APP_INFO_OFFSET : APP_INFO_OFFSET + APP_INFO_SIZE]
    )


def read_loadable_segments(elf_file: BinaryIO) -> list[LoadableSegment]:
    """The ELF's loadable segments, keyed by physical address and merged where contiguous.

    Physical rather than virtual addresses: initialized data is stored in flash but
    linked at its RAM address, so keying on the virtual address would place it wrongly
    and split what is really one contiguous run.
    """
    return _loadable_segments(ELFFile(elf_file))


def read_application_properties(elf_file: BinaryIO) -> bytes:
    """The raw `application_properties_t` struct embedded in the image."""
    return _application_properties(ELFFile(elf_file))


def read_application_info(elf_file: BinaryIO) -> GBL3ApplicationInfo:
    return _application_info(ELFFile(elf_file))


def read_bootloader_version(elf_file: BinaryIO) -> BootloaderVersion:
    return BootloaderVersion.from_int(_application_info(ELFFile(elf_file)).version)


def build_application_gbl3(
    elf_file: BinaryIO, *, metadata: bytes | None = None
) -> GBL3Image:
    """Build an application GBL from a linked ELF."""
    elf = ELFFile(elf_file)
    tags: list[GBL3TagBase] = [
        GBL3Header(version=GBL3_SPEC_VERSION, type=GBL3Type.NONE),
        _application_info(elf),
        *[
            GBL3EraseProg(address=segment.address, data=segment.data)
            for segment in _loadable_segments(elf)
        ],
    ]

    if metadata is not None:
        tags.append(GBL3Metadata(metadata=metadata))

    return GBL3Image(tags=tags).regenerate_crc()


def build_bootloader_gbl3(
    elf_file: BinaryIO, *, metadata: bytes | None = None
) -> GBL3Image:
    """Build a bootloader GBL from a linked ELF."""
    elf = ELFFile(elf_file)
    segments = _loadable_segments(elf)

    if len(segments) != 1:
        raise ParseError(
            f"Expected one contiguous bootloader segment, got {len(segments)}"
        )

    segment = segments[0]
    tags: list[GBL3TagBase] = [
        GBL3Header(version=GBL3_SPEC_VERSION, type=GBL3Type.NONE),
        GBL3Bootloader(
            version=BootloaderVersion.from_int(_application_info(elf).version),
            address=segment.address,
            # The bootloader payload is followed by a CRC32 of itself
            data=segment.data
            + (zlib.crc32(segment.data) & 0xFFFFFFFF).to_bytes(4, "little"),
        ),
    ]

    if metadata is not None:
        tags.append(GBL3Metadata(metadata=metadata))

    return GBL3Image(tags=tags).regenerate_crc()
