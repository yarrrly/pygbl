from __future__ import annotations

from typing import TypeVar

# CRC32(anything | CRC32(anything)) == CRC32(0x00000000)
VALID_CRC32 = 0x2144DF1C


class GBLError(Exception):
    """Base class for every error raised by this library."""


class ParseError(GBLError):
    """An image is malformed and cannot be parsed."""


class ValidationError(GBLError):
    """An image parsed successfully but is not valid."""


class MissingDependencyError(GBLError):
    """An optional dependency is needed for the requested operation."""


def pad_to_multiple(data: bytes, multiple: int, padding: bytes = b"\xff") -> bytes:
    if len(padding) != 1:
        raise ValueError(f"Padding must be a single byte, got {padding!r}")

    if multiple <= 1 or len(data) % multiple == 0:
        return data

    return data + padding * (multiple - len(data) % multiple)


def check_length(payload: bytes, size: int, what: str) -> None:
    if len(payload) != size:
        raise ParseError(f"{what} must be {size} bytes, got {len(payload)}")


T = TypeVar("T")
