from __future__ import annotations

import lzma

from pygbl.types import MissingDependencyError, ValidationError

try:
    import lz4.block
except ImportError:
    HAVE_LZ4 = False
else:
    HAVE_LZ4 = True

# What a corrupt stream, or one larger than the output limit, raises
DECOMPRESSION_ERRORS: tuple[type[Exception], ...] = (lzma.LZMAError, ValidationError)

if HAVE_LZ4:
    DECOMPRESSION_ERRORS += (lz4.block.LZ4BlockError,)

LZMA_LC = 1
LZMA_LP = 1
LZMA_PB = 2
LZMA_DICT_SIZE = 8192
LZMA_MAX_LC_PLUS_LP = 2
LZMA_MAX_DICT_SIZE = 8192
LZMA_NICE_LEN = 32
LZMA_DEPTH = 32

LZMA_PROPS_SIZE = 5
LZMA_SIZE_FIELD_SIZE = 8
LZMA_HEADER_SIZE = LZMA_PROPS_SIZE + LZMA_SIZE_FIELD_SIZE

# Raw LZ4 blocks do not record their decompressed size, so decompression needs an upper
# bound on the output. Program data tags are bounded by the device's flash size.
LZ4_MAX_DECOMPRESSED_SIZE = 4 * 1024 * 1024


def lzma_filters(
    *,
    lc: int = LZMA_LC,
    lp: int = LZMA_LP,
    pb: int = LZMA_PB,
    dict_size: int = LZMA_DICT_SIZE,
    nice_len: int = LZMA_NICE_LEN,
    depth: int = LZMA_DEPTH,
) -> list[dict[str, int]]:
    # The device rejects an image it cannot allocate buffers for, so catch it here
    # rather than at flash time.
    if lc + lp > LZMA_MAX_LC_PLUS_LP:
        raise ValueError(
            f"lc + lp must be at most {LZMA_MAX_LC_PLUS_LP}, got {lc} + {lp}"
        )

    if dict_size > LZMA_MAX_DICT_SIZE:
        raise ValueError(
            f"dict_size must be at most {LZMA_MAX_DICT_SIZE}, got {dict_size}"
        )

    return [
        {
            "id": lzma.FILTER_LZMA1,
            "lc": lc,
            "lp": lp,
            "pb": pb,
            "dict_size": dict_size,
            "mode": lzma.MODE_NORMAL,
            "mf": lzma.MF_BT4,
            "nice_len": nice_len,
            "depth": depth,
        }
    ]


def ends_with_end_marker(container: bytes) -> bool:
    """Check that an `.lzma` container's stream terminates with an end marker."""
    # `eof` alone is not enough: the decompressor stops at the declared size and never
    # looks for the marker. Declaring the size unknown again forces it to rely on one.
    probe = (
        container[:LZMA_PROPS_SIZE]
        + b"\xff" * LZMA_SIZE_FIELD_SIZE
        + container[LZMA_HEADER_SIZE:]
    )

    decompressor = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE)
    decompressor.decompress(probe)

    return decompressor.eof


def lzma_compress(data: bytes, **kwargs: int) -> bytes:
    """Compress into the `.lzma` container that the Gecko bootloader expects."""
    compressed = lzma.compress(
        data, format=lzma.FORMAT_ALONE, filters=lzma_filters(**kwargs)
    )

    # Python writes an unknown-size marker into the 8 byte size field. The bootloader
    # skips the field entirely, but `commander` writes the real size, so we match it.
    container = (
        compressed[:LZMA_PROPS_SIZE]
        + len(data).to_bytes(LZMA_SIZE_FIELD_SIZE, "little")
        + compressed[LZMA_HEADER_SIZE:]
    )

    # The bootloader requires `LZMA_STATUS_FINISHED_WITH_MARK` and fails the update
    # otherwise, after having already flashed past the end of the payload.
    if not ends_with_end_marker(container):
        raise ValidationError("LZMA stream does not end with an end marker")

    return container


def lzma_decompress(data: bytes, *, max_size: int | None = None) -> bytes:
    if max_size is None:
        return lzma.decompress(data, format=lzma.FORMAT_ALONE)

    decompressor = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE)
    # One byte past the limit is enough to tell that the stream is too large
    data = decompressor.decompress(data, max_length=max_size + 1)

    if len(data) > max_size:
        raise ValidationError(f"LZMA stream expands to more than {max_size} bytes")

    if not decompressor.eof:
        raise ValidationError("LZMA stream ends before its end marker")

    return data


def require_lz4() -> None:
    if not HAVE_LZ4:
        raise MissingDependencyError(
            "LZ4 compression needs the `lz4` package."
            " Install it with `pip install pygbl[lz4]`."
        )


def lz4_compress(data: bytes) -> bytes:
    """Compress into a raw LZ4 block, without a frame header or size prefix."""
    require_lz4()

    return lz4.block.compress(data, store_size=False)


def lz4_decompress(data: bytes, *, max_size: int = LZ4_MAX_DECOMPRESSED_SIZE) -> bytes:
    require_lz4()

    return lz4.block.decompress(data, uncompressed_size=max_size)
