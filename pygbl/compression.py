from __future__ import annotations

import lzma

from pygbl.types import MissingDependencyError

try:
    import lz4.block
except ImportError:
    HAVE_LZ4 = False
else:
    HAVE_LZ4 = True

# The Gecko bootloader's LZMA decoder is memory constrained. Its probability model needs
# `4 KiB + 1.5 KiB * (1 << (lc + lp))` of RAM and its dictionary buffer is statically
# sized, so images cannot use the LZMA defaults of lc=3, lp=0 and a large dictionary.
# `commander` emits lc=1, lp=1, pb=2 with an 8 KiB dictionary.
LZMA_LC = 1
LZMA_LP = 1
LZMA_PB = 2
LZMA_DICT_SIZE = 8192

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
) -> list[dict[str, int]]:
    return [
        {
            "id": lzma.FILTER_LZMA1,
            "lc": lc,
            "lp": lp,
            "pb": pb,
            "dict_size": dict_size,
        }
    ]


def lzma_compress(data: bytes, **kwargs: int) -> bytes:
    """Compress into the `.lzma` container that the Gecko bootloader expects."""
    compressed = lzma.compress(
        data, format=lzma.FORMAT_ALONE, filters=lzma_filters(**kwargs)
    )

    # Python writes an unknown-size marker into the 8 byte size field. The bootloader
    # skips the field entirely, but `commander` writes the real size, so we match it.
    return (
        compressed[:LZMA_PROPS_SIZE]
        + len(data).to_bytes(LZMA_SIZE_FIELD_SIZE, "little")
        + compressed[LZMA_HEADER_SIZE:]
    )


def lzma_decompress(data: bytes) -> bytes:
    return lzma.decompress(data, format=lzma.FORMAT_ALONE)


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
