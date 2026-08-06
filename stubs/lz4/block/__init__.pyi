"""Minimal stubs for the parts of `lz4.block` this library uses.

python-lz4 implements these in a C extension and ships no type information. Both
functions return `bytearray` when `return_bytearray=True`; we never pass it, so the
return type is narrowed to `bytes`.
"""

class LZ4BlockError(Exception): ...

def compress(
    source: bytes,
    mode: str = ...,
    acceleration: int = ...,
    compression: int = ...,
    store_size: bool = ...,
    return_bytearray: bool = ...,
) -> bytes: ...
def decompress(
    source: bytes,
    uncompressed_size: int = ...,
    return_bytearray: bool = ...,
) -> bytes: ...
