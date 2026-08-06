from __future__ import annotations

from cryptography.hazmat.primitives.asymmetric import ec
import pytest

from pygbl import (
    GBLApplicationInfo,
    GBLApplicationType,
    GBLCompression,
    GBLHeader,
    GBLImage,
    GBLMetadata,
    GBLProgramData1,
    GBLProgramData2,
    GBLProgramDataLZ4,
    GBLProgramDataLZMA,
    GBLType,
)
from pygbl.compression import (
    LZMA_DICT_SIZE,
    LZMA_HEADER_SIZE,
    LZMA_PROPS_SIZE,
    lz4_compress,
    lz4_decompress,
    lzma_compress,
    lzma_decompress,
)

PRODUCT_ID = bytes(range(16))
# Compressible, but not so uniform that the result is trivially small
PAYLOAD = bytes((i * 3) % 211 for i in range(8192)) + b"\x00" * 4096


def build_image() -> GBLImage:
    return GBLImage(
        tags=[
            GBLHeader(version=3, type=GBLType.NONE),
            GBLApplicationInfo(
                type=GBLApplicationType.ZWAVE,
                version=1,
                capabilities=0,
                product_id=PRODUCT_ID,
            ),
            GBLProgramData2(address=0x8006000, data=PAYLOAD),
            GBLProgramData2(address=0x8106000, data=PAYLOAD[:1024]),
            GBLMetadata(metadata=b'{"fw_type": "zwave_ncp"}'),
        ]
    ).regenerate_crc()


def test_lzma_props_match_commander() -> None:
    """`commander` emits lc=1, lp=1, pb=2 with an 8 KiB dictionary: props byte 0x64."""
    compressed = lzma_compress(PAYLOAD)
    props = compressed[0]

    assert compressed[:LZMA_PROPS_SIZE] == bytes.fromhex("6400200000")
    assert props % 9 == 1  # lc
    assert (props // 9) % 5 == 1  # lp
    assert (props // 9) // 5 == 2  # pb
    assert int.from_bytes(compressed[1:5], "little") == LZMA_DICT_SIZE == 8192


def test_lzma_records_the_uncompressed_size() -> None:
    """Python writes an unknown-size marker by default; we write the real size."""
    compressed = lzma_compress(PAYLOAD)
    size = int.from_bytes(compressed[LZMA_PROPS_SIZE:LZMA_HEADER_SIZE], "little")

    assert size == len(PAYLOAD)


@pytest.mark.parametrize("size", [1, 15, 368, 1384, 20000])
def test_lzma_roundtrip(size: int) -> None:
    data = bytes((i * 7) % 251 for i in range(size))

    assert lzma_decompress(lzma_compress(data)) == data


@pytest.mark.parametrize("size", [1, 15, 368, 1384, 20000])
def test_lz4_roundtrip(size: int) -> None:
    data = bytes((i * 7) % 251 for i in range(size))

    assert lz4_decompress(lz4_compress(data)) == data


def test_lz4_emits_a_raw_block() -> None:
    """No frame magic: the bootloader reads raw LZ4 blocks."""
    compressed = lz4_compress(PAYLOAD)

    assert not compressed.startswith(bytes.fromhex("04224d18"))


def test_compression_actually_shrinks() -> None:
    assert len(lzma_compress(PAYLOAD)) < len(PAYLOAD)
    assert len(lz4_compress(PAYLOAD)) < len(PAYLOAD)


@pytest.mark.parametrize(
    ("algorithm", "tag_type"),
    [
        (GBLCompression.LZMA, GBLProgramDataLZMA),
        (GBLCompression.LZ4, GBLProgramDataLZ4),
    ],
)
def test_compress_replaces_program_data_tags(
    algorithm: GBLCompression,
    tag_type: type[GBLProgramDataLZMA] | type[GBLProgramDataLZ4],
) -> None:
    compressed = build_image().compress(algorithm)

    assert len(compressed.get_tags(tag_type)) == 2
    assert compressed.get_tags(GBLProgramData2) == []
    assert [t.address for t in compressed.get_tags(tag_type)] == [0x8006000, 0x8106000]


@pytest.mark.parametrize("algorithm", list(GBLCompression))
def test_compress_decompress_roundtrip(algorithm: GBLCompression) -> None:
    image = build_image()

    assert image.compress(algorithm).decompress().serialize() == image.serialize()


@pytest.mark.parametrize("algorithm", list(GBLCompression))
def test_compressed_image_is_valid(algorithm: GBLCompression) -> None:
    compressed = build_image().compress(algorithm)
    data = compressed.serialize()

    assert GBLImage.from_bytes(data) == compressed


@pytest.mark.parametrize("algorithm", list(GBLCompression))
def test_compress_leaves_other_tags_alone(algorithm: GBLCompression) -> None:
    image = build_image()
    compressed = image.compress(algorithm)

    assert compressed.get_first_tag(GBLMetadata) == image.get_first_tag(GBLMetadata)
    assert compressed.get_first_tag(GBLApplicationInfo) == image.get_first_tag(
        GBLApplicationInfo
    )
    # Compression is not signalled in the header, only by the tag ids
    assert compressed.get_first_tag(GBLHeader).type == GBLType.NONE


def test_compress_handles_program_data1() -> None:
    """Both program data tag ids are interchangeable and both compress."""
    tags = list(build_image().tags)
    tags[2] = GBLProgramData1(address=0x8006000, data=PAYLOAD)
    variant = GBLImage(tags=tags).regenerate_crc()

    compressed = variant.compress(GBLCompression.LZMA)

    assert compressed.get_tags(GBLProgramData1) == []
    assert len(compressed.get_tags(GBLProgramDataLZMA)) == 2


def test_decompress_is_a_noop_on_plain_images() -> None:
    image = build_image()

    assert image.decompress().serialize() == image.serialize()


def test_compress_then_encrypt_then_sign_roundtrip() -> None:
    """The full pipeline `commander` uses for Z-Wave firmware."""
    image = build_image()
    key = bytes(range(16))
    private_key = ec.generate_private_key(ec.SECP256R1())

    sealed = image.compress(GBLCompression.LZMA).encrypt(key).sign(private_key)

    assert sealed.verify_signature(private_key.public_key())
    assert GBLImage.from_bytes(sealed.serialize()) == sealed
    assert sealed.decrypt(key).decompress().serialize() == image.serialize()
