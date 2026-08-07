from __future__ import annotations

from cryptography.hazmat.primitives.asymmetric import ec
import pytest

from pygbl import (
    GBL3ApplicationInfo,
    GBL3ApplicationType,
    GBL3Compression,
    GBL3EraseProg,
    GBL3Header,
    GBL3Image,
    GBL3Metadata,
    GBL3Prog,
    GBL3ProgLZ4,
    GBL3ProgLZMA,
    GBL3Type,
)
from pygbl.compression import (
    LZMA_DICT_SIZE,
    LZMA_HEADER_SIZE,
    LZMA_PROPS_SIZE,
    ends_with_end_marker,
    lz4_compress,
    lz4_decompress,
    lzma_compress,
    lzma_decompress,
)

PRODUCT_ID = bytes(range(16))
# Compressible, but not so uniform that the result is trivially small
PAYLOAD = bytes((i * 3) % 211 for i in range(8192)) + b"\x00" * 4096


def build_image() -> GBL3Image:
    return GBL3Image(
        tags=[
            GBL3Header(version=3, type=GBL3Type.NONE),
            GBL3ApplicationInfo(
                type=GBL3ApplicationType.ZWAVE,
                version=1,
                capabilities=0,
                product_id=PRODUCT_ID,
            ),
            GBL3EraseProg(address=0x8006000, data=PAYLOAD),
            GBL3EraseProg(address=0x8106000, data=PAYLOAD[:1024]),
            GBL3Metadata(metadata=b'{"fw_type": "zwave_ncp"}'),
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


@pytest.mark.parametrize("size", [0, 1, 15, 368, 1384, 20000])
def test_lzma_writes_an_end_marker(size: int) -> None:
    """The bootloader fails the update without one, after flashing past the payload."""
    data = bytes((i * 7) % 251 for i in range(size))

    assert ends_with_end_marker(lzma_compress(data))


def test_end_marker_check_rejects_a_truncated_stream() -> None:
    compressed = lzma_compress(PAYLOAD)

    assert not ends_with_end_marker(compressed[:-1])


@pytest.mark.parametrize(("lc", "lp"), [(3, 0), (1, 2), (2, 1), (0, 3)])
def test_lzma_rejects_an_oversized_probability_model(lc: int, lp: int) -> None:
    """The decoder has 10 KiB for the model, which caps `lc + lp` at 2."""
    with pytest.raises(ValueError, match="lc \\+ lp"):
        lzma_compress(PAYLOAD, lc=lc, lp=lp)


@pytest.mark.parametrize("dict_size", [16384, 65536, 1 << 20])
def test_lzma_rejects_an_oversized_dictionary(dict_size: int) -> None:
    """The decoder has a statically sized 8 KiB dictionary buffer."""
    with pytest.raises(ValueError, match="dict_size"):
        lzma_compress(PAYLOAD, dict_size=dict_size)


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
        (GBL3Compression.LZMA, GBL3ProgLZMA),
        (GBL3Compression.LZ4, GBL3ProgLZ4),
    ],
)
def test_compress_replaces_program_data_tags(
    algorithm: GBL3Compression,
    tag_type: type[GBL3ProgLZMA] | type[GBL3ProgLZ4],
) -> None:
    compressed = build_image().compress(algorithm)

    assert len(compressed.get_tags(tag_type)) == 2
    assert compressed.get_tags(GBL3EraseProg) == []
    assert [t.address for t in compressed.get_tags(tag_type)] == [0x8006000, 0x8106000]


@pytest.mark.parametrize("algorithm", list(GBL3Compression))
def test_compress_decompress_roundtrip(algorithm: GBL3Compression) -> None:
    image = build_image()

    assert image.compress(algorithm).decompress().serialize() == image.serialize()


@pytest.mark.parametrize("algorithm", list(GBL3Compression))
def test_compressed_image_is_valid(algorithm: GBL3Compression) -> None:
    compressed = build_image().compress(algorithm)
    data = compressed.serialize()

    assert GBL3Image.from_bytes(data) == compressed


@pytest.mark.parametrize("algorithm", list(GBL3Compression))
def test_compress_leaves_other_tags_alone(algorithm: GBL3Compression) -> None:
    image = build_image()
    compressed = image.compress(algorithm)

    assert compressed.get_first_tag(GBL3Metadata) == image.get_first_tag(GBL3Metadata)
    assert compressed.get_first_tag(GBL3ApplicationInfo) == image.get_first_tag(
        GBL3ApplicationInfo
    )
    # Compression is not signalled in the header, only by the tag ids
    assert compressed.get_first_tag(GBL3Header).type == GBL3Type.NONE


def test_compress_handles_program_data1() -> None:
    """Both program data tag ids are interchangeable and both compress."""
    tags = list(build_image().tags)
    tags[2] = GBL3Prog(address=0x8006000, data=PAYLOAD)
    variant = GBL3Image(tags=tags).regenerate_crc()

    compressed = variant.compress(GBL3Compression.LZMA)

    assert compressed.get_tags(GBL3Prog) == []
    assert len(compressed.get_tags(GBL3ProgLZMA)) == 2


def test_decompress_is_a_noop_on_plain_images() -> None:
    image = build_image()

    assert image.decompress().serialize() == image.serialize()


def test_compress_then_encrypt_then_sign_roundtrip() -> None:
    """The full pipeline `commander` uses for Z-Wave firmware."""
    image = build_image()
    key = bytes(range(16))
    private_key = ec.generate_private_key(ec.SECP256R1())

    sealed = image.compress(GBL3Compression.LZMA).encrypt(key).sign(private_key)

    assert sealed.verify_signature(private_key.public_key())
    assert GBL3Image.from_bytes(sealed.serialize()) == sealed
    assert sealed.decrypt(key).decompress().serialize() == image.serialize()
