"""Test against `commander`."""

from __future__ import annotations

import functools
import pathlib
import shutil
import subprocess
import tempfile

import pytest

import pygbl
from pygbl import (
    GBL3Compression,
    GBL3End,
    GBL3Image,
    GBL3ProgLZ4,
    GBL3ProgLZMA,
    read_encryption_key,
)
from pygbl.compression import (
    LZMA_HEADER_SIZE,
    LZMA_MAX_DICT_SIZE,
    LZMA_MAX_LC_PLUS_LP,
    LZMA_PROPS_SIZE,
    ends_with_end_marker,
    lzma_compress,
    lzma_decompress,
)

FILES = pathlib.Path(__file__).parent / "files"
COMMANDER = shutil.which("commander") or ""

# The two ELF segments of the ZWA-2 controller, our only committed LZMA reference
ZWA2 = "zwa2_controller"
SMALL_ADDRESS = 0x08006000
LARGE_ADDRESS = 0x08006180
COMMANDER_LZMA_LENGTHS = {SMALL_ADDRESS: 134, LARGE_ADDRESS: 147238}
OUR_LZMA_LENGTHS = {SMALL_ADDRESS: 134, LARGE_ADDRESS: 147237}

IMAGES = sorted(
    elf.name.removesuffix(".out")
    for elf in FILES.glob("*.out")
    if (FILES / f"{elf.name.removesuffix('.out')}_metadata.json").exists()
)

if not COMMANDER:
    pytest.skip("`commander` is not installed", allow_module_level=True)

CASES = [(base, algorithm) for base in IMAGES for algorithm in (None, *GBL3Compression)]
STORED = [base for base in IMAGES if (FILES / f"{base}_commander.gbl").exists()]


def case_id(case: tuple[str, GBL3Compression | None]) -> str:
    base, algorithm = case

    return f"{base}-{algorithm.value if algorithm else 'none'}"


def vendor_keys(base: str) -> tuple[pathlib.Path, pathlib.Path] | None:
    """The encryption and signing keys for an image, if the corpus ships them."""
    prefix = base.split("_", maxsplit=1)[0]
    encrypt = FILES / f"{prefix}_vendor_encrypt.key"
    sign = FILES / f"{prefix}_vendor_sign.key"

    return (encrypt, sign) if encrypt.exists() and sign.exists() else None


def build_ours(base: str, algorithm: GBL3Compression | None) -> GBL3Image:
    is_bootloader = "bootloader" in base
    builder = (
        pygbl.build_bootloader_gbl3 if is_bootloader else pygbl.build_application_gbl3
    )

    with (FILES / f"{base}.out").open("rb") as elf_file:
        image = builder(
            elf_file, metadata=(FILES / f"{base}_metadata.json").read_bytes()
        )

    return image if algorithm is None else image.compress(algorithm)


@functools.cache
def stored(base: str) -> GBL3Image:
    """The committed `commander` output for an image, decrypted."""
    image = GBL3Image.from_bytes((FILES / f"{base}_commander.gbl").read_bytes())
    keys = vendor_keys(base)

    return (
        image
        if keys is None
        else image.decrypt(read_encryption_key(keys[0].read_text()))
    )


@functools.cache
def build(base: str, algorithm: GBL3Compression | None) -> tuple[GBL3Image, GBL3Image]:
    """Build `base` with us and with `commander`, returning both decrypted."""
    keys = vendor_keys(base)

    with tempfile.TemporaryDirectory() as directory:
        output = pathlib.Path(directory) / f"{base}.gbl"

        # fmt: off
        command = [
            COMMANDER,
            "gbl", "create", str(output),
            "--bootloader" if "bootloader" in base else "--app",
            str(FILES / f"{base}.out"),
            "--metadata", str(FILES / f"{base}_metadata.json"),
        ]
        # fmt: on

        if algorithm is not None:
            command += ["--compress", algorithm.value]

        if keys is not None:
            command += ["--encrypt", str(keys[0]), "--sign", str(keys[1])]

        subprocess.run(command, check=True, capture_output=True)
        theirs = GBL3Image.from_bytes(output.read_bytes())

    if keys is not None:
        theirs = theirs.decrypt(read_encryption_key(keys[0].read_text()))

    return build_ours(base, algorithm), theirs


def lzma_tag(image: GBL3Image, address: int) -> GBL3ProgLZMA:
    return next(t for t in image.get_tags(GBL3ProgLZMA) if t.address == address)


# Against the committed reference images


@pytest.mark.parametrize("base", STORED)
def test_stored_reference_is_reproduced(base: str) -> None:
    """Every committed reference is uncompressed but ZWA-2, and those match exactly."""
    reference = stored(base)

    if any(isinstance(t, (GBL3ProgLZMA, GBL3ProgLZ4)) for t in reference.tags):
        pytest.skip("reference is compressed")

    assert build_ours(base, None).tags == reference.tags


def test_stored_lzma_props_match() -> None:
    """The props bytes are what the device reads to size its decoder buffers."""
    for tag in stored(ZWA2).get_tags(GBL3ProgLZMA):
        ours = lzma_compress(lzma_decompress(tag.data))

        assert (
            ours[:LZMA_PROPS_SIZE]
            == tag.data[:LZMA_PROPS_SIZE]
            == bytes.fromhex("6400200000")
        )
        assert (
            ours[LZMA_PROPS_SIZE:LZMA_HEADER_SIZE]
            == tag.data[LZMA_PROPS_SIZE:LZMA_HEADER_SIZE]
        )


def test_stored_lzma_lengths_are_stable() -> None:
    """Pins the encoder parameters: any change to them moves these lengths."""
    tags = stored(ZWA2).get_tags(GBL3ProgLZMA)

    assert {t.address: len(t.data) for t in tags} == COMMANDER_LZMA_LENGTHS
    assert {
        t.address: len(lzma_compress(lzma_decompress(t.data))) for t in tags
    } == OUR_LZMA_LENGTHS


def test_stored_small_lzma_tag_is_byte_identical() -> None:
    tag = lzma_tag(stored(ZWA2), SMALL_ADDRESS)

    assert lzma_compress(lzma_decompress(tag.data)) == tag.data


def test_stored_large_lzma_tag_differs_only_in_encoder_vintage() -> None:
    """`commander` links an LZMA SDK 16.04 era encoder that liblzma cannot reproduce."""
    tag = lzma_tag(stored(ZWA2), LARGE_ADDRESS)
    ours = lzma_compress(lzma_decompress(tag.data))

    # The two streams share a 4016 byte prefix, covering 6206 bytes of payload, and are
    # unrelated after that. The single byte is where two entirely different encodings
    # happen to land for this payload, not a localised difference.
    assert ours != tag.data
    assert len(ours) - len(tag.data) == -1


@pytest.mark.parametrize("path", sorted(FILES.glob("*.gbl")), ids=lambda p: p.name)
def test_real_images_respect_the_bootloader_limits(path: pathlib.Path) -> None:
    """Every LZMA tag in the wild fits the decoder's 10 KiB probs and 8 KiB dictionary."""
    image = pygbl.parse_firmware_image(path.read_bytes())

    if not isinstance(image, GBL3Image) or not image.get_tags(
        GBL3ProgLZMA, allow_missing=True
    ):
        pytest.skip("no LZMA tags")

    for tag in image.get_tags(GBL3ProgLZMA):
        props = tag.data[0]

        assert props % 9 + (props // 9) % 5 <= LZMA_MAX_LC_PLUS_LP
        assert (
            int.from_bytes(tag.data[1:LZMA_PROPS_SIZE], "little") <= LZMA_MAX_DICT_SIZE
        )


# Against a live `commander`


@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_images_without_compressed_tags_are_byte_identical(
    case: tuple[str, GBL3Compression | None],
) -> None:
    """Uncompressed images and bootloaders reproduce `commander` exactly."""
    ours, theirs = build(*case)

    if any(isinstance(t, (GBL3ProgLZMA, GBL3ProgLZ4)) for t in ours.tags):
        pytest.skip("image has compressed tags")

    assert ours.serialize(block_size=4) == theirs.serialize()


@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_only_a_compressed_payload_and_its_crc_may_differ(
    case: tuple[str, GBL3Compression | None],
) -> None:
    """Everything else `commander` writes is reproduced exactly."""
    ours, theirs = build(*case)

    differing = [type(a) for a, b in zip(ours.tags, theirs.tags, strict=True) if a != b]

    assert differing in (
        [],
        [GBL3ProgLZMA, GBL3End],
        [GBL3ProgLZ4, GBL3End],
    )


@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_decompressing_both_gives_the_same_image(
    case: tuple[str, GBL3Compression | None],
) -> None:
    """What reaches flash is identical even where the compressed bytes are not."""
    ours, theirs = build(*case)

    # Compared by tag, since `commander` pads the file itself and we do not
    assert ours.decompress().tags == theirs.decompress().tags


@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_lzma_props_match(case: tuple[str, GBL3Compression | None]) -> None:
    """The props bytes are what the device reads to size its decoder buffers."""
    ours, theirs = build(*case)

    assert [
        t.data[:LZMA_PROPS_SIZE]
        for t in ours.get_tags(GBL3ProgLZMA, allow_missing=True)
    ] == [
        t.data[:LZMA_PROPS_SIZE]
        for t in theirs.get_tags(GBL3ProgLZMA, allow_missing=True)
    ]


@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_both_encoders_write_lzma_end_markers(
    case: tuple[str, GBL3Compression | None],
) -> None:
    """The bootloader rejects a stream that does not finish with one."""
    ours, theirs = build(*case)

    for image in (ours, theirs):
        assert all(
            ends_with_end_marker(t.data)
            for t in image.get_tags(GBL3ProgLZMA, allow_missing=True)
        )
