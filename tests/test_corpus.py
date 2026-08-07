"""Round-trip every real-world image in `tests/files` byte-for-byte.

The files are a local corpus and are not committed. Each one is the smallest
representative of a distinct structural signature (tag sequence, header flags and
trailing-data class) taken from a much larger survey of images found in the wild.
"""

from __future__ import annotations

import pathlib

import pytest

from pygbl import EBLImage, GBL3Image, parse_firmware_image

FILES = pathlib.Path(__file__).parent / "files"
IMAGES = sorted(p for p in FILES.glob("*") if p.suffix in (".gbl", ".ebl"))


def image_id(path: pathlib.Path) -> str:
    return path.name


if not IMAGES:
    pytest.skip("no corpus files present", allow_module_level=True)


@pytest.mark.parametrize("path", IMAGES, ids=image_id)
def test_roundtrip_exact(path: pathlib.Path) -> None:
    """Tags plus whatever followed them reproduce the file byte-for-byte."""
    data = path.read_bytes()
    parsed = parse_firmware_image(data)
    assert isinstance(parsed, (GBL3Image, EBLImage))

    image, trailing = type(parsed).deserialize(data)

    assert image.serialize(block_size=1) + trailing == data


@pytest.mark.parametrize("path", IMAGES, ids=image_id)
def test_expected_type(path: pathlib.Path) -> None:
    image = parse_firmware_image(path.read_bytes())
    expected = GBL3Image if path.suffix == ".gbl" else EBLImage

    assert isinstance(image, expected)


@pytest.mark.parametrize("path", IMAGES, ids=image_id)
def test_no_unknown_tags(path: pathlib.Path) -> None:
    """Every tag in the corpus should be one we model explicitly."""
    image = parse_firmware_image(path.read_bytes())
    assert isinstance(image, (GBL3Image, EBLImage))

    unknown = [t for t in image.tags if type(t).__name__.endswith("UnknownTag")]

    assert unknown == []


@pytest.mark.parametrize("path", IMAGES, ids=image_id)
def test_regenerate_crc_is_stable(path: pathlib.Path) -> None:
    """Recomputing the CRC of an untouched image reproduces it exactly."""
    data = path.read_bytes()
    image = parse_firmware_image(data)
    assert isinstance(image, (GBL3Image, EBLImage))

    image, trailing = type(image).deserialize(data)

    assert image.regenerate_crc() == image
    assert image.regenerate_crc().serialize(block_size=1) + trailing == data


@pytest.mark.parametrize("path", IMAGES, ids=image_id)
def test_block_padding_extends_the_tag_region(path: pathlib.Path) -> None:
    data = path.read_bytes()
    image = parse_firmware_image(data)
    assert isinstance(image, (GBL3Image, EBLImage))

    unpadded = image.serialize(block_size=1)
    padded = image.serialize(block_size=128)

    assert padded.startswith(unpadded)
    assert len(padded) % 128 == 0
    assert padded[len(unpadded) :].strip(b"\xff") == b""
