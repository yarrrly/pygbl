"""GBLv4, checked against real Series 3 images from the community OTA repository.

The fixtures are a local corpus and are not committed. They are Philips Hue SiMG301
images, extracted from the `UPGRADE_IMAGE` and vendor subelements of their OTA files.
"""

from __future__ import annotations

import pathlib

import pytest

from pygbl import (
    GBL4_MAGIC,
    GBL4BundleVersion,
    GBL4Feature,
    GBL4Image,
    GBL4Manifest,
    GBL4ManifestFinish,
    GBL4ManifestInfo,
    GBL4MemorySection,
    GBL4MemorySectionBlob,
    GBL4MemorySectionInfo,
    GBL4Pad,
    GBL4Root,
    GBL4Signature,
    GBL4TagId,
    GBL4UnknownTag,
    GBL4UpdateMemorySection,
    GBL4UpdateProcess,
    HashValue,
    ParseError,
    parse_firmware_image,
)

FILES = pathlib.Path(__file__).parent / "files"
IMAGES = sorted(FILES.glob("*.gbl4"))

if not IMAGES:
    pytest.skip("no GBLv4 fixtures present", allow_module_level=True)


def load(path: pathlib.Path) -> GBL4Image:
    image = parse_firmware_image(path.read_bytes())
    assert isinstance(image, GBL4Image)

    return image


@pytest.mark.parametrize("path", IMAGES, ids=lambda p: p.name)
def test_roundtrip_exact(path: pathlib.Path) -> None:
    data = path.read_bytes()

    assert load(path).serialize() == data


@pytest.mark.parametrize("path", IMAGES, ids=lambda p: p.name)
def test_dispatches_to_gbl4(path: pathlib.Path) -> None:
    assert path.read_bytes().startswith(GBL4_MAGIC)
    assert isinstance(parse_firmware_image(path.read_bytes()), GBL4Image)


@pytest.mark.parametrize("path", IMAGES, ids=lambda p: p.name)
def test_no_unknown_tags(path: pathlib.Path) -> None:
    """Every tag in a real image should be one we model."""
    assert load(path).get_tags(GBL4UnknownTag, allow_missing=True) == []


@pytest.mark.parametrize("path", IMAGES, ids=lambda p: p.name)
def test_expected_structure(path: pathlib.Path) -> None:
    """A root containing a manifest, which nests the update process."""
    image = load(path)

    assert isinstance(image.root, GBL4Root)
    assert int(image.root.tag_id) == GBL4TagId.V4

    manifest = image.get_first_tag(GBL4Manifest)
    assert manifest in image.root.children

    # The update process is inside the manifest, not a sibling of it
    update_process = image.get_first_tag(GBL4UpdateProcess)
    assert update_process in manifest.children

    # And the manifest is closed by a zero-length marker
    finish = image.get_first_tag(GBL4ManifestFinish)
    assert finish.serialize_payload() == b""


@pytest.mark.parametrize("path", IMAGES, ids=lambda p: p.name)
def test_manifest_fields(path: pathlib.Path) -> None:
    image = load(path)

    info = image.get_first_tag(GBL4ManifestInfo)
    assert GBL4Feature.COMPRESSION in info.features
    assert GBL4Feature.ENCRYPTION_AESCCM in info.features

    bundle = image.get_first_tag(GBL4BundleVersion)
    assert len(bundle.product_id) == 16


@pytest.mark.parametrize("path", IMAGES, ids=lambda p: p.name)
def test_memory_sections(path: pathlib.Path) -> None:
    """Each memory section pairs a 212 byte info tag with its blob."""
    image = load(path)
    sections = image.get_tags(GBL4MemorySection)

    assert sections

    for section in sections:
        types = [type(child) for child in section.children]
        assert types == [GBL4MemorySectionInfo, GBL4MemorySectionBlob]

    for info in image.get_tags(GBL4MemorySectionInfo):
        assert len(info.serialize_payload()) == 212
        assert len(info.nonce) == 12


@pytest.mark.parametrize("path", IMAGES, ids=lambda p: p.name)
def test_hash_values_are_variable_length(path: pathlib.Path) -> None:
    """SHA-256 hashes occupy 32 bytes, not the 64 the SDK union reserves."""
    image = load(path)

    for update in image.get_tags(GBL4UpdateMemorySection):
        assert len(update.hash.value) in (32, 64)

    signature = image.get_first_tag(GBL4Signature)
    assert len(signature.signature.value) == 64


@pytest.mark.parametrize("path", IMAGES, ids=lambda p: p.name)
def test_padding_is_preserved(path: pathlib.Path) -> None:
    """`PAD` has the container bit set but holds raw bytes, so it must stay opaque."""
    for pad in load(path).get_tags(GBL4Pad, allow_missing=True):
        assert pad.data == pad.serialize_payload()


def test_hash_value_roundtrip() -> None:
    value = HashValue(hash_type=1, value=bytes(range(32)))

    assert HashValue.from_bytes(value.serialize()) == value
    assert len(value.serialize()) == 36


def test_hash_value_rejects_truncation() -> None:
    with pytest.raises(ParseError, match="Hash value is truncated"):
        HashValue.from_bytes(b"\x01\x00")


def test_rejects_non_v4_data() -> None:
    with pytest.raises(ParseError, match="not a GBLv4 image"):
        GBL4Image.from_bytes(b"\xeb\x17\xa6\x03" + bytes(16))


def test_rejects_truncated_root() -> None:
    with pytest.raises(ParseError, match="root tag is truncated"):
        GBL4Image.from_bytes(GBL4_MAGIC + (999).to_bytes(4, "little") + b"\x00")


def test_rejects_truncated_child() -> None:
    child = int(GBL4TagId.MANIFEST).to_bytes(4, "little") + (99).to_bytes(4, "little")
    root = GBL4_MAGIC + len(child).to_bytes(4, "little") + child

    with pytest.raises(ParseError, match="is truncated"):
        GBL4Image.from_bytes(root)
