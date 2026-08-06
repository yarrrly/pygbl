"""Building GBLs from linked ELFs, checked against `commander gbl create` output.

The fixtures are a local corpus and are not committed. Each `<name>.out` is an ELF
straight out of an SDK build, paired with `<name>_commander.gbl` — the image
`commander` produced from it — and the `<name>_metadata.json` it was handed.
"""

from __future__ import annotations

import io
import pathlib
import zlib

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from elftools.common.exceptions import ELFError
import pytest

from pygbl import (
    GBLApplicationInfo,
    GBLApplicationType,
    GBLBootloader,
    GBLCompression,
    GBLEncryptionInitAesCcm,
    GBLEraseProg,
    GBLHeader,
    GBLImage,
    GBLMetadata,
    GBLType,
    ParseError,
    build_application_gbl,
    build_bootloader_gbl,
    parse_firmware_image,
    read_application_info,
    read_application_properties,
    read_bootloader_version,
    read_loadable_segments,
)
from pygbl.elf import APP_PROPERTIES_MAGIC

FILES = pathlib.Path(__file__).parent / "files"

# Representatives for the format-level assertions
APPLICATION = "yellow_openthread_rcp"
BOOTLOADER = "zbt2_bootloader"

ELVES = sorted(p.stem for p in FILES.glob("*.out"))

if not ELVES:
    pytest.skip("no ELF fixtures present", allow_module_level=True)


def elf_path(name: str) -> pathlib.Path:
    return FILES / f"{name}.out"


def commander_gbl(name: str) -> GBLImage:
    image = parse_firmware_image((FILES / f"{name}_commander.gbl").read_bytes())
    assert isinstance(image, GBLImage)

    return image


def commander_bytes(name: str) -> bytes:
    return (FILES / f"{name}_commander.gbl").read_bytes()


def metadata_for(name: str) -> bytes | None:
    path = FILES / f"{name}_metadata.json"

    return path.read_bytes() if path.exists() else None


def rebuild(name: str) -> GBLImage:
    """Rebuild an image from its ELF the way `commander` would have."""
    theirs = commander_gbl(name)
    builder = (
        build_bootloader_gbl if theirs.has_tag(GBLBootloader) else build_application_gbl
    )

    with elf_path(name).open("rb") as f:
        return builder(f, metadata=metadata_for(name))


def is_encrypted(name: str) -> bool:
    return (
        GBLType.ENCRYPTION_AESCCM in commander_gbl(name).get_first_tag(GBLHeader).type
    )


PLAIN = [name for name in ELVES if not is_encrypted(name)]
ENCRYPTED = [name for name in ELVES if is_encrypted(name)]

AES_KEY = bytes.fromhex(
    (FILES / "zwa2_vendor_encrypt.key").read_text().split(":")[1].strip()
)
_KEY = load_pem_private_key(
    (FILES / "zwa2_vendor_sign.key").read_bytes(), password=None
)
assert isinstance(_KEY, ec.EllipticCurvePrivateKey)
PRIVATE_KEY = _KEY


@pytest.mark.parametrize("name", PLAIN)
def test_rebuild_from_elf_is_byte_exact(name: str) -> None:
    """The whole point: reproduce `commander gbl create` exactly, from the ELF alone."""
    assert rebuild(name).serialize(block_size=4) == commander_bytes(name)


@pytest.mark.parametrize("name", PLAIN)
def test_metadata_json_matches_the_embedded_tag(name: str) -> None:
    metadata = metadata_for(name)

    if metadata is None:
        pytest.skip(f"{name} has no metadata json")

    assert commander_gbl(name).get_metadata() == metadata


@pytest.mark.parametrize("name", ENCRYPTED)
def test_commander_signature_verifies(name: str) -> None:
    assert commander_gbl(name).verify_signature(PRIVATE_KEY.public_key())


@pytest.mark.parametrize("name", ENCRYPTED)
def test_encrypted_image_matches_the_elf(name: str) -> None:
    """Peel off encryption and compression, then compare against the ELF."""
    plain = commander_gbl(name).decrypt(AES_KEY).decompress()

    with elf_path(name).open("rb") as f:
        ours = build_application_gbl(f, metadata=plain.get_metadata())

    assert plain.get_tags(GBLEraseProg) == ours.get_tags(GBLEraseProg)
    assert plain.get_first_tag(GBLApplicationInfo) == ours.get_first_tag(
        GBLApplicationInfo
    )


@pytest.mark.parametrize("name", ENCRYPTED)
def test_we_can_seal_an_image_the_same_way(name: str) -> None:
    """Compressing, encrypting and signing our own build yields the same plaintext.

    Byte equality is impossible here: ECDSA picks a random `k` and liblzma does not
    match commander's encoder, so the ciphertext and signature legitimately differ.
    """
    theirs = commander_gbl(name)
    nonce = theirs.get_first_tag(GBLEncryptionInitAesCcm).nonce
    expected = theirs.decrypt(AES_KEY).decompress()

    with elf_path(name).open("rb") as f:
        ours = build_application_gbl(f, metadata=expected.get_metadata())

    sealed = (
        ours.compress(GBLCompression.LZMA)
        .encrypt(AES_KEY, nonce=nonce)
        .sign(PRIVATE_KEY)
    )

    assert sealed.verify_signature(PRIVATE_KEY.public_key())
    assert GBLImage.from_bytes(sealed.serialize()) == sealed

    # Compare the tag region: `decrypt` carries the outer file's padding through, and
    # commander's image is padded to a 4 byte boundary while ours is freshly built
    unsealed = sealed.decrypt(AES_KEY).decompress()

    assert unsealed.serialize_tags() == expected.serialize_tags()
    assert unsealed.tags == expected.tags


def test_loadable_segments_are_merged_and_sorted() -> None:
    with elf_path(APPLICATION).open("rb") as f:
        segments = read_loadable_segments(f)

    assert segments == sorted(segments, key=lambda s: s.address)

    # Merging means no two remaining segments are contiguous
    for previous, following in zip(segments, segments[1:]):
        assert previous.address + len(previous.data) < following.address


def test_application_properties_magic() -> None:
    with elf_path(APPLICATION).open("rb") as f:
        assert read_application_properties(f).startswith(APP_PROPERTIES_MAGIC)


def test_application_info_is_read_from_the_elf() -> None:
    with elf_path(APPLICATION).open("rb") as f:
        app_info = read_application_info(f)

    assert app_info == commander_gbl(APPLICATION).get_first_tag(GBLApplicationInfo)
    assert GBLApplicationType.THREAD in app_info.type


def test_application_gbl_without_metadata() -> None:
    with elf_path(APPLICATION).open("rb") as f:
        image = build_application_gbl(f)

    assert image.find_first_tag(GBLMetadata) is None
    assert image.get_first_tag(GBLHeader).type == GBLType.NONE
    image.validate()


def test_program_data_matches_the_segments() -> None:
    with elf_path(APPLICATION).open("rb") as f:
        segments = read_loadable_segments(f)

    with elf_path(APPLICATION).open("rb") as f:
        image = build_application_gbl(f)

    assert [(t.address, t.data) for t in image.get_tags(GBLEraseProg)] == [
        (s.address, s.data) for s in segments
    ]


def test_bootloader_version_is_read_from_the_elf() -> None:
    with elf_path(BOOTLOADER).open("rb") as f:
        version = read_bootloader_version(f)

    assert version == commander_gbl(BOOTLOADER).get_first_tag(GBLBootloader).version


def test_bootloader_payload_carries_its_own_crc() -> None:
    """`commander` appends a CRC32 of the payload, so the tag is 4 bytes longer."""
    with elf_path(BOOTLOADER).open("rb") as f:
        segments = read_loadable_segments(f)

    with elf_path(BOOTLOADER).open("rb") as f:
        bootloader = build_bootloader_gbl(f).get_first_tag(GBLBootloader)

    payload = segments[0].data

    assert len(bootloader.data) == len(payload) + 4
    assert bootloader.data[:-4] == payload
    assert bootloader.data[-4:] == (zlib.crc32(payload) & 0xFFFFFFFF).to_bytes(
        4, "little"
    )
    assert bootloader.address == segments[0].address


def test_bootloader_builder_rejects_split_images() -> None:
    """An application ELF has several separate runs, so it is not a bootloader."""
    with elf_path(APPLICATION).open("rb") as f:
        with pytest.raises(ParseError, match="one contiguous bootloader segment"):
            build_bootloader_gbl(f)


def test_rejects_non_elf_input() -> None:
    with pytest.raises(ELFError):
        build_application_gbl(io.BytesIO(b"not an ELF at all"))
