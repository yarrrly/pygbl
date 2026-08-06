from __future__ import annotations

import pathlib

import pytest

from pygbl import ParseError, read_encryption_key

FILES = pathlib.Path(__file__).parent / "files"
KEY_FILE = FILES / "zwa2_vendor_encrypt.key"


def test_reads_a_real_token_file() -> None:
    """The ZWA-2 vendor key, exactly as Simplicity Studio writes it."""
    if not KEY_FILE.exists():
        pytest.skip("no key fixture present")

    key = read_encryption_key(KEY_FILE.read_text())

    assert len(key) == 16
    assert key == bytes.fromhex("7f8fe53979b31bc556fcb131aff42414")


def test_accepts_bytes() -> None:
    if not KEY_FILE.exists():
        pytest.skip("no key fixture present")

    assert read_encryption_key(KEY_FILE.read_bytes()) == read_encryption_key(
        KEY_FILE.read_text()
    )


def test_accepts_lowercase_and_surrounding_whitespace() -> None:
    text = "  TOKEN_MFG_SECURE_BOOTLOADER_KEY:   7f8fe53979b31bc556fcb131aff42414  \n"

    assert read_encryption_key(text) == bytes.fromhex(
        "7f8fe53979b31bc556fcb131aff42414"
    )


def test_rejects_a_file_without_the_token() -> None:
    with pytest.raises(ParseError, match="No TOKEN_MFG_SECURE_BOOTLOADER_KEY"):
        read_encryption_key("# nothing useful here\n")


def test_rejects_a_key_of_the_wrong_length() -> None:
    with pytest.raises(ParseError, match="must be 16 bytes"):
        read_encryption_key("TOKEN_MFG_SECURE_BOOTLOADER_KEY: 0011223344556677\n")
