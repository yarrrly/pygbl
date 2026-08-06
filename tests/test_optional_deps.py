"""Signing, encryption and LZ4 are optional. Everything else works without them."""

from __future__ import annotations

from unittest.mock import patch

from cryptography.hazmat.primitives.asymmetric import ec
import pytest

from pygbl import (
    GBL3Compression,
    GBL3EraseProg,
    GBL3Header,
    GBL3Image,
    GBL3Metadata,
    GBL3Type,
    MissingDependencyError,
)

PAYLOAD = bytes((i * 3) % 211 for i in range(4096))
CRYPTO_EXTRA = r"pygbl\[crypto\]"
LZ4_EXTRA = r"pygbl\[lz4\]"


def build_image() -> GBL3Image:
    return GBL3Image(
        tags=[
            GBL3Header(version=3, type=GBL3Type.NONE),
            GBL3EraseProg(address=0x8006000, data=PAYLOAD),
            GBL3Metadata(metadata=b"{}"),
        ]
    ).regenerate_crc()


def test_encrypt_needs_cryptography() -> None:
    with (
        patch("pygbl.crypto.HAVE_CRYPTOGRAPHY", False),
        pytest.raises(MissingDependencyError, match=CRYPTO_EXTRA),
    ):
        build_image().encrypt(bytes(16))


def test_decrypt_needs_cryptography() -> None:
    encrypted = build_image().encrypt(bytes(16))

    with (
        patch("pygbl.crypto.HAVE_CRYPTOGRAPHY", False),
        pytest.raises(MissingDependencyError, match=CRYPTO_EXTRA),
    ):
        encrypted.decrypt(bytes(16))


def test_sign_needs_cryptography() -> None:
    private_key = ec.generate_private_key(ec.SECP256R1())

    with (
        patch("pygbl.crypto.HAVE_CRYPTOGRAPHY", False),
        pytest.raises(MissingDependencyError, match=CRYPTO_EXTRA),
    ):
        build_image().sign(private_key)


def test_verify_needs_cryptography() -> None:
    private_key = ec.generate_private_key(ec.SECP256R1())
    signed = build_image().sign(private_key)

    with (
        patch("pygbl.crypto.HAVE_CRYPTOGRAPHY", False),
        pytest.raises(MissingDependencyError, match=CRYPTO_EXTRA),
    ):
        signed.verify_signature(private_key.public_key())


def test_lz4_compress_needs_lz4() -> None:
    with (
        patch("pygbl.compression.HAVE_LZ4", False),
        pytest.raises(MissingDependencyError, match=LZ4_EXTRA),
    ):
        build_image().compress(GBL3Compression.LZ4)


def test_lz4_decompress_needs_lz4() -> None:
    compressed = build_image().compress(GBL3Compression.LZ4)

    with (
        patch("pygbl.compression.HAVE_LZ4", False),
        pytest.raises(MissingDependencyError, match=LZ4_EXTRA),
    ):
        compressed.decompress()


def test_parsing_needs_nothing() -> None:
    image = build_image()
    data = image.serialize()

    with (
        patch("pygbl.crypto.HAVE_CRYPTOGRAPHY", False),
        patch("pygbl.compression.HAVE_LZ4", False),
    ):
        assert GBL3Image.from_bytes(data) == image
        assert GBL3Image.from_bytes(data).serialize() == data
        assert image.regenerate_crc() == image
        assert image.get_metadata() == b"{}"


def test_lzma_needs_nothing() -> None:
    """LZMA is in the standard library, so it is always available."""
    image = build_image()

    with (
        patch("pygbl.crypto.HAVE_CRYPTOGRAPHY", False),
        patch("pygbl.compression.HAVE_LZ4", False),
    ):
        assert image.compress(GBL3Compression.LZMA).decompress().serialize() == (
            image.serialize()
        )
