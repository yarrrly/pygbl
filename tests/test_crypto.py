from __future__ import annotations

import dataclasses
import hashlib

from cryptography.hazmat.primitives.asymmetric import ec
import pytest

from pygbl import (
    GBLApplicationInfo,
    GBLApplicationType,
    GBLEncryptedData,
    GBLEncryptionInitAesCcm,
    GBLEnd,
    GBLHeader,
    GBLImage,
    GBLMetadata,
    GBLProgramData2,
    GBLSignature,
    GBLType,
)
from pygbl.crypto import (
    AES_KEY_SIZE,
    CCM_FLAGS,
    CCM_INITIAL_COUNTER,
    aes_ctr_crypt,
    counter_block,
    generate_nonce,
)
from pygbl.gbl import serialize_tag

KEY = bytes(range(AES_KEY_SIZE))
NONCE = bytes(range(0x10, 0x1C))
PRODUCT_ID = bytes(range(16))


@pytest.fixture
def private_key() -> ec.EllipticCurvePrivateKey:
    return ec.generate_private_key(ec.SECP256R1())


@pytest.fixture
def image() -> GBLImage:
    return GBLImage(
        tags=[
            GBLHeader(version=3, type=GBLType.NONE),
            GBLApplicationInfo(
                type=GBLApplicationType.ZWAVE,
                version=1,
                capabilities=0,
                product_id=PRODUCT_ID,
            ),
            GBLProgramData2(address=0x8006000, data=bytes(range(256)) * 4),
            GBLMetadata(metadata=b'{"fw_type": "zwave_ncp"}'),
        ]
    ).regenerate_crc()


def test_counter_block_layout():
    """flags (1) || nonce (12) || counter (3, big-endian), counter starting at 1."""
    assert counter_block(NONCE) == bytes([CCM_FLAGS]) + NONCE + b"\x00\x00\x01"
    assert (
        counter_block(NONCE, 0x123456) == bytes([CCM_FLAGS]) + NONCE + b"\x12\x34\x56"
    )
    assert CCM_FLAGS == 0x02
    assert CCM_INITIAL_COUNTER == 1


def test_counter_block_rejects_bad_nonce():
    with pytest.raises(ValueError, match="Nonce must be 12 bytes"):
        counter_block(b"\x00" * 8)


def test_aes_ctr_known_answer():
    """Vector lifted from a `commander --encrypt` image: nonce, key and first tag."""
    key = bytes.fromhex("7f8fe53979b31bc556fcb131aff42414")
    nonce = bytes.fromhex("9f939baad06c1ccfdc8d3347")
    plaintext = bytes.fromhex("f40a0af41c000000")  # the app info tag header

    assert aes_ctr_crypt(key, nonce, plaintext) == bytes.fromhex("db4a1006bd2bc839")


def test_aes_ctr_is_symmetric():
    data = bytes(range(256))

    assert aes_ctr_crypt(KEY, NONCE, aes_ctr_crypt(KEY, NONCE, data)) == data


def test_aes_ctr_rejects_bad_key():
    with pytest.raises(ValueError, match="Key must be 16 bytes"):
        aes_ctr_crypt(b"\x00" * 8, NONCE, b"data")


def test_generate_nonce():
    assert len(generate_nonce()) == 12
    assert generate_nonce() != generate_nonce()


def test_sign_and_verify(image, private_key):
    signed = image.sign(private_key)

    assert signed.verify_signature(private_key.public_key())
    assert len(signed.get_first_tag(GBLSignature).r) == 32
    assert len(signed.get_first_tag(GBLSignature).s) == 32


def test_signing_sets_header_flag(image, private_key):
    signed = image.sign(private_key)

    assert GBLType.SIGNATURE_ECDSA_P256 in signed.get_first_tag(GBLHeader).type


def test_signed_image_is_valid_and_reparses(image, private_key):
    signed = image.sign(private_key)
    data = signed.serialize()

    assert GBLImage.from_bytes(data) == signed
    assert GBLImage.from_bytes(data).verify_signature(private_key.public_key())


def test_signing_digest_excludes_signature_and_end(image, private_key):
    """The bootloader hashes every tag except the signature and end tags."""
    signed = image.sign(private_key)
    expected = hashlib.sha256(
        b"".join(
            serialize_tag(t)
            for t in signed.tags
            if not isinstance(t, (GBLSignature, GBLEnd))
        )
    ).digest()

    assert signed.signing_digest() == expected


def test_verify_rejects_corrupted_signature(image, private_key):
    signed = image.sign(private_key)
    signature = signed.get_first_tag(GBLSignature)
    corrupted = bytearray(signature.r)
    corrupted[5] ^= 0xFF

    tags = list(signed.tags)
    tags[tags.index(signature)] = dataclasses.replace(signature, r=bytes(corrupted))
    tampered = dataclasses.replace(signed, tags=tags).regenerate_crc()

    # The CRC is still valid, only the signature is wrong
    tampered.validate()
    assert not tampered.verify_signature(private_key.public_key())


def test_verify_rejects_tampered_payload(image, private_key):
    signed = image.sign(private_key)
    program = signed.get_first_tag(GBLProgramData2)
    data = bytearray(program.data)
    data[0] ^= 0xFF

    tags = list(signed.tags)
    tags[tags.index(program)] = dataclasses.replace(program, data=bytes(data))
    tampered = dataclasses.replace(signed, tags=tags).regenerate_crc()

    tampered.validate()
    assert not tampered.verify_signature(private_key.public_key())


def test_verify_rejects_wrong_key(image, private_key):
    signed = image.sign(private_key)
    other_key = ec.generate_private_key(ec.SECP256R1())

    assert not signed.verify_signature(other_key.public_key())


def test_verify_unsigned_image_raises(image, private_key):
    with pytest.raises(KeyError, match="not signed"):
        image.verify_signature(private_key.public_key())


def test_signing_is_stable_across_reparse(image, private_key):
    """Re-signing an already signed image replaces the signature rather than stacking."""
    signed = image.sign(private_key).sign(private_key)

    assert len(signed.get_tags(GBLSignature)) == 1
    assert signed.verify_signature(private_key.public_key())


def test_encrypt_and_decrypt(image):
    encrypted = image.encrypt(KEY, nonce=NONCE)

    assert encrypted.decrypt(KEY).serialize() == image.serialize()


def test_encrypt_sets_header_flag_and_init_tag(image):
    encrypted = image.encrypt(KEY, nonce=NONCE)
    init = encrypted.get_first_tag(GBLEncryptionInitAesCcm)

    assert GBLType.ENCRYPTION_AESCCM in encrypted.get_first_tag(GBLHeader).type
    assert init.nonce == NONCE
    assert init.msg_len == sum(
        len(t.ciphertext) for t in encrypted.get_tags(GBLEncryptedData)
    )


def test_encrypt_wraps_each_tag_separately(image):
    encrypted = image.encrypt(KEY, nonce=NONCE)
    plaintext_tags = [
        t for t in image.tags if not isinstance(t, (GBLHeader, GBLEnd, GBLSignature))
    ]
    encrypted_tags = encrypted.get_tags(GBLEncryptedData)

    assert len(encrypted_tags) == len(plaintext_tags)
    assert [len(t.ciphertext) for t in encrypted_tags] == [
        len(serialize_tag(t)) for t in plaintext_tags
    ]


def test_encrypted_keystream_is_continuous(image):
    """One continuous keystream spans all encrypted tags, so they cannot be reordered."""
    encrypted = image.encrypt(KEY, nonce=NONCE)
    ciphertext = b"".join(t.ciphertext for t in encrypted.get_tags(GBLEncryptedData))
    plaintext = b"".join(
        serialize_tag(t)
        for t in image.tags
        if not isinstance(t, (GBLHeader, GBLEnd, GBLSignature))
    )

    assert aes_ctr_crypt(KEY, NONCE, plaintext) == ciphertext


def test_encrypt_uses_a_random_nonce_by_default(image):
    first = image.encrypt(KEY).get_first_tag(GBLEncryptionInitAesCcm)
    second = image.encrypt(KEY).get_first_tag(GBLEncryptionInitAesCcm)

    assert first.nonce != second.nonce


def test_double_encryption_rejected(image):
    with pytest.raises(ValueError, match="already encrypted"):
        image.encrypt(KEY, nonce=NONCE).encrypt(KEY, nonce=NONCE)


def test_decrypt_unencrypted_image_raises(image):
    with pytest.raises(KeyError, match="not encrypted"):
        image.decrypt(KEY)


def test_encrypt_then_sign(image, private_key):
    """`commander` signs the ciphertext, so the signature survives on the sealed image."""
    sealed = image.encrypt(KEY, nonce=NONCE).sign(private_key)

    assert sealed.verify_signature(private_key.public_key())
    assert sealed.get_first_tag(GBLHeader).type == (
        GBLType.ENCRYPTION_AESCCM | GBLType.SIGNATURE_ECDSA_P256
    )
    assert GBLImage.from_bytes(sealed.serialize()) == sealed


def test_decrypt_drops_signature_and_flags(image, private_key):
    sealed = image.encrypt(KEY, nonce=NONCE).sign(private_key)
    decrypted = sealed.decrypt(KEY)

    assert decrypted.find_first_tag(GBLSignature) is None
    assert decrypted.get_first_tag(GBLHeader).type == GBLType.NONE
    assert decrypted.serialize() == image.serialize()
