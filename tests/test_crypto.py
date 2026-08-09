from __future__ import annotations

import dataclasses
import hashlib

from cryptography.hazmat.primitives.asymmetric import ec
import pytest

from pygbl import (
    GBL3ApplicationInfo,
    GBL3ApplicationType,
    GBL3EncryptedData,
    GBL3EncryptionInitAesCcm,
    GBL3End,
    GBL3EraseProg,
    GBL3Header,
    GBL3Image,
    GBL3Metadata,
    GBL3Signature,
    GBL3Type,
)
from pygbl.crypto import (
    AES_KEY_SIZE,
    CCM_FLAGS,
    CCM_INITIAL_COUNTER,
    aes_ctr_crypt,
    counter_block,
    generate_nonce,
)
from pygbl.gbl3 import serialize_tag

KEY = bytes(range(AES_KEY_SIZE))
NONCE = bytes(range(0x10, 0x1C))
PRODUCT_ID = bytes(range(16))
PRIVATE_KEY = ec.generate_private_key(ec.SECP256R1())


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
            GBL3EraseProg(address=0x8006000, data=bytes(range(256)) * 4),
            GBL3Metadata(metadata=b'{"fw_type": "zwave_ncp"}'),
        ]
    ).regenerate_crc()


def test_counter_block_layout() -> None:
    """flags (1) || nonce (12) || counter (3, big-endian), counter starting at 1."""
    assert counter_block(NONCE) == bytes([CCM_FLAGS]) + NONCE + b"\x00\x00\x01"
    assert (
        counter_block(NONCE, 0x123456) == bytes([CCM_FLAGS]) + NONCE + b"\x12\x34\x56"
    )
    assert CCM_FLAGS == 0x02
    assert CCM_INITIAL_COUNTER == 1


def test_counter_block_rejects_bad_nonce() -> None:
    with pytest.raises(ValueError, match="Nonce must be 12 bytes"):
        counter_block(b"\x00" * 8)


def test_aes_ctr_known_answer() -> None:
    """Vector lifted from a `commander --encrypt` image: nonce, key and first tag."""
    key = bytes.fromhex("7f8fe53979b31bc556fcb131aff42414")
    nonce = bytes.fromhex("9f939baad06c1ccfdc8d3347")
    plaintext = bytes.fromhex("f40a0af41c000000")  # the app info tag header

    assert aes_ctr_crypt(key, nonce, plaintext) == bytes.fromhex("db4a1006bd2bc839")


def test_aes_ctr_is_symmetric() -> None:
    data = bytes(range(256))

    assert aes_ctr_crypt(KEY, NONCE, aes_ctr_crypt(KEY, NONCE, data)) == data


def test_aes_ctr_rejects_bad_key() -> None:
    with pytest.raises(ValueError, match="Key must be 16 bytes"):
        aes_ctr_crypt(b"\x00" * 8, NONCE, b"data")


def test_generate_nonce() -> None:
    assert len(generate_nonce()) == 12
    assert generate_nonce() != generate_nonce()


def test_sign_and_verify() -> None:
    signed = build_image().sign(PRIVATE_KEY)

    assert signed.verify_signature(PRIVATE_KEY.public_key())
    assert len(signed.get_first_tag(GBL3Signature).r) == 32
    assert len(signed.get_first_tag(GBL3Signature).s) == 32


def test_signing_sets_header_flag() -> None:
    signed = build_image().sign(PRIVATE_KEY)

    assert GBL3Type.SIGNATURE_ECDSA_P256 in signed.get_first_tag(GBL3Header).type


def test_signed_image_is_valid_and_reparses() -> None:
    signed = build_image().sign(PRIVATE_KEY)
    data = signed.serialize()

    assert GBL3Image.from_bytes(data) == signed
    assert GBL3Image.from_bytes(data).verify_signature(PRIVATE_KEY.public_key())


def test_signing_digest_excludes_signature_and_end() -> None:
    """The bootloader hashes every tag except the signature and end tags."""
    signed = build_image().sign(PRIVATE_KEY)
    expected = hashlib.sha256(
        b"".join(
            serialize_tag(t)
            for t in signed.tags
            if not isinstance(t, (GBL3Signature, GBL3End))
        )
    ).digest()

    assert signed.signing_digest() == expected


def test_verify_rejects_corrupted_signature() -> None:
    signed = build_image().sign(PRIVATE_KEY)
    signature = signed.get_first_tag(GBL3Signature)
    corrupted = bytearray(signature.r)
    corrupted[5] ^= 0xFF

    tags = list(signed.tags)
    tags[tags.index(signature)] = dataclasses.replace(signature, r=bytes(corrupted))
    tampered = dataclasses.replace(signed, tags=tags).regenerate_crc()

    # The CRC is still valid, only the signature is wrong
    tampered.validate()
    assert not tampered.verify_signature(PRIVATE_KEY.public_key())


def test_verify_rejects_tampered_payload() -> None:
    signed = build_image().sign(PRIVATE_KEY)
    program = signed.get_first_tag(GBL3EraseProg)
    data = bytearray(program.data)
    data[0] ^= 0xFF

    tags = list(signed.tags)
    tags[tags.index(program)] = dataclasses.replace(program, data=bytes(data))
    tampered = dataclasses.replace(signed, tags=tags).regenerate_crc()

    tampered.validate()
    assert not tampered.verify_signature(PRIVATE_KEY.public_key())


def test_verify_rejects_wrong_key() -> None:
    signed = build_image().sign(PRIVATE_KEY)
    other_key = ec.generate_private_key(ec.SECP256R1())

    assert not signed.verify_signature(other_key.public_key())


def test_verify_unsigned_image_raises() -> None:
    with pytest.raises(KeyError, match="not signed"):
        build_image().verify_signature(PRIVATE_KEY.public_key())


def test_signing_is_stable_across_reparse() -> None:
    """Re-signing an already signed image replaces the signature rather than stacking."""
    signed = build_image().sign(PRIVATE_KEY).sign(PRIVATE_KEY)

    assert len(signed.get_tags(GBL3Signature)) == 1
    assert signed.verify_signature(PRIVATE_KEY.public_key())


def test_encrypt_and_decrypt() -> None:
    image = build_image()
    encrypted = image.encrypt(KEY, nonce=NONCE)

    assert encrypted.decrypt(KEY).serialize() == image.serialize()


def test_encrypt_sets_header_flag_and_init_tag() -> None:
    encrypted = build_image().encrypt(KEY, nonce=NONCE)
    init = encrypted.get_first_tag(GBL3EncryptionInitAesCcm)

    assert GBL3Type.ENCRYPTION_AESCCM in encrypted.get_first_tag(GBL3Header).type
    assert init.nonce == NONCE
    assert init.msg_len == sum(
        len(t.ciphertext) for t in encrypted.get_tags(GBL3EncryptedData)
    )


def test_encrypt_wraps_each_tag_separately() -> None:
    image = build_image()
    encrypted = image.encrypt(KEY, nonce=NONCE)
    plaintext_tags = [
        t for t in image.tags if not isinstance(t, (GBL3Header, GBL3End, GBL3Signature))
    ]
    encrypted_tags = encrypted.get_tags(GBL3EncryptedData)

    assert len(encrypted_tags) == len(plaintext_tags)
    assert [len(t.ciphertext) for t in encrypted_tags] == [
        len(serialize_tag(t)) for t in plaintext_tags
    ]


def test_encrypted_keystream_is_continuous() -> None:
    """One continuous keystream spans all encrypted tags, so they cannot be reordered."""
    image = build_image()
    encrypted = image.encrypt(KEY, nonce=NONCE)
    ciphertext = b"".join(t.ciphertext for t in encrypted.get_tags(GBL3EncryptedData))
    plaintext = b"".join(
        serialize_tag(t)
        for t in image.tags
        if not isinstance(t, (GBL3Header, GBL3End, GBL3Signature))
    )

    assert aes_ctr_crypt(KEY, NONCE, plaintext) == ciphertext


def test_encrypt_uses_a_random_nonce_by_default() -> None:
    image = build_image()
    first = image.encrypt(KEY).get_first_tag(GBL3EncryptionInitAesCcm)
    second = image.encrypt(KEY).get_first_tag(GBL3EncryptionInitAesCcm)

    assert first.nonce != second.nonce


def test_double_encryption_rejected() -> None:
    with pytest.raises(ValueError, match="already encrypted"):
        build_image().encrypt(KEY, nonce=NONCE).encrypt(KEY, nonce=NONCE)


def test_decrypt_unencrypted_image_raises() -> None:
    with pytest.raises(KeyError, match="not encrypted"):
        build_image().decrypt(KEY)


def test_encrypt_then_sign() -> None:
    """`commander` signs the ciphertext, so the signature survives on the sealed image."""
    sealed = build_image().encrypt(KEY, nonce=NONCE).sign(PRIVATE_KEY)

    assert sealed.verify_signature(PRIVATE_KEY.public_key())
    assert sealed.get_first_tag(GBL3Header).type == (
        GBL3Type.ENCRYPTION_AESCCM | GBL3Type.SIGNATURE_ECDSA_P256
    )
    assert GBL3Image.from_bytes(sealed.serialize()) == sealed


def test_decrypt_drops_signature_and_flags() -> None:
    image = build_image()
    sealed = image.encrypt(KEY, nonce=NONCE).sign(PRIVATE_KEY)
    decrypted = sealed.decrypt(KEY)

    assert decrypted.get_tags(GBL3Signature, allow_missing=True) == []
    assert decrypted.get_first_tag(GBL3Header).type == GBL3Type.NONE
    assert decrypted.serialize() == image.serialize()
