from __future__ import annotations

import os

from pygbl.types import MissingDependencyError

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, utils
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
except ImportError:
    HAVE_CRYPTOGRAPHY = False
else:
    HAVE_CRYPTOGRAPHY = True

# The bootloader runs AES-CTR with a CCM-style counter block:
#   flags (1 byte) || nonce (12 bytes) || counter (3 bytes, big-endian)
# and starts the counter at 1. See `btl_initAesCcm` and its caller in the GBL parser.
CCM_FLAGS = 0x02
CCM_INITIAL_COUNTER = 1

NONCE_SIZE = 12
AES_KEY_SIZE = 16
SIGNATURE_COMPONENT_SIZE = 32


def require_cryptography() -> None:
    if not HAVE_CRYPTOGRAPHY:
        raise MissingDependencyError(
            "Signing and encryption need the `cryptography` package."
            " Install it with `pip install pygbl[crypto]`."
        )


def generate_nonce() -> bytes:
    return os.urandom(NONCE_SIZE)


def counter_block(nonce: bytes, counter: int = CCM_INITIAL_COUNTER) -> bytes:
    if len(nonce) != NONCE_SIZE:
        raise ValueError(f"Nonce must be {NONCE_SIZE} bytes, got {len(nonce)}")

    return bytes([CCM_FLAGS]) + nonce + counter.to_bytes(3, "big")


def aes_ctr_crypt(
    key: bytes, nonce: bytes, data: bytes, *, counter: int = CCM_INITIAL_COUNTER
) -> bytes:
    """Encrypt or decrypt image data. AES-CTR is symmetric, so this does both."""
    require_cryptography()

    if len(key) != AES_KEY_SIZE:
        raise ValueError(f"Key must be {AES_KEY_SIZE} bytes, got {len(key)}")

    cipher = Cipher(algorithms.AES(key), modes.CTR(counter_block(nonce, counter)))
    encryptor = cipher.encryptor()

    return encryptor.update(data) + encryptor.finalize()


def load_public_key(key: bytes) -> ec.EllipticCurvePublicKey:
    """Load a public key stored as the raw `x || y` pair the certificates carry."""
    require_cryptography()

    if len(key) != 2 * SIGNATURE_COMPONENT_SIZE:
        raise ValueError(
            f"Key must be {2 * SIGNATURE_COMPONENT_SIZE} bytes, got {len(key)}"
        )

    return ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), b"\x04" + key)


def sign_digest(
    private_key: ec.EllipticCurvePrivateKey, digest: bytes
) -> tuple[bytes, bytes]:
    """Sign a SHA-256 digest, returning the raw `r` and `s` components."""
    require_cryptography()

    signature = private_key.sign(digest, ec.ECDSA(utils.Prehashed(hashes.SHA256())))
    r, s = utils.decode_dss_signature(signature)

    return (
        r.to_bytes(SIGNATURE_COMPONENT_SIZE, "big"),
        s.to_bytes(SIGNATURE_COMPONENT_SIZE, "big"),
    )


def verify_digest(
    public_key: ec.EllipticCurvePublicKey, digest: bytes, r: bytes, s: bytes
) -> bool:
    require_cryptography()

    signature = utils.encode_dss_signature(
        int.from_bytes(r, "big"), int.from_bytes(s, "big")
    )

    try:
        public_key.verify(signature, digest, ec.ECDSA(utils.Prehashed(hashes.SHA256())))
    except InvalidSignature:
        return False

    return True
