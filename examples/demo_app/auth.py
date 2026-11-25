"""Demo app for the PQC scanner: deliberately uses quantum-vulnerable crypto. Do not copy."""
import hashlib

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa


def make_server_key():
    # 1024-bit RSA: weak even before quantum computers
    return rsa.generate_private_key(public_exponent=65537, key_size=1024)


def encrypt_session_key(public_key, session_key: bytes) -> bytes:
    return public_key.encrypt(
        session_key,
        padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
    )


def ecdh_shared_secret(peer_public_key):
    private = ec.generate_private_key(ec.SECP256R1())
    return private.exchange(ec.ECDH(), peer_public_key)


def password_fingerprint(pw: str) -> str:
    return hashlib.md5(pw.encode()).hexdigest()
