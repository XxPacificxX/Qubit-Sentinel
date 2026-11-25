"""Detection rules for cryptography that a large quantum computer could break (and for PQC already in use).

Each rule matches one line. ``family`` is either a fixed algorithm family or ``None``,
in which case capture group 1 is normalized into a family (e.g. ``"EC"`` from Java's
``KeyPairGenerator.getInstance("EC")``). ``usage`` may be ``None`` to use the family default.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

PY = {".py"}
JS = {".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"}
JVM = {".java", ".kt", ".kts", ".scala"}
GO = {".go"}
CS = {".cs"}
ANY: set[str] = set()  # empty = every scanned file

# Shor's algorithm breaks these (factoring / discrete log)
QUANTUM_VULNERABLE = {"RSA", "EC", "ECDSA", "ECDH", "DH", "DSA", "Ed25519", "Ed448", "EdDSA",
                      "X25519", "X448", "TLS-classical", "SSH-key", "Certificate", "PrivateKey"}
CLASSICALLY_WEAK_HASH = {"MD5", "SHA-1"}
PQC = {"ML-KEM", "ML-DSA", "SLH-DSA", "PQC"}

FAMILY_USAGE = {
    "RSA": "public-key", "EC": "public-key", "ECDSA": "signature", "DSA": "signature",
    "Ed25519": "signature", "Ed448": "signature", "EdDSA": "signature",
    "ECDH": "key-exchange", "DH": "key-exchange", "X25519": "key-exchange", "X448": "key-exchange",
}

_NORMALIZE = {
    "rsa": "RSA", "rsa-pss": "RSA", "rsa-oaep": "RSA", "rsassa-pkcs1-v1_5": "RSA", "rs": "RSA", "ps": "RSA",
    "ec": "EC", "ecdsa": "ECDSA", "es": "ECDSA", "ecdh": "ECDH", "dh": "DH", "diffiehellman": "DH",
    "dsa": "DSA", "x25519": "X25519", "xdh": "X25519", "x448": "X448",
    "ed25519": "Ed25519", "ed448": "Ed448", "eddsa": "EdDSA",
}


def normalize_family(raw: str) -> str:
    s = raw.strip().lower()
    if "with" in s:  # Java signature names: SHA256withRSA, SHA384withECDSA
        s = s.split("with", 1)[1]
    return _NORMALIZE.get(s, raw.upper())


@dataclass(frozen=True)
class Rule:
    id: str
    pattern: re.Pattern
    family: str | None
    usage: str | None
    exts: frozenset[str]
    description: str


def _r(id, regex, family, usage, exts, description, flags=0):
    return Rule(id, re.compile(regex, flags), family, usage, frozenset(exts), description)


RULES: list[Rule] = [
    # --- key material and protocol config (any file) ---
    _r("generic.private-key", r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----",
       "PrivateKey", "key-material", ANY, "Private key committed to the repository"),
    _r("generic.certificate", r"-----BEGIN CERTIFICATE-----", "Certificate", "certificate", ANY,
       "X.509 certificate (RSA/ECDSA signed)"),
    _r("generic.ssh-key", r"\b(ssh-rsa|ssh-dss|ecdsa-sha2-nistp\d+|ssh-ed25519) AAAA", "SSH-key", "ssh-key", ANY,
       "SSH public key"),
    _r("generic.tls-kex", r"\b(?:TLS_ECDHE_|TLS_DHE_|TLS_RSA_WITH|ECDHE-(?:RSA|ECDSA)-|DHE-RSA-)", "TLS-classical",
       "protocol", ANY, "TLS cipher suite with classical key exchange"),
    _r("generic.openssl-cli", r"\bopenssl\s+(?:genrsa|ecparam|dhparam|req\b.*-newkey\s+(?:rsa|ec))", None, None, ANY,
       "OpenSSL CLI key generation"),
    _r("generic.jwt-alg", r"""['"]((?:RS|PS|ES)(?:256|384|512))['"]""", None, "signature", ANY,
       "JWT/JOSE signature algorithm"),
    _r("generic.webcrypto", r"""['"](RSA-OAEP|RSASSA-PKCS1-v1_5|RSA-PSS|ECDSA|ECDH)['"]""", None, None, ANY,
       "WebCrypto / JOSE algorithm name"),
    _r("generic.weak-hash", r"""\b(?:hashlib\.(md5|sha1)\b|(MD5|SHA1)\.new\(|createHash\(\s*['"](md5|sha1)['"]|MessageDigest\.getInstance\(\s*"(MD5|SHA-?1)"|\bcrypto/(md5|sha1)\b)""",
       "weak-hash", "hash", ANY, "MD5/SHA-1 (broken classically, not a quantum issue)"),
    _r("generic.aes128", r"(?i)\baes[-_]?128\b", "AES-128", "symmetric", ANY,
       "AES-128 (Grover's algorithm reduces margin; AES-256 preferred for long-term secrecy)"),
    _r("generic.pqc", r"(?i)\b(ml-?kem(?:-?\d+)?|kyber\d*|ml-?dsa(?:-?\d+)?|dilithium\d*|slh-?dsa|sphincs\+?|x25519mlkem768|liboqs|oqs\.(?:KeyEncapsulation|Signature))\b",
       "PQC", "pqc", ANY, "Post-quantum algorithm already in use"),

    # --- Python ---
    _r("py.cryptography.rsa", r"\brsa\.(generate_private_key|RSAPrivateKey|RSAPublicNumbers)\b", "RSA", None, PY,
       "cryptography: RSA"),
    _r("py.cryptography.ecdh", r"\bec\.ECDH\b", "ECDH", "key-exchange", PY, "cryptography: ECDH key exchange"),
    _r("py.cryptography.ecdsa", r"\bec\.ECDSA\b", "ECDSA", "signature", PY, "cryptography: ECDSA signature"),
    _r("py.cryptography.ec", r"\bec\.(?:generate_private_key|derive_private_key|SECP\d+[RK]1)\b", "EC", None, PY,
       "cryptography: elliptic-curve key"),
    _r("py.cryptography.dh", r"\bdh\.(generate_parameters|DHParameterNumbers)\b", "DH", "key-exchange", PY,
       "cryptography: finite-field Diffie-Hellman"),
    _r("py.cryptography.dsa", r"\bdsa\.generate_private_key\b", "DSA", "signature", PY, "cryptography: DSA"),
    _r("py.x25519", r"\bX25519Private(?:Key)?\b|\bx25519\.", "X25519", "key-exchange", PY, "X25519 key exchange"),
    _r("py.ed25519", r"\bEd25519Private(?:Key)?\b|\bed25519\.", "Ed25519", "signature", PY, "Ed25519 signatures"),
    _r("py.pycryptodome.rsa", r"\bRSA\.(generate|import_key|importKey)\(|\bPKCS1_OAEP\b|\bpkcs1_15\b|\bPKCS1_v1_5\b",
       "RSA", None, PY, "PyCryptodome: RSA"),
    _r("py.pycryptodome.ecc", r"\bECC\.(generate|import_key)\(|\bDSS\.new\(", "ECDSA", "signature", PY,
       "PyCryptodome: ECC/DSS"),
    _r("py.python-rsa", r"\brsa\.newkeys\(", "RSA", "public-key", PY, "python-rsa"),
    _r("py.paramiko", r"\bparamiko\.(RSAKey|ECDSAKey|DSSKey)\b", "SSH-key", "ssh-key", PY, "Paramiko SSH key"),

    # --- JavaScript / TypeScript ---
    _r("js.keypair", r"""generateKeyPair(?:Sync)?\(\s*['"](rsa|rsa-pss|ec|dsa|dh|x25519|x448|ed25519|ed448)['"]""",
       None, None, JS, "Node crypto key pair generation"),
    _r("js.ecdh", r"\bcreateECDH\(", "ECDH", "key-exchange", JS, "Node crypto ECDH"),
    _r("js.dh", r"\b(?:createDiffieHellman|getDiffieHellman)\(", "DH", "key-exchange", JS, "Node crypto DH"),
    _r("js.public-encrypt", r"\b(?:publicEncrypt|privateDecrypt)\(", "RSA", "encryption", JS, "Node RSA encryption"),
    _r("js.forge", r"\bforge\.pki\.rsa\b", "RSA", None, JS, "node-forge RSA"),
    _r("js.elliptic", r"""new\s+(?:EC|ec)\(\s*['"](?:secp256k1|p256|p384|curve25519|ed25519)['"]""", "EC", None, JS,
       "elliptic library"),

    # --- Java / Kotlin / Scala ---
    _r("jvm.keypair", r"""KeyPairGenerator\.getInstance\(\s*"(RSA|EC|DSA|DH|DiffieHellman|X25519|X448|XDH|Ed25519|Ed448|EdDSA)\"""",
       None, None, JVM, "JCA key pair generation"),
    _r("jvm.keyagreement", r"""KeyAgreement\.getInstance\(\s*"(ECDH|DH|DiffieHellman|XDH|X25519|X448)\"""",
       None, "key-exchange", JVM, "JCA key agreement"),
    _r("jvm.signature", r"""Signature\.getInstance\(\s*"(\w*with(?:RSA|ECDSA|DSA)\w*|Ed25519|Ed448|EdDSA)\"""",
       None, "signature", JVM, "JCA signature"),
    _r("jvm.cipher-rsa", r"""Cipher\.getInstance\(\s*"RSA""", "RSA", "encryption", JVM, "JCA RSA cipher"),

    # --- Go ---
    _r("go.rsa", r"\brsa\.(GenerateKey|EncryptOAEP|EncryptPKCS1v15|SignPKCS1v15|SignPSS)\b|\"crypto/rsa\"", "RSA", None, GO,
       "Go crypto/rsa"),
    _r("go.ecdsa", r"\becdsa\.(GenerateKey|Sign|SignASN1)\b|\"crypto/ecdsa\"", "ECDSA", "signature", GO, "Go crypto/ecdsa"),
    _r("go.ecdh", r"\"crypto/ecdh\"|\becdh\.(P256|P384|P521|X25519)\(\)", "ECDH", "key-exchange", GO, "Go crypto/ecdh"),
    _r("go.elliptic", r"\belliptic\.P(?:224|256|384|521)\(\)", "EC", None, GO, "Go crypto/elliptic curve"),
    _r("go.ed25519", r"\"crypto/ed25519\"", "Ed25519", "signature", GO, "Go crypto/ed25519"),
    _r("go.curve25519", r"golang\.org/x/crypto/curve25519", "X25519", "key-exchange", GO, "Go x/crypto/curve25519"),
    _r("go.dsa", r"\"crypto/dsa\"", "DSA", "signature", GO, "Go crypto/dsa"),

    # --- C# ---
    _r("cs.rsa", r"\bRSA\.Create\(|\bRSACryptoServiceProvider\b", "RSA", None, CS, ".NET RSA"),
    _r("cs.ecdsa", r"\bECDsa\.Create\(", "ECDSA", "signature", CS, ".NET ECDSA"),
    _r("cs.ecdh", r"\bECDiffieHellman\.Create\(", "ECDH", "key-exchange", CS, ".NET ECDH"),
    _r("cs.dsa", r"\bDSA\.Create\(", "DSA", "signature", CS, ".NET DSA"),
]

KEY_SIZE_PATTERNS = [
    re.compile(p) for p in (
        r"(?:key_size|modulusLength|keySize|KeySize|bits)\s*[:=]\s*(\d{3,5})",
        r"\.initialize\(\s*(\d{3,5})",
        r"GenerateKey\([^,()]+,\s*(\d{3,5})",
        r"RSA\.(?:generate|Create)\(\s*(\d{3,5})",
        r"newkeys\(\s*(\d{3,5})",
        r"genrsa\b.*?\s(\d{3,5})\b",
        r"dhparam\b.*?\s(\d{3,5})\b",
    )
]
