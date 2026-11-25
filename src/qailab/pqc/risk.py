"""Risk scoring and migration recommendations."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .rules import CLASSICALLY_WEAK_HASH, PQC, QUANTUM_VULNERABLE

if TYPE_CHECKING:
    from .scanner import Finding

LEVELS = ["info", "low", "medium", "high", "critical"]

KEM = ("ML-KEM-768 (FIPS 203). For TLS, use the hybrid X25519MLKEM768 group so you keep classical "
       "security while adding quantum resistance.")
SIG = ("ML-DSA-65 (FIPS 204) for general signing; SLH-DSA (FIPS 205) for long-lived roots and firmware "
       "where a conservative hash-based scheme is worth the larger signatures.")

LIB_HINT = {
    "python": "Python: liboqs-python (`oqs`) today; watch `cryptography` for native ML-KEM/ML-DSA.",
    "go": "Go 1.24+: `crypto/mlkem` in the standard library; ML-DSA via a vetted module (e.g. Cloudflare CIRCL).",
    "java": "Java: JDK 24+ ships ML-KEM (JEP 496) and ML-DSA (JEP 497); Bouncy Castle for older JDKs.",
    "kotlin": "Kotlin/JVM: JDK 24+ ML-KEM/ML-DSA, or Bouncy Castle.",
    "javascript": "Node.js: builds on OpenSSL 3.5+ expose ML-KEM/ML-DSA; otherwise a vetted library such as @noble/post-quantum.",
    "typescript": "Node.js: builds on OpenSSL 3.5+ expose ML-KEM/ML-DSA; otherwise a vetted library such as @noble/post-quantum.",
    "csharp": ".NET 10+: System.Security.Cryptography.MLKem / MLDsa / SlhDsa.",
}


def _shift(level: str, by: int) -> str:
    return LEVELS[max(0, min(len(LEVELS) - 1, LEVELS.index(level) + by))]


def assess(f: "Finding") -> None:
    reasons: list[str] = []
    f.quantum_vulnerable = f.family in QUANTUM_VULNERABLE

    if f.family in PQC:
        risk, rec = "info", "Already post-quantum. Confirm the parameter set (e.g. ML-KEM-768/1024, ML-DSA-65/87) and that a hybrid mode is used where interoperability matters."
        reasons.append("post-quantum algorithm")
    elif f.usage == "key-material":
        risk = "critical"
        reasons.append("private key material in source control")
        rec = "Remove the key from the repository and its history, rotate it, and load it from a secrets manager. Reissue as hybrid/PQC when your PKI supports it."
    elif f.family in CLASSICALLY_WEAK_HASH:
        risk = "medium"
        reasons.append(f"{f.family} is broken classically (collisions); quantum is not the issue here")
        rec = "Use SHA-256 or SHA3-256 (or BLAKE2/3 for non-standards contexts)."
    elif f.family == "AES-128":
        risk = "low"
        reasons.append("Grover's algorithm roughly halves symmetric security; AES-128 keeps ~64-bit quantum margin")
        rec = "Prefer AES-256 for data that must stay secret for decades."
    elif f.usage in ("key-exchange", "encryption", "protocol"):
        risk = "high"
        reasons.append("harvest-now-decrypt-later: recorded traffic/ciphertext can be decrypted once a quantum computer exists")
        rec = KEM
    elif f.usage == "public-key":
        risk = "high"
        reasons.append(f"{f.family} is broken by Shor's algorithm; usage (encryption vs. signing) not determined statically")
        rec = f"If used for encryption/key transport: {KEM} If used for signing: {SIG}"
    elif f.usage in ("signature", "ssh-key", "certificate"):
        risk = "medium"
        reasons.append("signatures can be forged once a quantum computer exists (no retroactive exposure)")
        rec = SIG if f.usage == "signature" else (
            "Plan hybrid/PQC SSH host and user keys (OpenSSH supports ML-KEM hybrid key exchange; PQC signatures are emerging)."
            if f.usage == "ssh-key" else
            "Inventory the issuing CA and certificate lifetimes; plan hybrid or ML-DSA certificates as your PKI and clients support them.")
    else:
        risk, rec = "medium", "Review manually."

    if f.family in ("RSA", "DH", "DSA") and f.key_size and f.key_size < 2048:
        risk = "critical"
        reasons.append(f"{f.key_size}-bit {f.family} is already weak against classical attacks")
    if f.usage == "key-material":
        pass  # a committed private key is a leak wherever it sits
    elif f.in_comment:
        risk = _shift(risk, -2)
        reasons.append("appears in a comment")
    elif f.in_test:
        risk = _shift(risk, -1)
        reasons.append("test/fixture code")

    if f.quantum_vulnerable:
        reasons.append("NIST IR 8547 (draft) plans to deprecate RSA/ECC at 112-bit security after 2030 and disallow them after 2035")
    f.risk, f.reasons, f.recommendation = risk, reasons, rec
