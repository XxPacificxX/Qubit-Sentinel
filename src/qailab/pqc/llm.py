"""Optional Claude review of static findings.

Static rules cannot tell a real RSA key exchange from a string in a docstring, or
whether an RSA key encrypts or signs. For each file with findings, Claude sees
only the lines around each finding (not the whole repository) and returns, per
finding: is it real usage, what it is used for, an adjusted risk, and a concrete
migration step for this code.

Requires ``pip install anthropic`` and credentials (``ANTHROPIC_API_KEY`` or ``ant auth login``).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .risk import LEVELS
from .scanner import Finding

MODEL = "claude-opus-5-5"
CONTEXT_LINES = 4

SYSTEM = """You are a cryptography engineer helping a team prepare a codebase for post-quantum migration.
A static scanner flagged the lines below. For each finding, decide from the surrounding code:
- whether it is real cryptographic usage (not a comment, docstring, test string, or unrelated identifier),
- what it is used for,
- the risk: critical (private keys in source, or keys already weak classically), high (key exchange or
  encryption protecting data that must stay secret: harvest-now-decrypt-later), medium (signatures,
  authentication), low or info (tests, already post-quantum, or not real usage),
- the concrete migration for this code: name the NIST algorithm (ML-KEM FIPS 203, ML-DSA FIPS 204,
  SLH-DSA FIPS 205), prefer hybrid modes for key exchange in protocols, and name a library for this
  language when you are confident one exists.
Keep rationale to one or two sentences. Return exactly one entry per finding id you were given."""

SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "is_real_usage": {"type": "boolean"},
                    "purpose": {"type": "string", "enum": ["key-exchange", "encryption", "signature", "authentication",
                                                           "certificate", "key-storage", "hashing", "other"]},
                    "risk": {"type": "string", "enum": LEVELS},
                    "rationale": {"type": "string"},
                    "migration": {"type": "string"},
                },
                "required": ["id", "is_real_usage", "purpose", "risk", "rationale", "migration"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["findings"],
    "additionalProperties": False,
}


def _excerpt(lines: list[str], fs: list[Finding]) -> str:
    keep: set[int] = set()
    for f in fs:
        keep.update(range(max(1, f.line - CONTEXT_LINES), min(len(lines), f.line + CONTEXT_LINES) + 1))
    out, prev = [], 0
    for n in sorted(keep):
        if prev and n != prev + 1:
            out.append("...")
        out.append(f"{n:>5}| {lines[n - 1]}")
        prev = n
    return "\n".join(out)


def build_prompt(path: str, lines: list[str], fs: list[Finding]) -> str:
    listing = "\n".join(
        f"- {f.id}: line {f.line}, scanner says {f.family} ({f.usage})"
        + (f", key size {f.key_size}" if f.key_size else "") + (f", curve {f.curve}" if f.curve else "")
        + f", static risk {f.risk}" for f in fs)
    return (f"File: {path} (language: {fs[0].language})\n\nFindings:\n{listing}\n\n"
            f"Code excerpt (line numbers on the left):\n```\n{_excerpt(lines, fs)}\n```")


def review_file(client, path: str, lines: list[str], fs: list[Finding]) -> dict[str, dict]:
    """One request per file. Returns finding id -> review."""
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        system=SYSTEM,
        messages=[{"role": "user", "content": build_prompt(path, lines, fs)}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"declined to review {path}")
    if response.stop_reason == "max_tokens":
        raise RuntimeError(f"review of {path} was cut off (max_tokens)")
    text = next(b.text for b in response.content if b.type == "text")
    valid = {f.id for f in fs}
    return {r["id"]: r for r in json.loads(text)["findings"] if r["id"] in valid}


def triage(findings: list[Finding], root: str | Path, client=None, log=print) -> int:
    """Attach Claude's review to ``finding.llm`` in place. Returns how many findings were reviewed."""
    import anthropic

    client = client or anthropic.Anthropic()
    by_file: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        by_file[f.file].append(f)

    reviewed = 0
    for path, fs in by_file.items():
        lines = (Path(root) / path).read_text(encoding="utf-8", errors="replace").splitlines()
        try:
            reviews = review_file(client, path, lines, fs)
        except anthropic.AuthenticationError:
            raise RuntimeError("No valid Anthropic credentials: set ANTHROPIC_API_KEY or run `ant auth login`.") from None
        except anthropic.RateLimitError:
            log(f"  [llm] rate limited on {path}; keeping static results for this file")
            continue
        except (anthropic.APIStatusError, anthropic.APIConnectionError, RuntimeError, json.JSONDecodeError) as e:
            log(f"  [llm] skipped {path}: {e}")
            continue
        for f in fs:
            if f.id in reviews:
                f.llm = {k: v for k, v in reviews[f.id].items() if k != "id"}
                reviewed += 1
        log(f"  [llm] reviewed {path} ({len(reviews)}/{len(fs)} findings)")
    return reviewed
