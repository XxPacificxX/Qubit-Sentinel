"""Outputs: CycloneDX 1.6 cryptography bill of materials (CBOM), findings JSON, and a Markdown report."""
from __future__ import annotations

import json
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .risk import LEVELS, LIB_HINT
from .scanner import ScanResult

PRIMITIVE = {"key-exchange": "key-agree", "encryption": "pke", "signature": "signature", "hash": "hash",
             "symmetric": "block-cipher", "protocol": "key-agree", "public-key": "pke", "pqc": "other",
             "ssh-key": "signature", "certificate": "signature", "key-material": "other"}
PQC_LEVEL = {"ML-KEM": 3, "ML-DSA": 3, "SLH-DSA": 1, "PQC": 1, "AES-128": 1}


def _effective_risk(f) -> str:
    return (f.llm or {}).get("risk") or f.risk


def to_cyclonedx(result: ScanResult) -> dict:
    groups = defaultdict(list)
    for f in result.findings:
        groups[(f.family, f.usage, f.key_size, f.curve)].append(f)
    components = []
    for (family, usage, key_size, curve), fs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        name = family + (f"-{key_size}" if key_size else "") + (f" ({curve})" if curve else "")
        props = {"primitive": PRIMITIVE.get(usage, "unknown"),
                 "nistQuantumSecurityLevel": 0 if fs[0].quantum_vulnerable else PQC_LEVEL.get(family, 0)}
        if curve:
            props["curve"] = curve
        if key_size:
            props["parameterSetIdentifier"] = str(key_size)
        components.append({
            "type": "cryptographic-asset",
            "bom-ref": f"crypto-{len(components) + 1}",
            "name": name,
            "cryptoProperties": {"assetType": "algorithm", "algorithmProperties": props},
            "evidence": {"occurrences": [{"location": f.file, "line": f.line} for f in fs]},
            "properties": [
                {"name": "qailab:usage", "value": usage},
                {"name": "qailab:max-risk", "value": max((_effective_risk(f) for f in fs), key=LEVELS.index)},
                {"name": "qailab:quantum-vulnerable", "value": str(fs[0].quantum_vulnerable).lower()},
            ],
        })
    return {
        "bomFormat": "CycloneDX", "specVersion": "1.6", "serialNumber": f"urn:uuid:{uuid.uuid4()}", "version": 1,
        "metadata": {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "tools": {"components": [{"type": "application", "name": "qailab-pqc-scan"}]},
                     "component": {"type": "application", "name": Path(result.root).name}},
        "components": components,
    }


def to_markdown(result: ScanResult, llm_used: bool) -> str:
    fs = sorted(result.findings, key=lambda f: (-LEVELS.index(_effective_risk(f)), f.file, f.line))
    by_risk = Counter(_effective_risk(f) for f in fs)
    vuln = sum(f.quantum_vulnerable for f in fs)
    pqc = sum(f.family in ("ML-KEM", "ML-DSA", "SLH-DSA", "PQC") for f in fs)
    lines = [
        f"# Post-quantum readiness: `{Path(result.root).name}`", "",
        f"Scanned {result.files_scanned} files, found {len(fs)} cryptographic usages: "
        f"**{vuln} quantum-vulnerable**, {pqc} already post-quantum."
        + (" Findings were reviewed by Claude." if llm_used else " Static analysis only (run with `--llm` for a Claude review)."),
        "", "| Risk | Count |", "|---|---|",
    ]
    lines += [f"| {lvl} | {by_risk[lvl]} |" for lvl in reversed(LEVELS) if by_risk[lvl]]
    lines += ["", "## Findings", "", "| Risk | Location | Algorithm | Usage | Note |", "|---|---|---|---|---|"]
    for f in fs:
        algo = f.family + (f"-{f.key_size}" if f.key_size else "") + (f" ({f.curve})" if f.curve else "")
        note = (f.llm or {}).get("rationale") or (f.reasons[0] if f.reasons else "")
        if f.llm and f.llm.get("is_real_usage") is False:
            note = "Likely false positive: " + note
        lines.append(f"| {_effective_risk(f)} | `{f.file}:{f.line}` | {algo} | {(f.llm or {}).get('purpose') or f.usage} "
                     f"| {note.replace('|', '/')} |")
    lines += ["", "## How to migrate", ""]
    seen = set()
    for f in fs:
        rec = (f.llm or {}).get("migration") or f.recommendation
        if not rec or rec in seen:
            continue
        seen.add(rec)
        same = [g for g in fs if ((g.llm or {}).get("migration") or g.recommendation) == rec]
        families = ", ".join(sorted({g.family for g in same}))
        where = ", ".join(f"`{w}`" for w in sorted({g.file for g in same}))
        lines.append(f"- **{families}** ({where}): {rec}")
    langs = sorted({f.language for f in fs if f.quantum_vulnerable and f.language in LIB_HINT})
    if langs:
        lines += ["", "### Post-quantum libraries for the languages found", ""]
        lines += [f"- {LIB_HINT[lang]}" for lang in langs]
    return "\n".join(lines) + "\n"


def write_reports(result: ScanResult, out_dir: str | Path, llm_used: bool = False) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"cbom": out / "cbom.cdx.json", "findings": out / "findings.json", "report": out / "report.md"}
    paths["cbom"].write_text(json.dumps(to_cyclonedx(result), indent=2))
    paths["findings"].write_text(json.dumps({"root": result.root, "files_scanned": result.files_scanned,
                                             "findings": [f.to_dict() for f in result.findings]}, indent=2))
    paths["report"].write_text(to_markdown(result, llm_used), encoding="utf-8")
    return paths
