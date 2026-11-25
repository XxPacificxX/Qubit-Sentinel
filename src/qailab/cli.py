"""Command line: ``qailab qec | noise | scan | report``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _qec(a) -> None:
    from .qec.train import QECConfig, train

    train(QECConfig(distance=a.distance, rounds=a.rounds, p=a.p, backend=a.backend, model=a.model,
                    train_shots=a.train_shots, test_shots=a.test_shots, epochs=a.epochs,
                    eval_ps=tuple(a.eval_p or ()), seed=a.seed, out_dir=a.out))


def _noise(a) -> None:
    from .noise.model import NoiseConfig, run

    run(NoiseConfig(num_circuits=a.circuits, backend=a.backend, shots=a.shots, max_depth=a.max_depth,
                    target=a.target, seed=a.seed, reuse_dataset=a.reuse, out_dir=a.out))


def _scan(a) -> None:
    from .pqc import scan, write_reports

    result = scan(a.path)
    print(f"[scan] {result.files_scanned} files, {len(result.findings)} findings")
    if a.llm:
        from .pqc.llm import triage

        n = triage(result.findings, result.root)
        print(f"[scan] Claude reviewed {n} findings")
    paths = write_reports(result, a.out, llm_used=a.llm)
    for k, p in paths.items():
        print(f"  {k:<8} {p}")
    if a.fail_on:
        from .pqc.risk import LEVELS

        worst = max((LEVELS.index((f.llm or {}).get("risk") or f.risk) for f in result.findings), default=0)
        if worst >= LEVELS.index(a.fail_on):
            sys.exit(1)


def _report(a) -> None:
    """Collect the latest results of all three tracks into one Markdown summary."""
    root = Path(a.results)
    out = ["# Quantum AI Lab: results", ""]
    qec = sorted((root / "qec").glob("*.json"), key=lambda p: p.stat().st_mtime)
    out += ["## 1. Neural decoder vs. PyMatching", ""]
    if qec:
        out += ["| run | p | no decoding | PyMatching | neural |", "|---|---|---|---|---|"]
        for p in qec:
            r = json.loads(p.read_text())
            for e in r["evaluations"]:
                out.append(f"| {p.stem} | {e['p']} | {e['no_decoding']['logical_error_rate']:.4f} | "
                           f"{e['pymatching']['logical_error_rate']:.4f} | {e['neural']['logical_error_rate']:.4f} |")
    else:
        out.append("_No runs yet: `qailab qec`_")
    out += ["", "## 2. Circuit fidelity prediction", ""]
    noise = sorted((root / "noise").glob("results_*.json"))
    if noise:
        out += ["| backend | target | model | MAE | R² | Spearman |", "|---|---|---|---|---|---|"]
        for p in noise:
            r = json.loads(p.read_text())
            for key in ("esp_calibrated", "gradient_boosting"):
                m = r[key]
                out.append(f"| {r['config']['backend']} | {r['config'].get('target', 'hellinger')} | {key} | "
                           f"{m['mae']:.4f} | {m['r2']:.3f} | {m['spearman']:.3f} |")
    else:
        out.append("_No runs yet: `qailab noise`_")
    out += ["", "## 3. Post-quantum readiness scans", ""]
    scans = sorted(root.glob("pqc*/findings.json"))
    if scans:
        for p in scans:
            r = json.loads(p.read_text())
            vuln = sum(f["quantum_vulnerable"] for f in r["findings"])
            out.append(f"- `{Path(r['root']).name}`: {len(r['findings'])} findings, {vuln} quantum-vulnerable "
                       f"([report]({p.parent.name}/report.md))")
    else:
        out.append("_No scans yet: `qailab scan PATH`_")
    dest = root / "SUMMARY.md"
    dest.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"[report] wrote {dest}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="qailab", description="Quantum AI Lab")
    sub = ap.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("qec", help="train a neural surface-code decoder and benchmark it against PyMatching")
    q.add_argument("--distance", type=int, default=3)
    q.add_argument("--rounds", type=int, default=None, help="syndrome rounds (default: distance)")
    q.add_argument("--p", type=float, default=0.03, help="physical error rate")
    q.add_argument("--eval-p", type=float, nargs="*", help="extra error rates to test generalization on")
    q.add_argument("--backend", choices=["auto", "stim", "numpy"], default="auto")
    q.add_argument("--model", choices=["transformer", "mlp"], default="transformer")
    q.add_argument("--train-shots", type=int, default=200_000)
    q.add_argument("--test-shots", type=int, default=50_000)
    q.add_argument("--epochs", type=int, default=6)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--out", default="results/qec")
    q.set_defaults(fn=_qec)

    n = sub.add_parser("noise", help="predict circuit fidelity on a simulated IBM device")
    n.add_argument("--circuits", type=int, default=600)
    n.add_argument("--backend", default="FakeManilaV2", help="fake backend from qiskit_ibm_runtime.fake_provider")
    n.add_argument("--shots", type=int, default=2000)
    n.add_argument("--max-depth", type=int, default=30)
    n.add_argument("--target", choices=["mirror", "hellinger"], default="mirror",
                   help="mirror: P(return to |0>) after C then C^-1 (clean noise signal); "
                        "hellinger: output distribution vs. ideal")
    n.add_argument("--seed", type=int, default=0)
    n.add_argument("--reuse", action="store_true", help="reuse the saved dataset instead of re-simulating")
    n.add_argument("--out", default="results/noise")
    n.set_defaults(fn=_noise)

    s = sub.add_parser("scan", help="inventory quantum-vulnerable cryptography in a codebase")
    s.add_argument("path")
    s.add_argument("--out", default="results/pqc")
    s.add_argument("--llm", action="store_true", help="have Claude review each finding (sends code excerpts to the Anthropic API)")
    s.add_argument("--fail-on", choices=["low", "medium", "high", "critical"], help="exit 1 at or above this risk (for CI)")
    s.set_defaults(fn=_scan)

    r = sub.add_parser("report", help="summarize all results into results/SUMMARY.md")
    r.add_argument("--results", default="results")
    r.set_defaults(fn=_report)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
