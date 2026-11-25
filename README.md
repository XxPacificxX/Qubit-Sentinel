# Quantum AI Lab

Three AI projects around one question: **what does computing look like once quantum computers arrive?**

| Track | Question | AI part | Quantum part |
|---|---|---|---|
| `qec` | Can a neural network correct quantum errors as well as the standard algorithm? | Transformer decoder (PyTorch) | Surface code syndromes, PyMatching baseline |
| `noise` | Can we predict how badly a circuit will run before spending hardware time on it? | Gradient-boosted regressor | Qiskit circuits on a simulated IBM device |
| `pqc` | Which cryptography in a codebase will a quantum computer break, and what replaces it? | Claude reviews each finding (optional) | Shor-vulnerable crypto → NIST ML-KEM / ML-DSA / SLH-DSA |

Everything runs on a laptop CPU. One CLI drives all three, and `qailab report` collects the results into `results/SUMMARY.md`.

## Setup

```powershell
python -m venv C:\Users\<you>\.venvs\qal
C:\Users\<you>\.venvs\qal\Scripts\python -m pip install -e ".[dev,llm]"
```

Run it as `python -m qailab ...` (with the venv's `python`).

**Notes for this Windows machine:**
- **Short venv path:** keep the venv at a short path. qiskit-aer fails to load from deep paths ("filename too long").
- **No `qailab.exe`:** Windows Application Control blocks pip-generated `.exe` launchers here, so use `python -m qailab`.
- **No Stim:** Application Control also blocks Stim's compiled module. The decoder track falls back to a NumPy phenomenological noise model automatically. On a machine where Stim loads, `pip install -e ".[stim]"` switches to circuit-level noise.

## 1. Neural decoder for the surface code (`qec`)

```powershell
python -m qailab qec --distance 3 --p 0.03 --eval-p 0.01 0.02 0.05
python -m qailab qec --distance 5 --p 0.02 --train-shots 1000000 --epochs 10   # harder, slower
```

- **Data:** the code samples detection events from a distance-d rotated surface code memory experiment, with d rounds of noisy syndrome measurement.
- **Model:** a transformer treats every detector as a token (whether it fired, plus its row, column and round) and predicts whether the logical qubit flipped.
- **Benchmark:** the same test shots go through PyMatching (minimum-weight perfect matching, the standard decoder).
- **Reported:** logical error rate with 95% intervals, decode time per shot, and a "no decoding" floor. With `--eval-p`, the model is also tested at error rates it wasn't trained on.
- **Tests:** they check the code itself. Checks commute, the logical operator isn't a stabilizer, the detector order matches PyMatching, and a larger code beats a smaller one below threshold.

**First run** (d=3, 3 rounds, phenomenological noise, trained at p=0.03 on 190k shots, about 6.5 min on CPU, 50k test shots per row):

| p | No decoding | PyMatching | Transformer |
|---|---|---|---|
| 0.01 | 8.14% | 1.07% ± 0.09 | 1.00% ± 0.09 |
| 0.02 | 15.23% | 3.89% ± 0.17 | 3.70% ± 0.17 |
| **0.03 (trained)** | 21.56% | 7.65% ± 0.23 | 7.58% ± 0.23 |
| 0.05 | 30.77% | 16.37% ± 0.32 | 16.39% ± 0.32 |

- **Accuracy:** the transformer matches matching-based decoding at every error rate, including ones it never trained on.
- **Speed:** PyMatching decodes in 0.3 µs per shot against 39 µs for the network on CPU, so on speed the classical algorithm still wins by a lot.
- **Where to go next:** neural decoders tend to pull ahead under correlated noise, which matching models poorly, and at larger distances.

## 2. Circuit fidelity prediction (`noise`)

```powershell
python -m qailab noise --circuits 600                     # mirror circuits (default)
python -m qailab noise --circuits 600 --target hellinger  # raw output distributions
python -m qailab noise --backend FakeNairobiV2 --reuse    # other device / reuse saved data
```

Random circuits are transpiled to a simulated IBM device (`qiskit_ibm_runtime.fake_provider`) and run with its noise model. Each circuit is described by features computed after transpiling:
- gate counts and depth
- summed gate and readout errors from the device's calibration data
- schedule length relative to the qubits' coherence time (T2)
- the classic **expected success probability** (ESP = product of (1 − error) over every operation), which is the baseline to beat

**What the first runs show (FakeManilaV2, 600 circuits):**

| Target | ESP calibrated (R² / rank corr.) | Gradient boosting (R² / rank corr.) |
|---|---|---|
| mirror (P(return to \|0⟩)) | 0.980 / 0.988 | 0.980 / 0.987 |
| hellinger (output vs. ideal) | 0.298 / 0.505 | 0.347 / 0.593 |

**What this means:**
- On the simulator, ESP is already nearly perfect for mirror circuits. That's because the simulated noise model is built from the same per-gate error rates ESP multiplies together, so there is nothing left for ML to learn.
- The Hellinger target is much harder to predict. A deep random circuit's ideal output is close to uniform, and noise barely changes a uniform distribution, so the score can stay high however noisy the device is.
- **The interesting experiment is real hardware**, where crosstalk, calibration drift and coherent errors break ESP's assumptions. See next steps.

## 3. Post-quantum crypto scanner (`pqc`)

```powershell
python -m qailab scan path\to\repo                 # static analysis only
python -m qailab scan path\to\repo --llm           # + Claude review of each finding
python -m qailab scan path\to\repo --fail-on high  # exit 1 for CI
```

**What it detects:**
- **Languages:** RSA, ECC (ECDSA/ECDH), DH, DSA, Ed25519 and X25519 in Python, JS/TS, Java/Kotlin, Go and C#.
- **Config and key files:** private keys, certificates, SSH keys, classical TLS cipher suites and JWT algorithms.
- **Other weak crypto:** MD5 and SHA-1 (broken classically) and AES-128 (smaller margin against Grover's algorithm).
- **Already post-quantum:** ML-KEM, ML-DSA and SLH-DSA usage, reported as good news.
- **Python details:** it parses the syntax tree to pull out key sizes and curve names.

**How it scores risk:**
- **Critical:** private keys in the repo, or keys that are already weak classically.
- **High:** key exchange and encryption. Encrypted data recorded today can be decrypted later ("harvest now, decrypt later").
- **Medium:** signatures, which can only be forged after a quantum computer exists.
- **Lowered:** findings in tests and comments.

**Outputs, written to `results/pqc/`:**
- `cbom.cdx.json`: a CycloneDX 1.6 **cryptography bill of materials**
- `findings.json`
- `report.md`: migration advice (ML-KEM-768 / hybrid X25519MLKEM768 for key exchange, ML-DSA-65 / SLH-DSA for signatures) plus post-quantum library pointers for each language found

`--llm` sends **only the lines around each finding** (not whole files) to Claude (`claude-opus-5-5`), one request per file. Claude confirms whether it's real usage, works out what it's used for, adjusts the risk and writes a concrete migration step. It needs `ANTHROPIC_API_KEY` or `ant auth login`, and it costs API credits.

Try it on the bundled demo: `python -m qailab scan examples/demo_app`.

## Layout

```
src/qailab/
  qec/    codes.py (surface code + samplers), models.py (transformer, MLP), train.py
  noise/  circuits.py, features.py, dataset.py, model.py
  pqc/    rules.py, scanner.py, risk.py, report.py (CBOM + Markdown), llm.py (Claude review)
  cli.py
examples/demo_app/   deliberately vulnerable code for the scanner
tests/               python -m pytest
```

## Next steps

- **Real hardware for `noise`:** with a free IBM Quantum account, run the same circuits through `qiskit_ibm_runtime.SamplerV2` and train on real fidelities. Then test whether the model beats ESP when the noise isn't the textbook kind. This is the publishable version of track 2.
- **Bigger codes for `qec`:** try d=5 and d=7, circuit-level noise via Stim (on a machine that allows it), and a graph neural network decoder that reuses one model across distances.
- **Scanner in CI:** add SARIF output so GitHub code scanning shows findings inline, and point it at a real repository.
