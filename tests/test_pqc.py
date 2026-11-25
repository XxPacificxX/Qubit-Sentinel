import json
from pathlib import Path
from types import SimpleNamespace

from qailab.pqc import scan, write_reports
from qailab.pqc.llm import triage

DEMO = Path(__file__).parent.parent / "examples" / "demo_app"


def _by(findings, file, family):
    return [f for f in findings if f.file.endswith(file) and f.family == family]


def test_demo_findings():
    r = scan(DEMO)
    fs = r.findings
    rsa = _by(fs, "auth.py", "RSA")
    assert rsa and rsa[0].key_size == 1024 and rsa[0].risk == "critical"
    assert any(f.curve == "SECP256R1" for f in _by(fs, "auth.py", "EC"))
    assert _by(fs, "auth.py", "ECDH")[0].risk == "high"
    assert _by(fs, "auth.py", "MD5")[0].quantum_vulnerable is False
    assert _by(fs, "server.js", "RSA")[0].key_size == 2048
    assert _by(fs, "server.js", "ECDH")
    assert _by(fs, "Signer.java", "ECDSA") and _by(fs, "Signer.java", "DH")
    assert any(f.key_size == 3072 for f in _by(fs, "keys.go", "RSA"))
    assert _by(fs, "keys.go", "PQC")[0].risk == "info"
    assert _by(fs, "demo_key.pem", "PrivateKey")[0].risk == "critical"
    assert _by(fs, "nginx-tls.conf", "TLS-classical")


def test_comments_and_tests_are_downgraded(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("key = rsa.generate_private_key(public_exponent=65537, key_size=4096)\n")
    (tmp_path / "notes.py").write_text("# we used to call rsa.generate_private_key here\n")
    fs = scan(tmp_path).findings
    test_f = next(f for f in fs if "test_x" in f.file)
    comment_f = next(f for f in fs if "notes" in f.file)
    assert test_f.in_test and test_f.risk == "medium"
    assert comment_f.in_comment and comment_f.risk == "low"


def test_reports(tmp_path):
    r = scan(DEMO)
    paths = write_reports(r, tmp_path)
    cbom = json.loads(paths["cbom"].read_text())
    assert cbom["bomFormat"] == "CycloneDX" and cbom["specVersion"] == "1.6"
    assert all(c["type"] == "cryptographic-asset" for c in cbom["components"])
    assert "quantum-vulnerable" in paths["report"].read_text(encoding="utf-8")


class _FakeMessages:
    """Stands in for client.beta.messages: echoes a review for every finding id in the prompt."""

    def __init__(self):
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        prompt = kw["messages"][0]["content"]
        ids = [line.split(":")[0].strip("- ") for line in prompt.splitlines() if line.startswith("- F")]
        body = {"findings": [{"id": i, "is_real_usage": True, "purpose": "signature", "risk": "low",
                              "rationale": "test", "migration": "use ML-DSA-65"} for i in ids]}
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=json.dumps(body))])


def test_llm_triage_with_fake_client():
    r = scan(DEMO)
    msgs = _FakeMessages()
    client = SimpleNamespace(beta=SimpleNamespace(messages=msgs))
    n = triage(r.findings, r.root, client=client, log=lambda *_: None)
    assert n == len(r.findings)
    assert all(f.llm["risk"] == "low" for f in r.findings)
    call = msgs.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert "BEGIN" not in call["messages"][0]["content"] or "demo_key.pem" in call["messages"][0]["content"]
