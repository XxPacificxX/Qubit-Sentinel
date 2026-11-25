"""Walk a codebase and inventory its cryptography."""
from __future__ import annotations

import ast
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .risk import assess
from .rules import FAMILY_USAGE, KEY_SIZE_PATTERNS, RULES, normalize_family

SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env", "__pycache__", "dist", "build",
             ".tox", ".mypy_cache", ".pytest_cache", "site-packages", "target", ".idea", ".vscode", ".next"}
SKIP_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".gz", ".tar", ".whl", ".pyc", ".so",
             ".dll", ".exe", ".bin", ".pt", ".onnx", ".npy", ".npz", ".lock", ".svg", ".woff", ".woff2", ".ttf",
             ".mp4", ".mp3", ".wav", ".ipynb"}
TEST_DIRS = {"test", "tests", "__tests__", "spec", "specs", "fixtures", "testdata"}
TEST_FILE = re.compile(r"(^test_.*|.*_test\.\w+$|.*\.(test|spec)\.\w+$)")
HASH_COMMENT = {".py", ".sh", ".bash", ".rb", ".pl", ".r", ".yaml", ".yml", ".toml", ".conf", ".cfg", ".ini",
                ".env", ".properties", ".tf", ""}
DASH_COMMENT = {".sql", ".lua", ".hs"}
C_COMMENT = ("//", "/*", "* ", "*/")
MAX_BYTES = 1_000_000


def _is_comment(line: str, ext: str) -> bool:
    s = line.lstrip()
    if s.startswith("-----"):  # PEM armor, never a comment
        return False
    if ext in HASH_COMMENT and s.startswith("#"):
        return True
    if ext in DASH_COMMENT and s.startswith("--"):
        return True
    return s.startswith(C_COMMENT) or s == "*"


@dataclass
class Finding:
    id: str
    rule_id: str
    file: str
    line: int
    family: str
    usage: str
    snippet: str
    description: str
    language: str
    key_size: int | None = None
    curve: str | None = None
    in_test: bool = False
    in_comment: bool = False
    quantum_vulnerable: bool = False
    risk: str = "medium"
    reasons: list[str] = field(default_factory=list)
    recommendation: str = ""
    llm: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScanResult:
    root: str
    files_scanned: int
    findings: list[Finding]


LANG = {".py": "python", ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
        ".ts": "typescript", ".tsx": "typescript", ".java": "java", ".kt": "kotlin", ".kts": "kotlin",
        ".scala": "scala", ".go": "go", ".cs": "csharp"}


def _iter_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in SKIP_EXTS or name.endswith(".min.js"):
                continue
            try:
                if p.stat().st_size > MAX_BYTES:
                    continue
            except OSError:
                continue
            yield p


def _read_text(p: Path) -> str | None:
    try:
        raw = p.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw[:8192]:
        return None
    return raw.decode("utf-8", errors="replace")


def _is_test(rel: Path) -> bool:
    return any(part.lower() in TEST_DIRS for part in rel.parts[:-1]) or bool(TEST_FILE.match(rel.name))


def _python_ast_details(text: str) -> dict[int, dict]:
    """Line -> {key_size, curve} from calls like rsa.generate_private_key(public_exponent=..., key_size=1024)."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}

    def dotted(node) -> str:
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts))

    out: dict[int, dict] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = dotted(node.func)
        if not name.endswith(("generate_private_key", "RSA.generate", "newkeys", "ECC.generate")):
            continue
        info: dict = {}
        for kw in node.keywords:
            if kw.arg in ("key_size", "bits") and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, int):
                info["key_size"] = kw.value.value
            if kw.arg == "curve":
                info["curve"] = str(kw.value.value) if isinstance(kw.value, ast.Constant) else dotted(
                    kw.value.func if isinstance(kw.value, ast.Call) else kw.value).split(".")[-1]
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, int) and arg.value >= 256:
                info.setdefault("key_size", arg.value)
            target = arg.func if isinstance(arg, ast.Call) else arg
            tail = dotted(target).split(".")[-1]
            if tail.upper().startswith(("SECP", "BRAINPOOL", "SECT")):
                info.setdefault("curve", tail)
        if info:
            out[node.lineno] = info
    return out


def _key_size_near(lines: list[str], idx: int) -> int | None:
    window = "\n".join(lines[idx: idx + 4])
    for pat in KEY_SIZE_PATTERNS:
        m = pat.search(window)
        if m:
            return int(m.group(1))
    return None


def scan(root: str | Path) -> ScanResult:
    root = Path(root).resolve()
    findings: list[Finding] = []
    seen: set[tuple[str, int, str]] = set()
    files = 0
    for path in _iter_files(root):
        text = _read_text(path)
        if text is None:
            continue
        files += 1
        rel = path.relative_to(root)
        ext = path.suffix.lower()
        lines = text.splitlines()
        ast_info = _python_ast_details(text) if ext == ".py" else {}
        in_test = _is_test(rel)
        for i, line in enumerate(lines):
            for rule in RULES:
                if rule.exts and ext not in rule.exts:
                    continue
                m = rule.pattern.search(line)
                if not m:
                    continue
                if rule.family == "weak-hash":
                    raw = next(g for g in m.groups() if g)
                    family = "MD5" if "5" in raw else "SHA-1"
                elif rule.family is not None:
                    family = rule.family
                elif m.groups() and m.group(1):
                    raw = m.group(1)
                    family = normalize_family(raw[:2] if rule.id == "generic.jwt-alg" else raw)
                else:  # e.g. openssl CLI
                    family = "RSA" if "rsa" in line.lower() else "DH" if "dhparam" in line else "EC"
                key = (str(rel), i + 1, family)
                if key in seen:
                    continue
                seen.add(key)
                usage = rule.usage or FAMILY_USAGE.get(family, "public-key")
                f = Finding(
                    id=f"F{len(findings) + 1:04d}", rule_id=rule.id, file=rel.as_posix(), line=i + 1,
                    family=family, usage=usage, snippet=line.strip()[:200], description=rule.description,
                    language=LANG.get(ext, ext.lstrip(".") or path.name), in_test=in_test,
                    in_comment=_is_comment(line, ext),
                )
                details = ast_info.get(i + 1, {})
                f.curve = details.get("curve")
                if family in ("RSA", "DH", "DSA"):
                    f.key_size = details.get("key_size") or _key_size_near(lines, i)
                assess(f)
                findings.append(f)
    return ScanResult(str(root), files, findings)
