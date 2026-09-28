"""Configuration, secrets and model registry. Config != secrets: secrets never live in YAML."""
from __future__ import annotations
import glob, hashlib, json, os, pathlib, re, threading
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"


def on_kaggle() -> bool:
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.isdir("/kaggle/working")


def _load_env_file(path: pathlib.Path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


# ---------------------------------------------------------------- secrets
_REDACT: set[str] = set()
_SECRET_NAME_RE = re.compile(r"(TOKEN|SECRET|PASSWORD|API_KEY|SERVICE_KEY|DATABASE_URL|_KEY$)", re.I)


class Secrets:
    """Resolution order: process env -> Kaggle Secrets (UserSecretsClient). Values are registered for log redaction."""
    KNOWN = ("DATABASE_URL", "SUPABASE_URL", "SUPABASE_SERVICE_KEY", "GITHUB_TOKEN", "ASTRA_API_TOKEN", "API_KEY")

    def __init__(self):
        self._vals: dict[str, str] = {}
        self._missing: set[str] = set()
        self._client = None
        self._lock = threading.Lock()

    def get(self, name: str, default=None):
        with self._lock:
            if name in self._vals:
                return self._vals[name]
            v = os.environ.get(name)
            if not v and name not in self._missing:
                v = self._kaggle(name)
                if not v:
                    self._missing.add(name)
            if v:
                self._vals[name] = v
                if len(v) >= 6:
                    _REDACT.add(v)
            return v or default

    def _kaggle(self, name):
        try:
            if self._client is None:
                from kaggle_secrets import UserSecretsClient  # only exists on Kaggle
                self._client = UserSecretsClient()
            return self._client.get_secret(name)
        except Exception:
            return None

    def prime(self):
        for n in self.KNOWN:
            self.get(n)

    def available(self) -> dict:
        return {n: bool(self.get(n)) for n in self.KNOWN}

    @staticmethod
    def redact(text: str) -> str:
        for v in _REDACT:
            if v and v in text:
                text = text.replace(v, "***REDACTED***")
        return text

    def resolve(self, template: str, extra: dict | None = None) -> str:
        """Replace ${NAME} with secret/env/extra values. Raises KeyError naming the missing variable."""
        def sub(m):
            n = m.group(1)
            v = (extra or {}).get(n) or self.get(n)
            if v is None:
                raise KeyError(n)
            return v
        return re.sub(r"\$\{([A-Z0-9_]+)\}", sub, template)


def scrubbed_env(extra: dict | None = None) -> dict:
    """Environment for child processes with anything secret-looking removed."""
    env = {k: v for k, v in os.environ.items() if not _SECRET_NAME_RE.search(k)}
    env.update(extra or {})
    return env


# ---------------------------------------------------------------- config
class Config:
    def __init__(self, config_dir=None, overrides: dict | None = None):
        _load_env_file(ROOT / ".env")
        self.dir = pathlib.Path(config_dir or os.environ.get("ASTRA_CONFIG_DIR") or CONFIG_DIR)
        self.runtime = yaml.safe_load((self.dir / "runtime.yaml").read_text())
        m = yaml.safe_load((self.dir / "models.yaml").read_text())
        self.models: dict = m["models"]
        self.hermes: dict = m.get("hermes", {})
        self.mcp: dict = (yaml.safe_load((self.dir / "mcp.yaml").read_text()) or {}).get("mcp", {})
        if overrides:
            self.runtime = _merge(self.runtime, overrides)
        if os.environ.get("ASTRA_BACKEND"):
            self.runtime["backend"] = os.environ["ASTRA_BACKEND"]
        self.work_dir = self._work_dir()
        self.models_dir = self._models_dir()
        self.logs_dir = self.work_dir / "logs"
        self.results_dir = pathlib.Path(os.environ.get("ASTRA_RESULTS_DIR") or ROOT / "benchmarks" / "results")
        for d in (self.work_dir, self.logs_dir, self.workspace_root):
            d.mkdir(parents=True, exist_ok=True)
        self._load_best()

    def __getitem__(self, k):
        return self.runtime[k]

    @property
    def workspace_root(self) -> pathlib.Path:
        w = self.runtime["tools"]["workspace_root"]
        return self.work_dir / "workspace" if w == "auto" else pathlib.Path(w)

    def _work_dir(self):
        v = os.environ.get("ASTRA_WORK_DIR") or self.runtime["work_dir"]
        if v == "auto":
            return pathlib.Path("/kaggle/working/astra_work") if on_kaggle() else pathlib.Path.cwd() / "astra_work"
        return pathlib.Path(v)

    def _models_dir(self):
        v = os.environ.get("ASTRA_MODELS_DIR") or self.runtime["models_dir"]
        if v != "auto":
            return pathlib.Path(v)
        files = [m["file"] for m in self.models.values()]
        if on_kaggle():
            for base in sorted(glob.glob("/kaggle/input/*")):
                for depth in ("", "*/", "*/*/"):
                    if any(glob.glob(f"{base}/{depth}{f}") for f in files):
                        hit = next(p for f in files for p in glob.glob(f"{base}/{depth}{f}"))
                        return pathlib.Path(hit).parent
        return self.work_dir / "models"

    def find_model_file(self, name: str):
        """Locate a GGUF by filename (case-insensitive) in models_dir, then anywhere under /kaggle/input (depth<=6). Cached."""
        cache = self.__dict__.setdefault("_found", {})
        if name in cache:
            return cache[name]
        hit = None
        low = name.lower()
        for root in (self.models_dir, pathlib.Path("/kaggle/input")):
            if not root.is_dir():
                continue
            base = len(root.parts)
            for dp, dn, fn in os.walk(root, followlinks=True):
                if len(pathlib.Path(dp).parts) - base > 6:
                    dn[:] = []
                    continue
                hit = next((pathlib.Path(dp) / f for f in fn if f.lower() == low), None)
                if hit:
                    break
            if hit:
                break
        cache[name] = hit
        return hit

    def _load_best(self):
        p = self.results_dir / "best.json"
        if p.exists():
            try:
                self.apply_overlay(json.loads(p.read_text()))
            except Exception:
                pass

    def apply_overlay(self, overlay: dict):
        """Benchmarked/degraded values override registry defaults (gpu_layers, context)."""
        for k, v in (overlay or {}).items():
            if k in self.models:
                for f in ("gpu_layers", "context"):
                    if v.get(f) is not None:
                        self.models[k][f] = v[f]


class ModelRegistry:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def keys(self):
        return list(self.cfg.models)

    def spec(self, key: str, overrides: dict | None = None) -> dict:
        if key not in self.cfg.models:
            raise KeyError(f"unknown model '{key}'")
        s = dict(self.cfg.models[key])
        s["key"] = key
        found = self.cfg.find_model_file(s["file"])
        s["path"] = str(found or (self.cfg.models_dir / s["file"]))
        s["backend"] = "mock" if self.cfg["backend"] == "mock" else (s.get("backend") or self.cfg["backend"])  # mock is a global force
        s["n_ctx"] = int(s.get("context", 8192))
        s["n_gpu_layers"] = int(s.get("gpu_layers", -1))
        for k, v in (overrides or {}).items():
            if v is not None:
                s[k] = v
        return s

    def by_capability(self, cap: str) -> list[str]:
        return [k for k, m in self.cfg.models.items() if cap in m.get("capabilities", [])]

    def capabilities(self) -> dict:
        return {k: m.get("capabilities", []) for k, m in self.cfg.models.items()}

    def validate(self, key: str, checksum: bool = False) -> dict:
        s = self.spec(key)
        if s["backend"] == "mock":
            return {"model": key, "status": "ok", "size_mb": 0, "note": "mock backend"}
        p = pathlib.Path(s["path"])
        if not p.exists():
            return {"model": key, "status": "missing", "path": str(p)}
        try:
            size = p.stat().st_size
            with open(p, "rb") as f:
                magic = f.read(4)
        except OSError as e:
            return {"model": key, "status": "unreadable", "error": str(e)}
        if magic != b"GGUF":
            return {"model": key, "status": "corrupt", "error": "bad GGUF magic", "path": str(p)}
        if size < 50 * 1024 * 1024:
            return {"model": key, "status": "corrupt", "error": f"suspiciously small ({size} bytes)"}
        if checksum and s.get("sha256"):
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(8 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != s["sha256"]:
                return {"model": key, "status": "corrupt", "error": "sha256 mismatch"}
        return {"model": key, "status": "ok", "size_mb": size // (1 << 20), "path": str(p)}
