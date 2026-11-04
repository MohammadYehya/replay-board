"""Settings read from `.env` (and the process environment, which wins)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def read_env_file(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines. PowerShell saves UTF-16 and Notepad may add a BOM."""
    if not path.exists():
        return {}
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
        if "\x00" in text:
            raise UnicodeDecodeError("utf-8", raw, 0, 1, "nul bytes: probably utf-16")
    except UnicodeDecodeError:
        text = raw.decode("utf-16")
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.lower().startswith("export "):
            key = key[7:].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        out[key] = value
    return out


@dataclass(frozen=True)
class Settings:
    api_key: str
    api_base: str
    model_spark: str
    model_wonder: str
    model_transcribe: str
    model_sam: str
    cache_dir: Path
    max_parallel: int
    # "auto", "cpu" (never decode lectures on the GPU, see media.gpu_decode_ok), or a value not understood, which
    # media.decode_setting treats as cpu
    video_decode: str = "auto"


# Words for VIDEO_DECODE: people write "off" or "false" as often as "cpu"
_DECODE_CPU = {"cpu", "software", "sw", "off", "no", "none", "false", "0", "disable", "disabled"}
_DECODE_AUTO = {"auto", "gpu", "on", "yes", "true", "1", ""}


def _decode_mode(value: str) -> str:
    v = value.strip().lower()
    return "cpu" if v in _DECODE_CPU else "auto" if v in _DECODE_AUTO else v


def load_settings(env_path: Path | None = None) -> Settings:
    env = read_env_file(env_path or ROOT / ".env")

    def get(name: str, default: str = "") -> str:
        return os.environ.get(name) or env.get(name) or default

    cache = Path(get("META_API_CACHE", str(DATA_DIR / "cache" / "meta")))
    if not cache.is_absolute():
        cache = ROOT / cache
    return Settings(
        api_key=get("MODEL_API_KEY"),
        api_base=get("META_API_BASE", "https://api.meta.ai/v1").rstrip("/"),
        model_spark=get("MODEL_SPARK", "muse-spark-1.3"),
        model_wonder=get("MODEL_WONDER", "muse-wonder-1.0"),
        model_transcribe=get("MODEL_TRANSCRIBE", "muse-voice-transcribe-1.0"),
        model_sam=get("MODEL_SAM", "sam-3.1"),
        cache_dir=cache,
        max_parallel=int(get("META_API_MAX_PARALLEL", "8")),
        video_decode=_decode_mode(get("VIDEO_DECODE", "auto")),
    )
