"""Find and start Zyphra's ggml ZONOS3 server (zonos3-server) for the ZONOS3 Server node.

The model never loads inside ComfyUI: ZONOS3 pins its own CUDA kernels and audio libraries, so it runs as its own
process and the nodes talk to it over HTTP. One server per URL is kept for the life of ComfyUI.
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from .client import Zonos3Client, Zonos3Error

DATA = Path(__file__).parent / "data"
_procs: dict[str, subprocess.Popen] = {}

EXE = "zonos3-server.exe" if os.name == "nt" else "zonos3-server"
# Where the server binary lands: an extracted release (next to start-zonos3.*), or a source checkout built with CMake
# (servers/ggml/build-cuda for -DGGML_CUDA=ON, servers/ggml/build otherwise; MSVC adds bin/Release).
_EXE_CANDIDATES = [
    ".",
    "bin",
    "servers/ggml/build-cuda/bin/Release",
    "servers/ggml/build-cuda/bin",
    "servers/ggml/build-cuda/Release",
    "servers/ggml/build-cuda",
    "servers/ggml/build-vulkan/bin/Release",
    "servers/ggml/build-vulkan/bin",
    "servers/ggml/build/bin/Release",
    "servers/ggml/build/bin",
    "servers/ggml/build/Release",
    "servers/ggml/build",
]
_MODEL_CANDIDATES = ["models", "gguf", "servers/ggml/gguf", "."]
_BACKBONES = ["backbone-ptq.gguf", "backbone-q8_0.gguf", "backbone-f16.gguf", "backbone-qat.gguf"]


def find_server_exe(root: Path) -> Path | None:
    if root.is_file():
        return root
    for rel in _EXE_CANDIDATES:
        p = root / rel / EXE
        if p.is_file():
            return p
    return None


def find_model_dir(root: Path, model_dir: str) -> Path | None:
    cands = [Path(model_dir)] if model_dir.strip() else [root / rel for rel in _MODEL_CANDIDATES]
    for c in cands:
        for b in _BACKBONES:
            if (c / b).is_file():
                return c
    return None


def find_voices_dir(root: Path, models: Path) -> Path | None:
    for c in (models / "voices", root / "default_voices", root / "assets" / "default_voices"):
        if c.is_dir():
            return c
    return None


def laci_args(root: Path) -> list[str]:
    """--laci-heads for every server language (the default *_heads.json, overridden per speech code), as serve.sh does."""
    heads = None
    for c in (root / "laci_heads", root / "assets" / "laci_heads"):
        if (c / "zonos3_mini_heads.json").is_file():
            heads = c
            break
    if heads is None:
        return []
    langs = json.loads((DATA / "languages.json").read_text(encoding="utf-8"))
    out = []
    for lang in langs:
        f = heads / f"zonos3_mini_heads_{lang['speech']}.json"
        if not f.is_file():
            f = heads / "zonos3_mini_heads.json"
        out += ["--laci-heads", f"{lang['code']}={f}"]
    return out


def find_ffmpeg(root: Path, models: Path) -> str | None:
    if os.environ.get("ZONOS3_FFMPEG"):
        return os.environ["ZONOS3_FFMPEG"]
    name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    for c in (models / "bin" / name, root / name, root / "bin" / name):
        if c.is_file():
            return str(c)
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:  # ships with many ComfyUI installs
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def build_command(root: Path, model_dir: str, url: str, device: str, max_frames: int) -> tuple[list[str], dict, Path]:
    exe = find_server_exe(root)
    if exe is None:
        raise Zonos3Error(
            f"No {EXE} found under {root}. Point zonos3_folder at the ZONOS3 release folder or the ZONOS3-Mini "
            "checkout you built with CMake (servers/ggml/build-cuda)."
        )
    root = root if root.is_dir() else exe.parent
    models = find_model_dir(root, model_dir)
    if models is None:
        raise Zonos3Error(
            "No ZONOS3 backbone (backbone-ptq.gguf) found. Download the model files (README, step 2) with:\n"
            "  hf download Zyphra/ZONOS3-Mini gguf/backbone-ptq.gguf gguf/zac-keep2.gguf gguf/spk-encoder.gguf "
            "gguf/steer.gguf --local-dir <zonos3_folder>\n"
            "  hf download Zyphra/ZONOS3-Mini --include \"gguf/voices/*\" --local-dir <zonos3_folder>\n"
            "then set model_folder to the gguf folder."
        )
    backbone = next(models / b for b in _BACKBONES if (models / b).is_file())
    u = urlparse(url)
    cmd = [str(exe), str(backbone), "--host", u.hostname or "127.0.0.1", "--port", str(u.port or 1919)]
    cmd += ["--gpu" if device == "gpu" else "--cpu"]
    for comp in ("spk-encoder", "zac-encoder", "ssl-teacher", "steer"):
        f = models / f"{comp}.gguf"
        if f.is_file():
            cmd += [f"--{comp}", str(f)]
    voices = find_voices_dir(root, models)
    if voices:
        cmd += ["--tts-default-voices-dir", str(voices)]
    cmd += ["--max", str(max_frames)]
    cmd += laci_args(root)
    env = dict(os.environ)
    ff = find_ffmpeg(root, models)
    if ff:
        env["ZONOS3_FFMPEG"] = ff
    return cmd, env, models


def ensure_server(url: str, root: str, model_dir: str, device: str, max_frames: int, token: str,
                  startup_timeout: float = 300.0) -> Zonos3Client:
    client = Zonos3Client(url, token)
    if client.healthy():
        return client
    if not root.strip():
        raise Zonos3Error(
            f"No ZONOS3 server is answering at {url}. Start it (start-zonos3.bat), or set zonos3_folder on the "
            "ZONOS3 Server node so it starts itself."
        )
    proc = _procs.get(url)
    if proc is None or proc.poll() is not None:
        cmd, env, models = build_command(Path(root.strip().strip('"')), model_dir.strip().strip('"'), url, device,
                                         max_frames)
        log_path = models / "zonos3-server.log"
        log = open(log_path, "ab")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        print(f"[ZONOS3] starting {cmd[0]} {cmd[1]} (log: {log_path})", file=sys.stderr)
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(models),
                                creationflags=flags)
        proc._zonos3_log = log_path  # type: ignore[attr-defined]
        _procs[url] = proc
    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = _tail(getattr(proc, "_zonos3_log", None))
            raise Zonos3Error(f"The ZONOS3 server exited during startup (code {proc.returncode}).\n{tail}")
        if client.healthy(timeout=1.0):
            return client
        time.sleep(0.5)
    raise Zonos3Error(f"The ZONOS3 server did not answer at {url} within {int(startup_timeout)} s.")


def _tail(path, n: int = 25) -> str:
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""


@atexit.register
def _stop_all():
    for p in _procs.values():
        if p.poll() is None:
            p.terminate()
