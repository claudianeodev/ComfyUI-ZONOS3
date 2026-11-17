"""First-run setup for ZONOS3: downloads the portable server (with ffmpeg) and the model files into ComfyUI/models/zonos3.

Everything goes into one folder, so deleting that folder removes it all. Downloads are checked (SHA-256 for the server
package and the model files) and written to a .part file first, so a stopped download never leaves a broken file.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from .client import Zonos3Error

PACK_DIR = Path(__file__).resolve().parent.parent
RELEASE = json.loads((Path(__file__).parent / "data" / "release.json").read_text(encoding="utf-8"))
HF_REPO = RELEASE["model_repo"]
HF_REVISION = RELEASE.get("model_revision", "main")
MODEL_FILES = RELEASE["model_files"]  # paths inside the HF repo, e.g. gguf/backbone-ptq.gguf
VOICES_DIR = RELEASE["voices_dir"]    # gguf/voices
REFERENCE_FILES = RELEASE.get("reference_files", [])  # extra encoders for raw-clip cloning (about 5.6 GB)
UA = {"User-Agent": "ComfyUI-ZONOS3"}


def default_root() -> Path:
    try:
        import folder_paths  # ComfyUI

        return Path(folder_paths.models_dir) / "zonos3"
    except Exception:
        return PACK_DIR.parent.parent / "models" / "zonos3"


def _progress(total_bytes: int):
    try:
        from comfy.utils import ProgressBar

        return ProgressBar(max(1, total_bytes))
    except Exception:
        return None


def _download(url: str, dest: Path, sha256: str | None = None, size: int | None = None, label: str = "",
              bar=None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    got = 0
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
            total = size or int(r.headers.get("Content-Length") or 0)
            last = -1
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                h.update(chunk)
                got += len(chunk)
                if bar is not None:
                    bar.update(len(chunk))
                pct = int(got * 100 / total) if total else -1
                if pct >= 0 and pct // 10 != last and total > 20_000_000:
                    last = pct // 10
                    print(f"[ZONOS3] downloading {label or dest.name}: {pct}%", file=sys.stderr)
    except Exception as e:
        part.unlink(missing_ok=True)
        raise Zonos3Error(f"Download failed for {label or dest.name} ({url}): {e}") from None
    if size and got != size:
        part.unlink(missing_ok=True)
        raise Zonos3Error(f"Download of {label or dest.name} was incomplete ({got} of {size} bytes). Press Run again.")
    if sha256 and h.hexdigest().lower() != sha256.lower():
        part.unlink(missing_ok=True)
        raise Zonos3Error(f"Download of {label or dest.name} failed its checksum. Press Run again.")
    os.replace(part, dest)


def _download_parts(parts: list[dict], dest: Path, sha256: str, size: int, label: str, bar=None) -> None:
    """Download a file published as several parts (each checked), join them and check the whole file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    pieces = []
    for i, p in enumerate(parts, 1):
        piece = dest.with_name(f"{dest.name}.{i:03d}")
        if not (piece.is_file() and piece.stat().st_size == p["size"]):
            _download(p["url"], piece, p.get("sha256"), p["size"], f"{label} part {i}/{len(parts)}", bar)
        elif bar is not None:
            bar.update(p["size"])
        pieces.append(piece)
    h = hashlib.sha256()
    tmp = dest.with_name(dest.name + ".part")
    with open(tmp, "wb") as out:
        for piece in pieces:
            with open(piece, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    out.write(chunk)
                    h.update(chunk)
    if tmp.stat().st_size != size or h.hexdigest().lower() != sha256.lower():
        tmp.unlink(missing_ok=True)
        raise Zonos3Error(f"The joined {label} failed its checksum. Press Run again.")
    os.replace(tmp, dest)
    for piece in pieces:
        piece.unlink(missing_ok=True)


def _hf_tree(path: str) -> list[dict]:
    url = f"https://huggingface.co/api/models/{HF_REPO}/tree/{HF_REVISION}/{path}?recursive=true"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
            return [e for e in json.load(r) if e.get("type") == "file"]
    except Exception as e:
        raise Zonos3Error(f"Couldn't list the ZONOS3 model files on Hugging Face: {e}") from None


def _hf_url(path: str) -> str:
    return f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/{urllib.parse.quote(path)}"


def _needed_models(root: Path) -> list[dict]:
    want = set(MODEL_FILES)
    entries = [e for e in _hf_tree("gguf") if e["path"] in want or e["path"].startswith(VOICES_DIR + "/")]
    missing = [p for p in MODEL_FILES if p not in {e["path"] for e in entries}]
    if missing:
        raise Zonos3Error(f"Zyphra's model repo no longer has: {', '.join(missing)}")
    todo = []
    for e in entries:
        dest = root / e["path"]
        size = (e.get("lfs") or {}).get("size") or e.get("size")
        if dest.is_file() and (not size or dest.stat().st_size == size):
            continue
        todo.append({"path": e["path"], "dest": dest, "size": size, "sha256": (e.get("lfs") or {}).get("oid")})
    return todo


def server_dir(root: Path) -> Path:
    return root / "server"


def ffmpeg_path(root: Path) -> Path:
    """The portable server package ships an LGPL ffmpeg build next to the server, for cloning from clips."""
    return server_dir(root) / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")


def is_installed(root: Path) -> bool:
    exe = server_dir(root) / ("zonos3-server.exe" if os.name == "nt" else "zonos3-server")
    return exe.is_file() and all((root / p).is_file() for p in MODEL_FILES)


def _platform_package() -> dict | None:
    if sys.platform == "win32":
        return RELEASE["server"].get("win-cuda")
    return None


def install(root: Path) -> None:
    """Download whatever is missing. Safe to call every run: finished files are skipped."""
    root.mkdir(parents=True, exist_ok=True)
    pkg = _platform_package()
    sdir = server_dir(root)
    exe = sdir / ("zonos3-server.exe" if os.name == "nt" else "zonos3-server")
    stamp = sdir / ".package"

    jobs = []
    if not exe.is_file() or (stamp.read_text().strip() if stamp.is_file() else "") != (pkg or {}).get("sha256", ""):
        if pkg is None:
            raise Zonos3Error(
                "Automatic setup of the ZONOS3 server is only available on Windows with an NVIDIA GPU for now. On "
                "Mac or Linux, start Zyphra's server yourself (README, step 2) and leave zonos3_folder empty."
            )
        jobs.append(("server", pkg["size"]))
    models = _needed_models(root)
    jobs += [("model", m["size"] or 0) for m in models]
    if not jobs:
        return

    total = sum(s for _, s in jobs)
    print(f"[ZONOS3] first-run setup: downloading {total / 1e9:.2f} GB into {root}", file=sys.stderr)
    bar = _progress(total)

    if any(k == "server" for k, _ in jobs):
        zpath = root / "downloads" / "zonos3-server.zip"
        if pkg.get("parts"):
            _download_parts(pkg["parts"], zpath, pkg["sha256"], pkg["size"], "ZONOS3 server", bar)
        else:
            _download(pkg["url"], zpath, pkg["sha256"], pkg["size"], "ZONOS3 server", bar)
        tmp = root / "server.new"
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmp)
        inner = [p for p in tmp.iterdir()]
        src = inner[0] if len(inner) == 1 and inner[0].is_dir() else tmp
        shutil.rmtree(sdir, ignore_errors=True)
        shutil.move(str(src), str(sdir))
        shutil.rmtree(tmp, ignore_errors=True)
        if os.name != "nt":  # zip doesn't keep the executable bit
            for f in sdir.iterdir():
                if f.is_file() and not f.suffix:
                    f.chmod(0o755)
        (sdir / ".package").write_text(pkg["sha256"])
        zpath.unlink(missing_ok=True)

    for m in models:
        _download(_hf_url(m["path"]), m["dest"], m["sha256"], m["size"], m["path"].split("/")[-1], bar)

    shutil.rmtree(root / "downloads", ignore_errors=True)
    print("[ZONOS3] setup complete", file=sys.stderr)


def has_reference_files(root: Path) -> bool:
    return all((root / p).is_file() for p in REFERENCE_FILES)


def install_reference_files(root: Path) -> None:
    """Raw-clip cloning needs two more encoders (about 5.6 GB). Fetched only when someone first picks it."""
    want = set(REFERENCE_FILES)
    todo = []
    for e in _hf_tree("gguf"):
        if e["path"] not in want:
            continue
        dest = root / e["path"]
        size = (e.get("lfs") or {}).get("size") or e.get("size")
        if dest.is_file() and (not size or dest.stat().st_size == size):
            continue
        todo.append({"path": e["path"], "dest": dest, "size": size, "sha256": (e.get("lfs") or {}).get("oid")})
    if not todo:
        return
    total = sum(m["size"] or 0 for m in todo)
    print(f"[ZONOS3] raw-clip cloning: downloading {total / 1e9:.2f} GB of encoders into {root} (one time)",
          file=sys.stderr)
    bar = _progress(total)
    for m in todo:
        _download(_hf_url(m["path"]), m["dest"], m["sha256"], m["size"], m["path"].split("/")[-1], bar)
