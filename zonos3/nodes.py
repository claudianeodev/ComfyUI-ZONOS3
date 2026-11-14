"""ComfyUI nodes for Zyphra's ZONOS3-Mini text-to-speech.

Five nodes: Server, Voice, Voice Blend, Voice Design and Text to Speech. The model runs in Zyphra's own local server
(ggml on Windows/Linux, MLX on Mac); these nodes send it requests and return standard ComfyUI AUDIO.
"""
from __future__ import annotations

import base64
import io
import json
import wave
from pathlib import Path

import numpy as np
import torch

from .client import Zonos3Client, Zonos3Error
from .launcher import ensure_server

DATA = Path(__file__).parent / "data"
SPEC = json.loads((DATA / "controls.json").read_text(encoding="utf-8"))
LANGS = json.loads((DATA / "languages.json").read_text(encoding="utf-8"))
VOICES = json.loads((DATA / "voices.json").read_text(encoding="utf-8"))

SAMPLE_RATE = 44100
MAX_CLIP_SECONDS = 60
CATEGORY = "audio/ZONOS3"
CLONE = "Clone from audio"
LANG_CHOICES = [f"{l['label']} ({l['code']})" for l in LANGS]
VOICE_CHOICES = [f"{v['name']} ({v['language']}, {v['gender']}{'' if v['style'] in ('—', '-') else ', ' + v['style']})"
                 for v in VOICES]
VOICE_BY_CHOICE = dict(zip(VOICE_CHOICES, VOICES))
ACCENT_CHOICES = ["none"] + [a["label"] for a in SPEC["accents"]]
PRESET_CHOICES = ["none"] + list(SPEC["presets"])
RAW_SETS = set(SPEC["raw_sets"])
STEER = SPEC["steer_release"]
CLONE_METHODS = ["voice profile (stable)", "raw clip (closest match)"]


def _lang_code(choice: str) -> str:
    return choice.rsplit("(", 1)[-1].rstrip(")") if "(" in choice else choice


def _clone_mode(method: str) -> str:
    return "reference" if method.startswith("raw") else "embedding"


def _voice_id(client: Zonos3Client, voice: dict) -> str:
    """The server id of a built-in voice: the shipped id, else matched by name in GET /v2/speakers."""
    speakers = client.speakers()
    ids = {s["id"] for s in speakers}
    if voice["id"] in ids:
        return voice["id"]
    name = voice["name"].lower()
    for s in speakers:
        hay = f"{s.get('label') or ''} {s.get('original_name') or ''}".lower()
        if hay.split("/")[-1].startswith(name) or f"/{name}_" in hay or hay.startswith(name + "_"):
            return s["id"]
    raise Zonos3Error(
        f"The server has no built-in voice '{voice['name']}'. Download the voices "
        "(hf download Zyphra/ZONOS3-Mini --include \"gguf/voices/*\") and restart the server."
    )


def _audio_to_wav_b64(audio: dict) -> tuple[str, float]:
    wav = audio["waveform"]
    sr = int(audio["sample_rate"])
    if wav.dim() == 3:
        wav = wav[0]
    mono = wav.float().mean(dim=0).cpu().numpy()
    mono = mono[: sr * MAX_CLIP_SECONDS]
    seconds = len(mono) / sr
    if seconds < 1.0:
        raise Zonos3Error("The clone clip is shorter than a second. Use 10–30 seconds of clear speech.")
    pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).round().astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii"), seconds


# ---------------------------------------------------------------------------------------------------------- Server
class Zonos3Server:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server_url": ("STRING", {"default": "http://127.0.0.1:1919"}),
                "zonos3_folder": ("STRING", {"default": "", "tooltip":
                    "Folder with zonos3-server.exe (a ZONOS3 release, or the ZONOS3-Mini checkout built with CMake). "
                    "Leave empty if you start the server yourself."}),
                "model_folder": ("STRING", {"default": "", "tooltip":
                    "Folder with backbone-ptq.gguf and its companions. Empty = look in zonos3_folder (models, gguf)."}),
                "device": (["gpu", "cpu"], {"default": "gpu"}),
                "max_seconds": ("INT", {"default": 600, "min": 30, "max": 2000, "step": 30, "tooltip":
                    "Longest audio one generation may make (only used when this node starts the server)."}),
            },
            "optional": {
                "access_token": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("ZONOS3_SERVER",)
    RETURN_NAMES = ("server",)
    FUNCTION = "connect"
    CATEGORY = CATEGORY
    DESCRIPTION = "Connects to the local ZONOS3 server, and starts it if it isn't running and you gave its folder."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # re-check the server every run

    def connect(self, server_url, zonos3_folder, model_folder, device, max_seconds, access_token=""):
        client = ensure_server(server_url, zonos3_folder, model_folder, device, int(max_seconds * 15), access_token)
        return ({"url": client.url, "token": client.token},)


def _client(server: dict) -> Zonos3Client:
    return Zonos3Client(server["url"], server.get("token", ""))


# ---------------------------------------------------------------------------------------------------------- Voice
class Zonos3Voice:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server": ("ZONOS3_SERVER",),
                "voice": ([CLONE] + VOICE_CHOICES, {"default": VOICE_CHOICES[6]}),
                "clone_method": (CLONE_METHODS, {"default": CLONE_METHODS[0], "tooltip":
                    "Voice profile turns the clip into a compact voice print: steadier, and works with every Voice "
                    "Design slider. Raw clip feeds the audio itself: can sound closer, but Who/Texture/Room/Accent "
                    "sliders then switch it to a voice profile."}),
            },
            "optional": {
                "clone_audio": ("AUDIO", {"tooltip": "10–30 s of one person speaking clearly. Up to 60 s is used."}),
            },
        }

    RETURN_TYPES = ("ZONOS3_VOICE",)
    RETURN_NAMES = ("voice",)
    FUNCTION = "make"
    CATEGORY = CATEGORY
    DESCRIPTION = "Picks one of the 27 built-in voices, or clones a voice from an audio clip."

    def make(self, server, voice, clone_method, clone_audio=None):
        mode = _clone_mode(clone_method)
        if voice == CLONE:
            if clone_audio is None:
                raise Zonos3Error("Connect a clip to clone_audio, or pick a built-in voice.")
            b64, seconds = _audio_to_wav_b64(clone_audio)
            if seconds < 5:
                print(f"[ZONOS3] the clone clip is only {seconds:.1f} s; 10–30 s clones much better.")
            return ({"label": "cloned voice", "speaker_mode": mode, "speaker_audio_base64": b64,
                     "speaker_audio_name": "clone.wav"},)
        v = VOICE_BY_CHOICE[voice]
        return ({"label": v["name"], "speaker_mode": mode, "speaker_embedding_id": _voice_id(_client(server), v)},)


# ---------------------------------------------------------------------------------------------------------- Blend
class Zonos3VoiceBlend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server": ("ZONOS3_SERVER",),
                "voice_a": (VOICE_CHOICES, {"default": VOICE_CHOICES[1]}),
                "voice_b": (VOICE_CHOICES, {"default": VOICE_CHOICES[4]}),
                "blend": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                     "tooltip": "0 = all voice A, 1 = all voice B."}),
            },
        }

    RETURN_TYPES = ("ZONOS3_VOICE",)
    RETURN_NAMES = ("voice",)
    FUNCTION = "make"
    CATEGORY = CATEGORY
    DESCRIPTION = "Morphs between two built-in voices."

    def make(self, server, voice_a, voice_b, blend):
        c = _client(server)
        a, b = VOICE_BY_CHOICE[voice_a], VOICE_BY_CHOICE[voice_b]
        return ({"label": f"{a['name']}–{b['name']} {blend:.2f}", "speaker_mode": "embedding",
                 "speaker_blend_embedding_id_a": _voice_id(c, a), "speaker_blend_embedding_id_b": _voice_id(c, b),
                 "speaker_blend_t": float(blend)},)


# ---------------------------------------------------------------------------------------------------------- Design
def _slider(c: dict):
    lo, hi = c["safe"]
    tip = c.get("effect") or ""
    if c["type"] == "bipolar":
        tip = f"{c['poles'][0]} ← → {c['poles'][1]}. {tip}"
    return ("FLOAT", {"default": 0.0, "min": float(lo), "max": float(hi), "step": 0.1, "tooltip": tip.strip()})


def _add(controls: dict, mapping: dict, amount: float):
    for key, w in mapping.items():
        controls[key] = controls.get(key, 0.0) + w * amount


class Zonos3VoiceDesign:
    @classmethod
    def INPUT_TYPES(cls):
        req = {"preset": (PRESET_CHOICES, {"default": "none", "tooltip": "A starting point; the sliders add to it."})}
        for c in SPEC["controls"]:
            req[c["id"]] = _slider(c)
        req["accent"] = (ACCENT_CHOICES, {"default": "none"})
        req["accent_strength"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.1})
        req["steer_hold"] = ("FLOAT", {"default": STEER["hold"], "min": 0.0, "max": 1.0, "step": 0.05, "tooltip":
            f"Delivery and mood run at full strength for the first {STEER['onset_seconds']:g} s, then relax to this "
            "share so long reads don't drift louder and higher. 1 = constant."})
        return {"required": req}

    RETURN_TYPES = ("ZONOS3_DESIGN",)
    RETURN_NAMES = ("design",)
    FUNCTION = "make"
    CATEGORY = CATEGORY
    DESCRIPTION = ("25 sliders: who (gender, age, pitch, intonation), texture, room, delivery and mood. "
                   "Slider ranges stop where Zyphra measured the voice still sounds like itself. Go easy when stacking Delivery "
                   "and Mood: Liveliness, Happy and Chatty together raise the pitch a lot.")

    def make(self, preset, accent, accent_strength, steer_hold, **sliders):
        values = {c["id"]: 0.0 for c in SPEC["controls"]}
        accents = {}
        if preset != "none":
            for k, v in SPEC["presets"][preset].items():
                if k.startswith("accent:"):
                    accents[k.split(":", 1)[1]] = float(v)
                elif k in values:
                    values[k] += float(v)
        for k, v in sliders.items():
            if k in values:
                values[k] += float(v)
        if accent != "none":
            accents = {a["id"]: accent_strength for a in SPEC["accents"] if a["label"] == accent}

        controls: dict[str, float] = {}
        by_id = {c["id"]: c for c in SPEC["controls"]}
        for cid, v in values.items():
            c = by_id[cid]
            lo, hi = c["range"]
            v = min(max(v, lo), hi)  # presets + sliders may stack past the safe range; never past the hard range
            if v > 0:
                _add(controls, c["map"]["+"], v)
            elif v < 0 and "-" in c["map"]:
                _add(controls, c["map"]["-"], -v)
        acc_by_id = {a["id"]: a for a in SPEC["accents"]}
        for aid, v in accents.items():
            if v > 0 and aid in acc_by_id:
                _add(controls, acc_by_id[aid]["map"], min(v, acc_by_id[aid]["safe"][1]))
        controls = {k: round(v, 4) for k, v in controls.items() if abs(v) > 1e-6}
        design = {"controls": controls}
        if controls and abs(steer_hold - STEER["hold"]) > 1e-6:
            design["steer_hold"] = float(steer_hold)
        return (design,)


# ---------------------------------------------------------------------------------------------------------- TTS
class Zonos3TextToSpeech:
    _laci_cache: dict[str, set] = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server": ("ZONOS3_SERVER",),
                "voice": ("ZONOS3_VOICE",),
                "text": ("STRING", {"multiline": True, "default":
                    "Hi! This voice was made on my own computer, in ComfyUI, with ZONOS3-Mini."}),
                "language": (LANG_CHOICES, {"default": LANG_CHOICES[0]}),
                "seed": ("INT", {"default": 7, "min": 0, "max": 4294967295, "control_after_generate": "fixed"}),
                "quality_guard": ("BOOLEAN", {"default": True, "tooltip":
                    "Retakes words the model skips or garbles (Zyphra's LACI). Needs the server started with its "
                    "alignment heads; the Server node does that when it starts the server."}),
            },
            "optional": {
                "design": ("ZONOS3_DESIGN",),
                "temperature": ("FLOAT", {"default": 0.75, "min": 0.1, "max": 1.5, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "speak"
    CATEGORY = CATEGORY
    DESCRIPTION = "Turns text into speech with a ZONOS3 voice. 43 languages, up to 10 minutes per run."

    def _laci_languages(self, client: Zonos3Client) -> set:
        if not self._laci_cache.get(client.url):  # re-ask while empty: the server may have restarted with heads
            caps = client.capabilities()
            self._laci_cache[client.url] = set(caps.get("laci_languages") or [])
        return self._laci_cache[client.url]

    def speak(self, server, voice, text, language, seed, quality_guard, design=None, temperature=0.75):
        if not text.strip():
            raise Zonos3Error("The text is empty.")
        client = _client(server)
        lang = _lang_code(language)
        req = {k: v for k, v in voice.items() if k != "label"}
        req.update({"text": text, "language": lang, "seed": int(seed), "temperature": float(temperature)})
        if design and design.get("controls"):
            req["controls"] = dict(design["controls"])
            if "steer_hold" in design:
                req["steer_hold"] = design["steer_hold"]
            raw = any(k.split(".", 1)[0] in RAW_SETS for k in req["controls"])
            if raw and req.get("speaker_mode") == "reference":
                print("[ZONOS3] Who/Texture/Room/Accent sliders need a voice profile; using one for this voice.")
                req["speaker_mode"] = "embedding"
        if quality_guard:
            if lang in self._laci_languages(client):
                req["laci"] = True
            else:
                print(f"[ZONOS3] quality guard is off for {lang}: the server has no alignment heads for it.")
        audio, headers = client.generate(req)
        if audio.size == 0:
            raise Zonos3Error("The server returned no audio.")
        sr = int(headers.get("x-audio-sample-rate", SAMPLE_RATE))
        print(f"[ZONOS3] {len(audio) / sr:.1f} s, seed {headers.get('x-seed', seed)}, voice {voice.get('label')}")
        wav = torch.from_numpy(audio).reshape(1, 1, -1)
        return ({"waveform": wav, "sample_rate": sr},)


NODE_CLASS_MAPPINGS = {
    "ZONOS3Server": Zonos3Server,
    "ZONOS3Voice": Zonos3Voice,
    "ZONOS3VoiceBlend": Zonos3VoiceBlend,
    "ZONOS3VoiceDesign": Zonos3VoiceDesign,
    "ZONOS3TextToSpeech": Zonos3TextToSpeech,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ZONOS3Server": "ZONOS3 Server",
    "ZONOS3Voice": "ZONOS3 Voice",
    "ZONOS3VoiceBlend": "ZONOS3 Voice Blend",
    "ZONOS3VoiceDesign": "ZONOS3 Voice Design",
    "ZONOS3TextToSpeech": "ZONOS3 Text to Speech",
}
