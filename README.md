# ComfyUI-ZONOS3

Zyphra's **ZONOS3-Mini** voice model in ComfyUI, by [Promptus](https://www.promptus.ai). Clone a voice from a short
clip, design a new one with sliders, or use one of 27 built-in voices, in 43 languages.

ZONOS3-Mini runs in Zyphra's own local server, not inside ComfyUI. It pins its own CUDA kernels and audio libraries,
and loading those into ComfyUI would break other node packs. Zyphra's servers for Windows, Linux and Mac all speak the
same API, so this one pack works everywhere. The output is standard ComfyUI `AUDIO`.

Project page: https://claudianeodev.github.io/ComfyUI-ZONOS3/

## Nodes

| Node | What it does |
|---|---|
| **ZONOS3 Server** | Connects to the local ZONOS3 server; on Windows with NVIDIA it installs and starts it for you |
| **ZONOS3 Voice** | One of 27 built-in voices, or a clone from any audio clip (up to 60 s is used) |
| **ZONOS3 Voice Blend** | Morphs between two built-in voices |
| **ZONOS3 Voice Design** | 25 sliders (gender, age, pitch, intonation, texture, room, whisper or loud, pace, chatty, happy, sad, calm and more), 15 accents and 8 presets |
| **ZONOS3 Text to Speech** | Text in, audio out. 43 languages, with a quality guard that retakes skipped words |

Two templates ship in `example_workflows`: **ZONOS3 Clone Your Voice** and **ZONOS3 Design a Voice**.

The sliders stop at the ranges Zyphra measured as keeping the voice recognisable. A default "steer hold" relaxes
delivery and mood after about 4 seconds, so long reads don't drift louder and higher.

## Install

### Promptus (one click)

In Promptus, open **CosyTemplates**, search for **ZONOS3**, download **ZONOS3 Clone Your Voice** or **ZONOS3 Design a
Voice**, and press Run. The first Run sets everything up by itself (Windows with an NVIDIA GPU).

### ComfyUI

Either clone the pack:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/claudianeodev/ComfyUI-ZONOS3.git
```

or drop the single file [`zonos3_comfyui.py`](https://github.com/claudianeodev/ComfyUI-ZONOS3/releases/download/server-v1/zonos3_comfyui.py)
into `ComfyUI/custom_nodes`. Restart ComfyUI, load one of the two templates and press Run.

### What the first Run does (Windows + NVIDIA)

With `auto_setup` on (the default), the **ZONOS3 Server** node sets itself up in `ComfyUI/models/zonos3`, one folder you
can delete to remove everything:

1. downloads the prebuilt ZONOS3 server for Windows (about 790 MB, from this repo's
   [server-v1 release](https://github.com/claudianeodev/ComfyUI-ZONOS3/releases/tag/server-v1); CUDA 12.8, RTX 20
   to 50 series, ffmpeg included) and checks its SHA-256;
2. downloads the model and the 27 voices from [Zyphra/ZONOS3-Mini](https://huggingface.co/Zyphra/ZONOS3-Mini)
   (about 1.2 GB);
3. starts the server and keeps it running while ComfyUI is open. The log is `models/zonos3/gguf/zonos3-server.log`.

The **raw clip** clone method needs two more files (about 5.6 GB). They download the first time you pick it, and the
server restarts to load them.

### Mac, Linux, or your own server

Auto-setup is Windows-only for now. Run Zyphra's server yourself (see the
[ZONOS3-Mini repo](https://github.com/Zyphra/ZONOS3-Mini)), leave `zonos3_folder` empty, and set `server_url`.

## Good to know

- **Clone clip:** 10–30 seconds of one person speaking clearly works best.
- **Voice profile vs raw clip:** voice profile is steadier and works with every slider. Raw clip can sound closer;
  the Who, Texture, Room and Accent sliders switch it to a voice profile.
- **Mood and delivery** are a nudge, not a guarantee. Try a few seeds.
- **Stacking:** Liveliness, Happy and Chatty each raise the pitch, and together they add up. In our tests, all three at
  their maximum took Thaddeus from about 97 Hz to about 394 Hz; chatty 1.5 + happy 1 kept him near 111 Hz.
- **Consent:** only clone your own voice, or voices you have permission to use.

## Licence

This node pack is MIT (see `LICENSE` and `NOTICE`). The ZONOS3-Mini weights are Zyphra's, under Apache 2.0, and
aren't included here.
