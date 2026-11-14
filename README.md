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
| **ZONOS3 Server** | Connects to the local ZONOS3 server, and starts it for you if you give it the server's folder |
| **ZONOS3 Voice** | One of 27 built-in voices, or a clone from any audio clip (up to 60 s is used) |
| **ZONOS3 Voice Blend** | Morphs between two built-in voices |
| **ZONOS3 Voice Design** | 25 sliders (gender, age, pitch, intonation, texture, room, whisper or loud, pace, chatty, happy, sad, calm and more), 15 accents and 8 presets |
| **ZONOS3 Text to Speech** | Text in, audio out. 43 languages, with a quality guard that retakes skipped words |

Two templates ship in `example_workflows`: **ZONOS3 Clone Your Voice** and **ZONOS3 Design a Voice**.

The sliders stop at the ranges Zyphra measured as keeping the voice recognisable. A default "steer hold" relaxes
delivery and mood after about 4 seconds, so long reads don't drift louder and higher.

## Install

### 1. The node pack

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/claudianeodev/ComfyUI-ZONOS3.git
```

Restart ComfyUI. The pack needs nothing beyond ComfyUI itself.

### 2. The ZONOS3 server and model

Zyphra doesn't publish prebuilt Windows binaries yet, so build the ggml server once (Git, CMake, Visual Studio 2022
Build Tools with C++, Rust, and the CUDA Toolkit 12.8+ for an NVIDIA GPU):

```bat
git clone --recurse-submodules https://github.com/Zyphra/ZONOS3-Mini.git zonos3
cd zonos3\servers\ggml
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --config Release -j
```

No NVIDIA GPU? Leave out `-DGGML_CUDA=ON` and use `build` instead of `build-cuda`. The CPU build runs about 8–10 times
faster than real time.

Download the model files (about 1.2 GB) into `zonos3\servers\ggml`:

```bat
hf download Zyphra/ZONOS3-Mini gguf/backbone-ptq.gguf gguf/zac-keep2.gguf gguf/spk-encoder.gguf gguf/steer.gguf --local-dir .
hf download Zyphra/ZONOS3-Mini --include "gguf/voices/*" --local-dir .
```

The "raw clip" clone method also needs `gguf/zac-encoder.gguf` and `gguf/ssl-teacher.gguf` (5.6 GB). The default
"voice profile" method doesn't.

You also need `ffmpeg` on your PATH for cloning from a clip.

### 3. Point the Server node at it

On **ZONOS3 Server**, set `zonos3_folder` to the `zonos3` checkout. The node finds `zonos3-server.exe`, the model in
`servers\ggml\gguf`, the 27 voices and the quality-guard files, starts the server and keeps it running while ComfyUI
is open. The server log is `gguf\zonos3-server.log`.

Already running the server yourself (for example `start-zonos3.bat`, or the MLX server on a Mac)? Leave
`zonos3_folder` empty and set `server_url`.

## Good to know

- **Clone clip:** 10–30 seconds of one person speaking clearly works best.
- **Voice profile vs raw clip:** voice profile is steadier and works with every slider. Raw clip can sound closer;
  the Who, Texture, Room and Accent sliders switch it to a voice profile.
- **Mood and delivery** are a nudge, not a guarantee. Try a few seeds.
- **Stacking:** Liveliness, Happy and Chatty each raise the pitch, and together they add up. In our test, chatty 2 +
  lively 3 + happy 2 took a deep announcer voice from about 115 Hz to about 225 Hz. Chatty 1.5 + happy 1 kept it near
  140 Hz.
- **Consent:** clone your own voice, or voices you have written permission to use. Zyphra's acceptable use policy
  bans impersonation and deepfakes.

## Licence

This node pack is MIT (see `LICENSE` and `NOTICE`). The ZONOS3-Mini weights are Zyphra's, under Apache 2.0, and
aren't included here.
