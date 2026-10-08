# Voice

## Default Dennis Explainer voice

| Setting | Value |
| --- | --- |
| Model | Kokoro-82M v1.0 |
| Inference package | `kokoro==0.9.4` |
| Stock voice | `am_michael` |
| Language | American English, `lang_code="a"` |
| Speed | 0.94 |
| Model output | 24 kHz mono |
| Video normalization | -16 LUFS / -1.5 dBTP, resampled to 48 kHz |

The stock voice was accepted in the physical Cthulhu reference render on
23 September 2026. [voices/dennis-explainer.toml](voices/dennis-explainer.toml)
records that approval; [PROJECT_STATE.md](PROJECT_STATE.md) records the
reference episode. This does not replace a listening check after a fresh
installation or a later voice/runtime change.

The voice is a supplied Kokoro voicepack and does not clone or imitate a named
third party.

## Separate CPU runtime

Kokoro and Torch are optional and stay outside the Source2Reel core dependency
graph. Install the voice stack explicitly:

```bash
./tools/setup-voice.sh
```

It uses `.venv-voice-kokoro/bin/python`, selected by the voice profile.
Setup installs the English `en_core_web_sm==3.8.0` spaCy model into that
environment and verifies `spacy.load("en_core_web_sm")` and
`KPipeline(lang_code="a")` before narration.

CPU Torch comes from the official CPU wheel index. Kokoro's remaining
dependencies are installed separately, then `kokoro==0.9.4` is installed with
`--no-deps`. The core resolver does not install a CUDA/NVIDIA Torch stack.

Local LLM inference can use Vulkan independently of the CPU voice stack.

## Listen to the configured voice

```bash
./s2r voice-test
```

The normalized sample is written to `output/tests/voice-test.wav`.
Listen to it on the target machine before producing an episode.

## Pronunciation rules

The default profile applies these substitutions before Kokoro narration:

| Source text | Spoken form |
| --- | --- |
| CP-9951 | C P ninety-nine fifty-one |
| ARMv6 | A R M V six |
| /dev/fb1 | slash dev slash F B one |
| v.666 | version six sixty-six |

## eSpeak preview

`--preview-espeak` explicitly permits an eSpeak fallback when Kokoro cannot
render a scene. The preview supports either an `espeak` or `espeak-ng`
executable and still normalizes the resulting audio through FFmpeg.

The preview is for smoke tests. It does not change the configured production
voice or mark a Kokoro render as accepted.
