# VOICE

## Dennis Explainer Voice — MVP candidate

Engine/model: Kokoro-82M v1.0  
Inference: \`kokoro==0.9.4\`  
Voice: \`am_michael\`  
Language: American English (\`lang_code='a'\`)  
Speed: 0.94  
Target delivery: clear, calm, technical, natural  
Model-rate output: 24 kHz mono; Source2Reel normalizes to -16 LUFS / -1.5 dBTP and 48 kHz for mux.

This voice does not imitate or clone a named third party. It is a stock Kokoro voicepack.

## Runtime architecture

Kokoro is intentionally **not a Source2Reel core Python dependency**.

The permanent-voice candidate is installed explicitly with:

\`\`\`bash
./tools/setup-voice.sh
\`\`\`

The script creates the separate runtime:

\`\`\`text
.venv-voice-kokoro/bin/python
\`\`\`

Setup explicitly installs `en_core_web_sm==3.8.0` through `uv pip`
into the voice interpreter. It checks `spacy.load("en_core_web_sm")` and
initializes `KPipeline(lang_code="a")`, preventing Misaki from invoking
pip during first narration. `./s2r voice-test` renders
`output/tests/voice-test.wav` for a physical listening decision.

The Dennis profile records that interpreter in \`voices/dennis-explainer.toml\`.

For the first frozen baseline, Torch is installed from the official CPU wheel index. Kokoro's remaining non-Torch dependencies are installed separately and \`kokoro==0.9.4\` is then installed with \`--no-deps\`. This prevents the reusable core resolver from silently selecting a CUDA/NVIDIA Torch stack.

The local LLM may simultaneously use Vulkan/RADV on the AMD GPU. ROCm is not required for TTS.

## Pronunciation rules

- CP-9951 → “C P ninety-nine fifty-one”
- ARMv6 → “A R M V six”
- /dev/fb1 → “slash dev slash F B one”
- v.666 → “version six sixty-six”

## eSpeak

eSpeak is a renderer smoke-test fallback only. It is not the Dennis Explainer production voice and must never be silently promoted to one.

## Freeze gate

The candidate becomes permanent only after the first canonical Kokoro render is listened to on physical Cthulhu. Once accepted, later episodes reuse the exact model release, voice, speed, runtime policy and pronunciation conventions unless a documented engine-wide migration is approved.
