# VOICE

## Dennis Explainer Voice — MVP candidate

Engine/model: Kokoro-82M v1.0
Inference: `kokoro==0.9.4`
Voice: `am_michael`
Language: American English (`lang_code='a'`)
Speed: 0.94
Target delivery: clear, calm, technical, natural
Output stem: mono source at model rate, normalized by engine to -16 LUFS / -1.5 dBTP and 48 kHz for mux

This voice does not imitate or clone a named third party. It is a stock Kokoro voicepack.

## Pronunciation rules
Narration source should spell awkward identifiers for speech when needed:
- CP-9951 → “C P ninety-nine fifty-one”
- ARMv6 → “A R M V six”
- /dev/fb1 → “slash dev slash F B one”
- v.666 → “version six sixty-six”

## Freeze gate
The candidate becomes permanent only after the first canonical Kokoro render is listened to on Cthulhu. Once accepted, later episodes reuse this exact model release, voice, speed and pronunciation conventions unless a documented engine-wide migration is approved.
