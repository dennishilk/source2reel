from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Source2Reel isolated Kokoro worker")
    parser.add_argument("--out", required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--language-code", required=True)
    parser.add_argument("--speed", type=float, required=True)
    parser.add_argument("--sample-rate", type=int, required=True)
    parser.add_argument("--pause-ms", type=int, default=110)
    args = parser.parse_args()

    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    text = sys.stdin.read()
    pipeline = KPipeline(lang_code=args.language_code)
    generated = list(pipeline(text, voice=args.voice, speed=args.speed))
    chunks = []
    for index, (_, _, audio) in enumerate(generated):
        chunks.append(audio)
        if index != len(generated) - 1:
            chunks.append(np.zeros(int(args.sample_rate * args.pause_ms / 1000), dtype=np.float32))
    if not chunks:
        raise RuntimeError("Kokoro produced no audio")
    sf.write(args.out, np.concatenate(chunks), args.sample_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
