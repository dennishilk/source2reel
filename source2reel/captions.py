"""Deterministic narration captions; timing can be replaced by an aligner later."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class CaptionEvent:
    start: float
    end: float
    lines: tuple[str, ...]


def caption_enabled(scene: dict, is_video: bool, cfg: dict) -> bool:
    default = (cfg.get("captions", {}).get("enabled", True) and
               scene["type"] not in {"SECTION_TITLE", "OUTRO"})
    return scene.get("captions", {}).get("enabled", default)


def _chunks(narration: str, max_lines: int, max_chars: int, max_words: int):
    words = narration.split()
    chunks: list[tuple[str, ...]] = []
    current: list[str] = []
    lines: list[str] = []
    for word in words:
        if len(word) > max_chars:
            raise ValueError(f"Caption word exceeds {max_chars} characters: {word[:40]}")
        prospective = " ".join(current + [word])
        if current and (len(prospective) > max_chars or len(" ".join(lines + [prospective]).split()) > max_words):
            lines.append(" ".join(current))
            current = []
        if len(lines) == max_lines:
            chunks.append(tuple(lines))
            lines = []
        current.append(word)
        if word.endswith((".", "?", "!", ";", ":")) and lines:
            chunks.append(tuple(lines + [" ".join(current)]))
            lines, current = [], []
    if current: lines.append(" ".join(current))
    if lines: chunks.append(tuple(lines))
    return chunks


def estimate_events(narration: str, voice_duration: float, *, max_lines: int = 2,
                    max_chars: int = 48, max_words: int = 18) -> list[CaptionEvent]:
    if voice_duration <= 0: raise ValueError("voice_duration must be positive")
    if min(max_lines, max_chars, max_words) <= 0: raise ValueError("caption limits must be positive")
    chunks = _chunks(narration, max_lines, max_chars, max_words)
    if not chunks: return []
    weights = [sum(len(re.findall(r"\S+", line)) for line in lines) +
               (0.6 if lines[-1].endswith((".", "?", "!")) else 0) for lines in chunks]
    total = sum(weights)
    events = []
    start = 0.0
    for i, lines in enumerate(chunks):
        end = voice_duration if i == len(chunks)-1 else voice_duration * sum(weights[:i+1]) / total
        events.append(CaptionEvent(start, end, lines))
        start = end
    return events


def _ass_time(seconds: float) -> str:
    centis = round(seconds * 100)
    hours, remainder = divmod(centis, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, cs = divmod(remainder, 100)
    return f"{hours}:{minutes:02}:{secs:02}.{cs:02}"


def write_ass(path: Path, events: list[CaptionEvent]) -> None:
    # 1920x1080 canvas; bottom 144 px margin positions two lines in the
    # reserved information strip beneath evidence and diagram content.
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,DejaVu Sans,46,&H00F2EEE8,&H00F2EEE8,&H00120D08,&H00120D08,-1,0,0,0,100,100,0,0,1,3,0,2,100,100,145,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    def escape(s: str) -> str:
        return s.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", " ")
    rows = [f"Dialogue: 0,{_ass_time(e.start)},{_ass_time(e.end)},Caption,,0,0,0,,"+
            r"\N".join(escape(line) for line in e.lines) for e in events]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")


def ass_filter(path: Path) -> str:
    # Filtergraph quoting has two escape layers: quote FFmpeg's option value,
    # then escape quote/backslash for the outer filtergraph parser.
    raw = str(path.resolve()).replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")
    return "ass=filename='" + raw.replace("\\", "\\\\").replace("'", r"\'") + "'"
