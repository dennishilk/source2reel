from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import wave
from unittest.mock import patch

from PIL import Image, ImageChops, ImageDraw
from source2reel.captions import caption_enabled, estimate_events, write_ass
from source2reel.renderer import _encode_static_segment, _encode_video_evidence, _visual_scene
from source2reel.schema import validate_episode, validate_presentation
from source2reel.templates import render_scene, render_video_shell
from source2reel.util import run

ROOT = Path(__file__).resolve().parents[1]
FROZEN_BLOBS = {
  "projects/cisco-doom-episode-001/episode-review-2.json": "f84478d1f19a0f99b9b7a1e9b26ebe9808774f0c",
  "projects/cisco-doom-episode-001/episode-review-3.json": "e35b3a35ed2fc690c6ffc124a40193480da96238",
  "projects/cisco-doom-episode-001/episode.json": "bff3eb875af0285aed07e72db5c231b929e147b5",
  "projects/cisco-doom-episode-001/manifests/e0132-visual-inspection.json": "fd46f3c4daabb77e78c378c18521e13c83c8b1ff",
  "projects/cisco-doom-episode-001/manifests/evidence.json": "18ae14fdb34bc68c9f2e01b603c2551e28fc38f9",
  "projects/cisco-doom-episode-001/manifests/planner-evidence.json": "8b0575df907de9a4fc7bea514cba47d03330b85d",
  "projects/cisco-doom-episode-001/manifests/research-compact-parts/part-01.json": "9fc9e7cea17d61c8f18d202fe8198fc339328532",
  "projects/cisco-doom-episode-001/manifests/research-compact-parts/part-02.json": "e06c36f199a649eb68f21ce031fcaba75bf58216",
  "projects/cisco-doom-episode-001/manifests/research-compact-parts/part-03.json": "859b298ffe9edea47980a2a3690f09356ba356ae",
  "projects/cisco-doom-episode-001/manifests/research-compact-parts/part-04.json": "0e0f0d1ebfbad12f6818db47efc0cb773276d125",
  "projects/cisco-doom-episode-001/manifests/research-compact.json": "5847e5a9de26f0e2bb43113a96450fa26fa642b1",
  "projects/cisco-doom-episode-001/manifests/research.json": "5bc43cceacc1a2a0e3b7a686887656b440447b7e"
}

class VisualProductTests(unittest.TestCase):
    def test_caption_segmentation_and_timing(self):
        narration = ("The phone executes Doom locally. The keypad generates input events. "
                     "Audio reaches the phone local relay over a FIFO. The Cisco media system "
                     "plays audio via loopback. The Applications entry survives a cold boot.")
        events = estimate_events(narration, 24.2, max_chars=34, max_words=11)
        self.assertGreater(len(events), 3)
        self.assertEqual(events[0].start, 0)
        self.assertEqual(events[-1].end, 24.2)
        self.assertEqual(" ".join(" ".join(e.lines) for e in events), narration)
        for a, b in zip(events, events[1:]): self.assertEqual(a.end, b.start)
        for e in events:
            self.assertGreater(e.end, e.start)
            self.assertLessEqual(e.end, 24.2)
            self.assertLessEqual(len(e.lines), 2)
            self.assertTrue(all(len(line) <= 34 for line in e.lines))
            self.assertLessEqual(len(" ".join(e.lines).split()), 11)

    def test_caption_defaults_and_override(self):
        static = {"type": "TERMINAL_EVIDENCE"}
        video = {"type": "HERO"}
        self.assertTrue(caption_enabled(static, False, {}))
        self.assertTrue(caption_enabled(video, True, {}))
        self.assertFalse(caption_enabled(static, False, {"captions": {"enabled": False}}))
        self.assertFalse(caption_enabled(video, True, {"captions": {"enabled": False}}))
        self.assertFalse(caption_enabled({**video, "captions": {"enabled": False}}, True, {}))
        self.assertFalse(caption_enabled({**static, "captions": {"enabled": False}}, False, {}))
        self.assertFalse(caption_enabled({"type": "OUTRO"}, False, {}))

    def test_schema_diagram_and_offset(self):
        scene = {"id": "s1", "type": "DATA_FLOW", "narration": "A to B.",
                 "evidence_refs": ["E1"], "diagram": {}}
        ep = {"version": 1, "title": "Test", "scenes": [scene]}
        with self.assertRaisesRegex(ValueError, "explicit"): validate_episode(ep, {"E1"})
        scene["diagram"] = {"nodes": ["Supported A", "Supported B"]}
        validate_episode(ep, {"E1"})
        scene["media"] = {"start_seconds": 105.0}
        with self.assertRaisesRegex(ValueError, "evidence scene"): validate_episode(ep, {"E1"})
        scene.update(type="PROJECT_EVIDENCE", asset_ref="E1")
        validate_episode(ep, {"E1"})
        scene["notes"] = "Use gameplay around 105 seconds."
        del scene["media"]
        with self.assertRaisesRegex(ValueError, "media.start_seconds"):
            validate_episode(ep, {"E1"})
        scene["media"] = {"start_seconds": 105.0}
        validate_episode(ep, {"E1"})
        scene["media"]["start_seconds"] = -1
        with self.assertRaisesRegex(ValueError, "non-negative"): validate_episode(ep, {"E1"})

    def test_frozen_jsons_unchanged(self):
        for relative, expected in FROZEN_BLOBS.items():
            path = ROOT / relative
            if not path.exists(): continue  # slim local checkout; full repository has all 12
            data = path.read_bytes()
            actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            self.assertEqual(actual, expected, relative)

    def test_outro_metadata_and_layout(self):
        presentation = json.loads((ROOT / "projects/cisco-doom-episode-001/presentation.json").read_text())
        validate_presentation(presentation, True)
        with self.assertRaisesRegex(ValueError, "OUTRO requires"):
            validate_presentation({}, True)
        episode = json.loads((ROOT / "projects/cisco-doom-episode-001/episode.json").read_text())
        ssh = next(scene for scene in episode["scenes"] if scene["id"] == "s005")
        rendered_ssh = _visual_scene(ssh, presentation)
        self.assertEqual(rendered_ssh["title"], "SSH Access - The Turning Point")
        self.assertNotIn("\ufffd", rendered_ssh["title"])
        self.assertEqual(ssh["title"], "SSH Access — The Turning Point")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "outro.png"
            scene = {"id": "outro", "type": "OUTRO"}
            render_scene(scene, None, output, ROOT, {}, presentation=presentation)
            image = Image.open(output)
            self.assertEqual(image.size, (1920, 1080))
            base = Image.new("RGB", image.size, (8, 13, 18))
            for box in ((100,230,1810,450),(100,540,1810,640),(100,690,1810,820)):
                self.assertIsNotNone(ImageChops.difference(image.crop(box),base.crop(box)).getbbox())
            without_titles = Path(tmp) / "endcard-without-titles.png"
            render_scene(scene, None, without_titles, ROOT, {},
                         presentation={"outro": presentation["outro"]})
            self.assertEqual(output.read_bytes(), without_titles.read_bytes())
            changed = {"outro": {**presentation["outro"], "links": [
                {"label": "DOCS", "url": "example.org/new-project"}]}}
            other = Path(tmp) / "other.png"
            render_scene(scene, None, other, ROOT, {}, presentation=changed)
            self.assertNotEqual(output.read_bytes(), other.read_bytes())

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg unavailable")
    def test_ffmpeg_caption_pixels(self):
        with tempfile.TemporaryDirectory(prefix="s2r caption ") as tmp:
            d = Path(tmp)
            frame = d / "frame.png"
            Image.new("RGB", (1920,1080), (8,13,18)).save(frame)
            audio = d / "audio.wav"
            with wave.open(str(audio), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
                w.writeframes(b"\0\0" * 24000)
            ass = d / "caption.ass"
            write_ass(ass, estimate_events("Visible caption proof.", 1.0))
            seg = d / "segment.mp4"
            _encode_static_segment(frame,audio,seg,1.1,0.1,ass)
            shot = d / "shot.png"
            run(["ffmpeg","-y","-loglevel","error","-ss","0.4","-i",seg,"-frames:v","1",shot])
            image = Image.open(shot).convert("RGB")
            region=image.crop((350,800,1570,950))
            pixels=region.get_flattened_data() if hasattr(region,"get_flattened_data") else region.getdata()
            count = sum(1 for r,g,b in pixels if min(r,g,b)>130)
            self.assertGreater(count,500)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg unavailable")
    def test_video_caption_pixels_and_contained_footage(self):
        with tempfile.TemporaryDirectory(prefix="s2r video caption ") as tmp:
            d=Path(tmp)
            source=d/"source.png"
            image=Image.new("RGB",(640,480),(24,30,40)); draw=ImageDraw.Draw(image)
            draw.rectangle((0,0,65,65),fill=(245,30,30))
            draw.rectangle((574,0,639,65),fill=(30,245,30))
            draw.rectangle((0,414,65,479),fill=(30,30,245))
            draw.rectangle((574,414,639,479),fill=(245,245,30))
            image.save(source)
            video=d/"source.mp4"
            run(["ffmpeg","-y","-loglevel","error","-loop","1","-framerate","30",
                 "-i",source,"-t","1","-c:v","libx264","-pix_fmt","yuv420p",video])
            shell=d/"shell.png"
            render_video_shell({"title":"Video proof"},shell,ROOT,{},captions_enabled=True)
            audio=d/"voice.wav"
            with wave.open(str(audio),"wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
                w.writeframes(b"\0\0"*24000)
            ass=d/"caption.ass"
            write_ass(ass,estimate_events("Visible video caption proof.",1.0))
            seg=d/"segment.mp4"
            _encode_video_evidence(shell,video,audio,seg,1.1,0.1,0.0,ass)
            shot=d/"shot.png"
            run(["ffmpeg","-y","-loglevel","error","-ss","0.4","-i",seg,
                 "-frames:v","1",shot])
            result=Image.open(shot).convert("RGB")
            self.assertEqual(result.size,(1920,1080))
            corners=((580,190,(245,30,30)),(1340,190,(30,245,30)),
                     (580,745,(30,30,245)),(1340,745,(245,245,30)))
            for x,y,expected in corners:
                actual=result.getpixel((x,y))
                self.assertTrue(all(abs(a-b)<60 for a,b in zip(actual,expected)),
                                (x,y,actual,expected))
            region=result.crop((350,800,1570,950))
            pixels=region.get_flattened_data() if hasattr(region,"get_flattened_data") else region.getdata()
            self.assertGreater(sum(1 for r,g,b in pixels if min(r,g,b)>130),500)

    def test_ffmpeg_offset_command(self):
        with patch("source2reel.renderer.run") as mock:
            _encode_video_evidence(Path("shell.png"),Path("video.mp4"),Path("audio.wav"),
                                   Path("segment.mp4"),2,0.5,105.0)
        cmd = mock.call_args.args[0]
        self.assertEqual(cmd[cmd.index("-ss")+1],"105.000")

    def test_ffmpeg_caption_switch(self):
        with patch("source2reel.renderer.run") as mock:
            _encode_static_segment(Path("frame.png"),Path("voice.wav"),Path("out.mp4"),2,0.5)
        self.assertNotIn("ass=",mock.call_args.args[0][mock.call_args.args[0].index("-filter_complex")+1])
        with patch("source2reel.renderer.run") as mock:
            _encode_static_segment(Path("frame.png"),Path("voice.wav"),Path("out.mp4"),2,0.5,Path("captions.ass"))
        self.assertIn("ass=",mock.call_args.args[0][mock.call_args.args[0].index("-filter_complex")+1])

if __name__ == "__main__":
    unittest.main()
