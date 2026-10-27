"""Record the captioned video guides and screenshots used in the README.

One continuous browser session is recorded against a running server, then cut into short
clips. Captions are drawn on the page itself, so the videos need no voice and no editing.

Requirements (not part of the package): ``pip install playwright imageio-ffmpeg`` and
``playwright install chromium-headless-shell``. Start the server first, with a real LLM
provider configured, then run::

    python scripts/record_guides.py --base-url http://127.0.0.1:8765

The server makes paid LLM calls during the recording (a few ideas requests and one script).
"""

from __future__ import annotations

import argparse
import contextlib
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import Page, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

VIEWPORT = {"width": 1280, "height": 760}

CAPTION_JS = """
(text) => {
  let box = document.getElementById("guide-caption");
  if (!box) {
    box = document.createElement("div");
    box.id = "guide-caption";
    Object.assign(box.style, {
      position: "fixed", left: "50%", bottom: "28px", transform: "translateX(-50%)",
      maxWidth: "980px", padding: "14px 22px", borderRadius: "12px", zIndex: "99999",
      background: "rgba(20, 24, 31, 0.92)", color: "#fff", font: "600 22px/1.35 Inter, sans-serif",
      textAlign: "center", boxShadow: "0 8px 30px rgba(0,0,0,.25)", pointerEvents: "none",
    });
    document.body.appendChild(box);
  }
  box.textContent = text;
  box.style.display = text ? "block" : "none";
}
"""


@dataclass
class Clip:
    name: str
    title: str
    start: float
    end: float = 0.0


class Recorder:
    def __init__(self, page: Page, shots: Path) -> None:
        self.page = page
        self.shots = shots
        self.t0 = time.monotonic()
        self.clips: list[Clip] = []
        self.current_caption = ""

    def now(self) -> float:
        return time.monotonic() - self.t0

    def clip(self, name: str, title: str) -> None:
        if self.clips:
            self.clips[-1].end = self.now()
        self.clips.append(Clip(name, title, self.now()))

    def caption(self, text: str, hold: float = 2.8) -> None:
        self.current_caption = text
        self.page.evaluate(CAPTION_JS, text)
        self.page.wait_for_timeout(int(hold * 1000))

    def shot(self, name: str) -> None:
        self.page.evaluate(CAPTION_JS, "")
        self.page.screenshot(path=str(self.shots / f"{name}.png"))
        self.page.evaluate(CAPTION_JS, self.current_caption)

    def scroll_to(self, selector: str, hold: float = 1.2) -> None:
        self.page.locator(selector).first.scroll_into_view_if_needed()
        self.page.evaluate("window.scrollBy({top: -90, behavior: 'instant'})")
        self.page.wait_for_timeout(int(hold * 1000))

    def glide(self, pixels: int, steps: int = 12, pause_ms: int = 120) -> None:
        for _ in range(steps):
            self.page.mouse.wheel(0, pixels / steps)
            self.page.wait_for_timeout(pause_ms)

    def finish(self) -> None:
        self.clips[-1].end = self.now()


def get_ideas(rec: Recorder, count: int) -> None:
    page = rec.page
    page.fill("#f-count", str(count))
    page.select_option("#f-format", index=0)
    page.fill("#f-topic", "")
    page.click("#ideas-submit")
    # The summary line is already filled after the first request, so wait for the button:
    # it is disabled while a request runs and enabled again when the new ideas are shown.
    with contextlib.suppress(PlaywrightTimeoutError):  # it may finish before we look
        page.wait_for_function("document.querySelector('#ideas-submit').disabled", timeout=3000)
    page.wait_for_function("!document.querySelector('#ideas-submit').disabled", timeout=120_000)
    page.wait_for_selector("#ideas-summary:not(:empty)", timeout=10_000)
    page.wait_for_timeout(600)


def run_session(base_url: str, video_dir: Path, shots: Path) -> tuple[Path, list[Clip]]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport=VIEWPORT,
            record_video_dir=str(video_dir),
            record_video_size=VIEWPORT,
            bypass_csp=True,  # only so the caption overlay can be styled from the script
            device_scale_factor=1,
        )
        page = context.new_page()
        page.goto(base_url, wait_until="networkidle")
        rec = Recorder(page, shots)

        rec.clip("01-get-ideas", "Get ideas from your own lessons")
        rec.caption("Ask for ideas: how many, which format, and an optional topic.")
        get_ideas(rec, 15)
        rec.caption("Every idea comes from your lessons and names the course, lesson and section.")
        rec.shot("ideas")
        rec.glide(1400, steps=20)
        rec.caption("Ideas are spread across lessons: at most two from any one lesson.", hold=2.4)
        page.evaluate("window.scrollTo({top: 0, behavior: 'instant'})")

        rec.clip("02-check-the-source", "Check where an idea came from")
        rec.scroll_to("#ideas-list .card .source-chip")
        rec.caption("Open the source to read the lesson passage behind the idea.")
        page.locator("#ideas-list .card .source-chip").first.click()
        page.wait_for_timeout(500)
        rec.caption("The quote is highlighted. Code checks that it appears word for word.")
        rec.shot("source")

        rec.clip("03-mark-used", "Mark ideas you'll use, never get them again")
        cards = page.locator("#ideas-list .card")
        for i in range(2):
            rec.scroll_to(f"#ideas-list .card >> nth={i}", hold=0.6)
            cards.nth(i).get_by_role("button", name="I'll use this").click()
            page.wait_for_timeout(700)
        rec.caption("Mark the ideas you'll use. They go into your history.")
        page.evaluate("window.scrollTo({top: 0, behavior: 'instant'})")
        rec.caption("Ask again: passages you used in the last 30 days are skipped.", hold=2.2)
        get_ideas(rec, 5)
        rec.caption("Fresh ideas only. No repeats.")

        rec.clip("04-write-script", "Turn an idea into a script in your voice")
        rec.scroll_to("#ideas-list .card")
        rec.caption("Pick an idea and write the script in your own voice.")
        first = page.locator("#ideas-list .card").first
        first.get_by_role("button", name="Write script").click()
        page.wait_for_selector("#ideas-list .card .script-box", timeout=120_000)
        page.wait_for_timeout(600)
        rec.scroll_to("#ideas-list .card .script-box")
        rec.caption(
            "It follows your style sheet and past posts. Numbers not in the lesson get flagged."
        )
        rec.shot("script")
        rec.glide(500, steps=8)
        rec.caption("Copy it and record. That's it.", hold=2.0)

        rec.clip("05-coverage", "See which lessons you haven't used yet")
        page.evaluate("window.scrollTo({top: 0, behavior: 'instant'})")
        page.click("#tab-coverage")
        page.wait_for_selector("#coverage-list :first-child", timeout=30_000)
        rec.caption("Library coverage: which lessons you've used, and which you never have.")
        rec.shot("coverage")
        rec.glide(700, steps=10)

        rec.clip("06-voice", "Your voice, written down")
        page.evaluate("window.scrollTo({top: 0, behavior: 'instant'})")
        page.click("#tab-voice")
        page.wait_for_selector("#voice-body :first-child", timeout=30_000)
        rec.caption("Your voice: phrases you use, words you never say, and your past posts.")
        rec.shot("voice")
        rec.glide(900, steps=12)
        rec.caption("Change the style sheet any time. The next script follows it.", hold=2.4)
        rec.finish()

        video = page.video
        context.close()
        browser.close()
        if video is None:
            raise RuntimeError("Playwright did not record a video")
        return Path(video.path()), rec.clips


def cut(source: Path, clips: list[Clip], out_dir: Path) -> list[Path]:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for clip in clips:
        target = out_dir / f"{clip.name}.mp4"
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{clip.start:.2f}",
            "-to",
            f"{clip.end:.2f}",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "26",
            "-preset",
            "slow",
            "-movflags",
            "+faststart",
            "-an",
            str(target),
        ]
        subprocess.run(command, check=True)  # noqa: S603 - fixed arguments, no shell
        written.append(target)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--videos", type=Path, default=Path("assets/videos"))
    parser.add_argument("--screenshots", type=Path, default=Path("assets/screenshots"))
    args = parser.parse_args()
    args.screenshots.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        raw, clips = run_session(args.base_url, Path(tmp), args.screenshots)
        written = cut(raw, clips, args.videos)
    for clip, path in zip(clips, written, strict=True):
        print(f"{path}  {clip.end - clip.start:5.1f} s  {clip.title}")


if __name__ == "__main__":
    main()
