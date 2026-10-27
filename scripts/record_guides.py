"""Record the captioned video guides and screenshots used in the README.

One continuous browser session is recorded against a running server, then cut into short
clips. The overlay in ``guide_overlay.js`` frames the block being described, places the caption
next to it and moves a drawn cursor to each button, so the videos need no voice and no editing.
Waits for the model are cut out of the clips, and the caption says so.

Requirements (not part of the package): ``pip install playwright imageio-ffmpeg`` and
``playwright install chromium-headless-shell``. Start the server first, with a real LLM
provider configured and an empty ideas log, then run::

    python scripts/record_guides.py --base-url http://127.0.0.1:8765

The server makes paid LLM calls during the recording (two ideas requests and one script).
"""

from __future__ import annotations

import argparse
import contextlib
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.sync_api import Locator, Page

VIEWPORT = {"width": 1120, "height": 720}
OVERLAY_JS = Path(__file__).with_name("guide_overlay.js")
ANIMATION_FPS = 15


@dataclass
class Clip:
    name: str
    title: str
    start: float
    end: float = 0.0
    skips: list[tuple[float, float]] = field(default_factory=list)


def segments(clip: Clip) -> list[tuple[float, float]]:
    """The parts of a clip that stay in the video, in order.

    >>> segments(Clip("a", "t", 0.0, 10.0, [(2.0, 5.0), (7.0, 8.0)]))
    [(0.0, 2.0), (5.0, 7.0), (8.0, 10.0)]
    >>> segments(Clip("a", "t", 3.0, 6.0))
    [(3.0, 6.0)]
    """
    parts = []
    start = clip.start
    for skip_start, skip_end in sorted(clip.skips):
        if skip_start > start:
            parts.append((start, skip_start))
        start = max(start, skip_end)
    if clip.end > start:
        parts.append((start, clip.end))
    return parts


def trim_filter(parts: list[tuple[float, float]]) -> str:
    """An ffmpeg filter that keeps only ``parts`` of the input and joins them.

    >>> trim_filter([(5.0, 7.5)])
    '[0:v]trim=start=5.00:end=7.50,setpts=PTS-STARTPTS[p0];[p0]concat=n=1:v=1:a=0[out]'
    >>> trim_filter([(0.0, 2.0), (5.0, 7.5)]).split(";")[1:]
    ['[0:v]trim=start=5.00:end=7.50,setpts=PTS-STARTPTS[p1]', '[p0][p1]concat=n=2:v=1:a=0[out]']
    """
    chains = [
        f"[0:v]trim=start={a:.2f}:end={b:.2f},setpts=PTS-STARTPTS[p{i}]"
        for i, (a, b) in enumerate(parts)
    ]
    joined = "".join(f"[p{i}]" for i in range(len(parts)))
    return ";".join([*chains, f"{joined}concat=n={len(parts)}:v=1:a=0[out]"])


def hold_for(text: str) -> float:
    """Seconds a caption stays on screen: enough to read it at a relaxed pace.

    >>> hold_for("Short.")
    2.4
    >>> hold_for("Open the source to read the lesson passage behind the idea.")
    4.6
    """
    return round(max(2.4, 1.3 + 0.3 * len(text.split())), 1)


class Recorder:
    def __init__(self, page: Page, shots: Path) -> None:
        self.page = page
        self.shots = shots
        self.t0 = time.monotonic()
        self.clips: list[Clip] = []

    def now(self) -> float:
        return time.monotonic() - self.t0

    def pause(self, seconds: float) -> None:
        self.page.wait_for_timeout(int(seconds * 1000))

    def clip(self, name: str, title: str) -> None:
        if self.clips:
            self.say("")  # the last caption of a clip must not carry over into the next one
            self.pause(0.3)
            self.clips[-1].end = self.now()
        self.clips.append(Clip(name, title, self.now()))

    def install_overlay(self) -> None:
        self.page.add_script_tag(content=OVERLAY_JS.read_text(encoding="utf-8"))

    def focus(self, target: Locator, text: str, hold: float | None = None) -> None:
        """Spotlight on ``target``, caption next to it, then hold long enough to read."""
        target.evaluate("(el, text) => guide.focus(el, text)", text)
        self.pause(hold if hold is not None else hold_for(text))

    def say(self, text: str, hold: float | None = None) -> None:
        """Caption alone at the bottom, no spotlight."""
        self.page.evaluate("(text) => guide.say(text)", text)
        if text:
            self.pause(hold if hold is not None else hold_for(text))

    def point_at(self, target: Locator) -> None:
        target.evaluate("(el) => guide.reveal(el)")
        box = target.bounding_box()
        if box is None:
            raise RuntimeError("element to point at is not visible")
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        self.page.evaluate("([x, y]) => guide.point(x, y)", [x, y])
        self.pause(0.8)

    def press(self, target: Locator) -> None:
        self.point_at(target)
        self.page.evaluate("() => guide.tap()")
        target.click()

    def step_aside(self) -> None:
        # The real mouse goes to the empty margin so nothing stays in hover state after a
        # scroll, and the drawn cursor moves off the content.
        self.page.mouse.move(VIEWPORT["width"] - 8, VIEWPORT["height"] - 8)
        self.page.evaluate("() => guide.park()")

    def click(self, target: Locator) -> None:
        self.press(target)
        self.step_aside()

    def type_into(self, target: Locator, text: str) -> None:
        self.press(target)
        target.press("Control+A")
        target.press_sequentially(text, delay=110)
        self.pause(0.4)

    def scroll_by(self, pixels: int) -> None:
        self.say("")
        self.page.evaluate("(px) => guide.scrollBy(px)", pixels)

    def scroll_top(self) -> None:
        self.say("")
        self.page.evaluate("() => guide.scrollTop()")

    def shot(self, name: str) -> None:
        self.page.evaluate("() => guide.setVisible(false)")
        self.page.screenshot(path=str(self.shots / f"{name}.png"))
        self.page.evaluate("() => guide.setVisible(true)")

    def wait_for_model(self, ready: str, keep: float, timeout_ms: int = 120_000) -> None:
        """Wait for a model call to finish; only the first ``keep`` seconds stay in the clip."""
        start = self.now() + keep
        self.page.wait_for_function(ready, timeout=timeout_ms)
        # The caption and spotlight fade out inside the cut, so the clip jumps straight from
        # the waiting screen to the clean result.
        self.say("")
        self.pause(0.4)
        end = self.now()
        if end > start:
            self.clips[-1].skips.append((start, end))

    def finish(self) -> None:
        self.clips[-1].end = self.now()


def get_ideas(rec: Recorder, count: int, caption: str) -> None:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    page = rec.page
    rec.type_into(page.locator("#f-count"), str(count))
    rec.click(page.locator("#ideas-submit"))
    # The summary line is already filled after the first request, so wait for the button:
    # it is disabled while a request runs and enabled again when the new ideas are shown.
    with contextlib.suppress(PlaywrightTimeoutError):  # it may finish before we look
        page.wait_for_function("document.querySelector('#ideas-submit').disabled", timeout=3000)
    rec.focus(page.locator(".form-actions"), caption, hold=0)
    rec.wait_for_model("!document.querySelector('#ideas-submit').disabled", keep=hold_for(caption))
    page.wait_for_selector("#ideas-summary:not(:empty)", timeout=10_000)


def record(rec: Recorder) -> None:
    page = rec.page
    cards = page.locator("#ideas-list .card")
    first = cards.first

    rec.clip("01-get-ideas", "Get ideas from your own lessons")
    form = page.locator("#ideas-form")
    rec.focus(form, "Ask for ideas: how many, which format, and an optional topic.")
    get_ideas(rec, 15, "Reading the lessons and writing ideas. The wait is cut from this video.")
    summary = page.locator("#ideas-summary")
    rec.focus(summary, "How many ideas came back, and from how many lessons.")
    rec.focus(first, "Each idea has a format, a title, a first line and a short plan.")
    rec.shot("ideas")
    rec.focus(first.locator(".source"), "And the lesson it came from, with the exact quote.")
    rec.scroll_by(900)
    rec.say("At most two ideas from any one lesson, so they are spread across the library.")

    rec.clip("02-check-the-source", "Check where an idea came from")
    rec.focus(first.locator(".source-chip"), "Open the source to see the lesson passage.")
    rec.click(first.locator(".source-chip"))
    rec.pause(0.4)
    rec.focus(first.locator(".source-body"), "The quote is highlighted inside the lesson.")
    rec.shot("source")
    rec.focus(
        first.locator(".source-body"),
        "Code checks that the quote appears word for word. Ideas that fail are dropped.",
    )
    rec.click(first.locator(".source-chip"))  # close it again for the next clip

    rec.clip("03-mark-used", "Mark ideas you'll use, never get them again")
    for i in range(2):
        card = cards.nth(i)
        rec.focus(card, "Mark the ideas you'll use." if i == 0 else "One more.", hold=1.6)
        rec.click(card.get_by_role("button", name="I'll use this"))
        rec.pause(0.6)
    rec.focus(cards.nth(1), "Used ideas go into your history.")
    rec.scroll_top()
    rec.focus(form, "Ask again. What you used in the last 30 days is skipped.")
    get_ideas(rec, 5, "Asking again. The wait is cut from this video.")
    rec.focus(summary, "None of the passages you used comes back.")
    toggle = page.locator("#history-toggle")
    rec.focus(toggle, "Everything you used is kept here, with the date.", hold=1.8)
    rec.click(toggle)
    page.wait_for_selector("#history-list .card", timeout=30_000)
    rec.focus(page.locator("#history-list"), "Your history: the ideas you used and when.")
    rec.click(toggle)

    rec.clip("04-write-script", "Turn an idea into a script in your voice")
    rec.focus(first, "Pick an idea and write the script.", hold=2.0)
    rec.click(first.get_by_role("button", name="Write script"))
    writing = "Writing in her voice. The wait is cut from this video."
    rec.focus(first.locator(".card-actions"), writing, hold=0)
    rec.wait_for_model("!!document.querySelector('#ideas-list .script-box')", hold_for(writing))
    script = first.locator(".script-box")
    rec.focus(script, "The script follows her style sheet and past posts.")
    rec.shot("script")
    rec.focus(script, "If it mentions a number that is not in the lesson, you get a warning.")
    rec.focus(script.locator(".script-actions"), "Copy it and record.", hold=2.2)

    rec.clip("05-coverage", "See which lessons you haven't used yet")
    rec.scroll_top()
    rec.click(page.locator("#tab-coverage"))
    page.wait_for_selector("#coverage-list .lesson", timeout=30_000)
    rec.pause(0.4)
    rec.focus(page.locator("#coverage-totals"), "How much of your library you have used so far.")
    rec.shot("coverage")
    rec.focus(
        page.locator("#coverage-list .lesson.never").first,
        "Lessons you have never used come first. That's where fresh ideas are.",
    )
    rec.focus(
        page.locator("#coverage-list .lesson:not(.never)").first,
        "Used lessons show how often and when.",
    )

    rec.clip("06-voice", "Your voice, written down")
    rec.scroll_top()
    rec.click(page.locator("#tab-voice"))
    page.wait_for_selector("#voice-body .voice-section", timeout=30_000)
    rec.pause(0.4)
    sections = page.locator("#voice-body .voice-section")
    rec.shot("voice")
    rec.focus(sections.nth(1), "Phrases she uses.", hold=2.2)
    rec.focus(sections.nth(2), "And words she never says.", hold=2.2)
    rec.focus(
        page.locator("#voice-body .voice-section:has(.card)").locator(".card").first,
        "Her past posts. Scripts are written to sound like them.",
    )
    rec.say("Change the style sheet any time. The next script follows it.")
    rec.finish()


def run_session(base_url: str, video_dir: Path, shots: Path) -> tuple[Path, list[Clip]]:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport=VIEWPORT,
            record_video_dir=str(video_dir),
            record_video_size=VIEWPORT,
            bypass_csp=True,  # only so the overlay can be injected from the script
            device_scale_factor=1,
        )
        page = context.new_page()
        rec = Recorder(page, shots)  # the video starts with the page, so time starts here
        page.goto(base_url, wait_until="networkidle")
        rec.install_overlay()
        record(rec)

        video = page.video
        context.close()
        browser.close()
        if video is None:
            raise RuntimeError("Playwright did not record a video")
        return Path(video.path()), rec.clips


def cut(source: Path, clips: list[Clip], out_dir: Path) -> list[Path]:
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for clip in clips:
        target = out_dir / f"{clip.name}.mp4"
        command = [ffmpeg, "-y", "-loglevel", "error", "-i", str(source)]
        command += ["-filter_complex", trim_filter(segments(clip)), "-map", "[out]"]
        command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-preset", "slow"]
        command += ["-movflags", "+faststart", "-an", str(target)]
        subprocess.run(command, check=True)  # noqa: S603 - fixed arguments, no shell
        written.append(target)
    return written


# GitHub does not play mp4 files stored in a repository, but it shows animated images inline.
# Animated WebP keeps the motion smooth at a fraction of the size of a GIF.
def to_animations(videos: list[Path], out_dir: Path) -> None:
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out_dir.mkdir(parents=True, exist_ok=True)
    for video in videos:
        target = out_dir / f"{video.stem}.webp"
        command = [ffmpeg, "-y", "-loglevel", "error", "-i", str(video)]
        command += ["-vf", f"fps={ANIMATION_FPS}", "-c:v", "libwebp_anim", "-quality", "80"]
        command += ["-compression_level", "3", "-loop", "0", str(target)]
        subprocess.run(command, check=True)  # noqa: S603 - fixed arguments, no shell


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--videos", type=Path, default=Path("assets/videos"))
    parser.add_argument("--animations", type=Path, default=Path("assets/animations"))
    parser.add_argument("--screenshots", type=Path, default=Path("assets/screenshots"))
    args = parser.parse_args()
    args.screenshots.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        raw, clips = run_session(args.base_url, Path(tmp), args.screenshots)
        written = cut(raw, clips, args.videos)
    to_animations(written, args.animations)
    for clip, path in zip(clips, written, strict=True):
        kept = sum(b - a for a, b in segments(clip))
        print(f"{path}  {kept:5.1f} s  {clip.title}")


if __name__ == "__main__":
    main()
