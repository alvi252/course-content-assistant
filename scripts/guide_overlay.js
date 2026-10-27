// Overlay for the recorded video guides, injected by scripts/record_guides.py (not part of the app).
// A spotlight frames the block being described and dims the rest, the caption sits next to that
// block and never on the highlighted part, and a drawn cursor glides to each button before a click.
(() => {
  if (window.guide) return;

  const PAD = 8; // spotlight padding around the element
  const GAP = 16; // space between the spotlight and the caption
  const EDGE = 16; // closest the caption gets to the viewport edge

  const style = document.createElement("style");
  style.textContent = `
    #g-spot {
      position: fixed; left: 0; top: 0; border-radius: 14px; pointer-events: none; z-index: 99990;
      box-shadow: 0 0 0 2.5px #b83f0e, 0 0 0 9px rgba(184, 63, 14, 0.16),
        0 0 0 200vmax rgba(28, 20, 14, 0.42);
      opacity: 0; transition: opacity 0.35s ease;
    }
    #g-cap {
      position: fixed; left: 0; top: 0; max-width: 460px; padding: 13px 18px 14px;
      border-radius: 12px; z-index: 99992; pointer-events: none;
      background: #211a15; color: #fff; font: 600 18px/1.42 Inter, system-ui, sans-serif;
      box-shadow: 0 14px 34px rgba(0, 0, 0, 0.3);
      opacity: 0; transform: translateY(6px); transition: opacity 0.28s ease, transform 0.28s ease;
    }
    #g-cap.on { opacity: 1; transform: none; }
    #g-cap::before {
      content: ""; display: block; width: 28px; height: 3px; border-radius: 2px;
      background: #f08a5d; margin-bottom: 8px;
    }
    #g-cursor {
      position: fixed; left: 0; top: 0; width: 24px; height: 24px; z-index: 99995;
      pointer-events: none; filter: drop-shadow(0 2px 3px rgba(0, 0, 0, 0.35));
      background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M3 2l7.5 19 2.6-7.9L21 10.5z' fill='%23211a15' stroke='white' stroke-width='1.6' stroke-linejoin='round'/%3E%3C/svg%3E") no-repeat;
      transition: transform 0.12s ease;
    }
    #g-cursor.down { transform: scale(0.82); }
    .g-ripple {
      position: fixed; width: 18px; height: 18px; margin: -9px 0 0 -9px; border-radius: 50%;
      border: 3px solid #b83f0e; z-index: 99994; pointer-events: none;
      animation: g-ripple 0.6s ease-out forwards;
    }
    @keyframes g-ripple { from { transform: scale(0.4); opacity: 1; } to { transform: scale(3); opacity: 0; } }
  `;
  document.head.appendChild(style);

  const make = (id) => {
    const node = document.createElement("div");
    node.id = id;
    document.body.appendChild(node);
    return node;
  };
  const spot = make("g-spot");
  const cap = make("g-cap");
  const cursor = make("g-cursor");

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const wait = (ms) => new Promise((done) => setTimeout(done, ms));
  const ease = (p) => (p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2);

  let target = null; // element in the spotlight; its box is read every frame, so scrolling is followed
  let spotLimit = null; // lowest y the spotlight may reach when the caption has to sit below it
  const rest = { cx: innerWidth * 0.66, cy: innerHeight * 0.72 }; // where the cursor waits at first
  const now = { x: 0, y: 0, w: 0, h: 0, ...rest };
  const goal = { x: 0, y: 0, w: 0, h: 0, ...rest };
  let started = false;
  let last = performance.now();

  function frame(t) {
    const dt = Math.min(64, t - last);
    last = t;
    // The app re-renders a card's contents on every click, so a removed element is skipped.
    if (target && target.isConnected) {
      // The frame moves with the element, also while it is off screen during a scroll.
      const r = target.getBoundingClientRect();
      const top = r.top - PAD;
      let bottom = r.bottom + PAD;
      if (spotLimit !== null) bottom = Math.max(top + 24, Math.min(bottom, spotLimit));
      Object.assign(goal, { x: r.left - PAD, y: top, w: r.width + 2 * PAD, h: bottom - top });
      if (!started) {
        Object.assign(now, { x: goal.x, y: goal.y, w: goal.w, h: goal.h });
        started = true;
      }
    }
    const a = 1 - Math.exp(-dt / 90); // spotlight: quick ease-out
    const b = 1 - Math.exp(-dt / 140); // cursor: a little slower, like a hand
    for (const k of ["x", "y", "w", "h"]) now[k] += (goal[k] - now[k]) * a;
    now.cx += (goal.cx - now.cx) * b;
    now.cy += (goal.cy - now.cy) * b;
    Object.assign(spot.style, {
      transform: `translate(${now.x}px, ${now.y}px)`,
      width: `${now.w}px`,
      height: `${now.h}px`,
    });
    cursor.style.left = `${now.cx - 3}px`;
    cursor.style.top = `${now.cy - 2}px`;
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  function scrollToY(y) {
    const y0 = window.scrollY;
    const end = clamp(y, 0, document.documentElement.scrollHeight - innerHeight);
    const dist = end - y0;
    if (Math.abs(dist) < 2) return Promise.resolve();
    const ms = clamp(350 + Math.abs(dist) * 0.45, 450, 1300);
    const t0 = performance.now();
    return new Promise((done) => {
      const step = (t) => {
        const p = Math.min(1, (t - t0) / ms);
        window.scrollTo({ top: y0 + dist * ease(p), behavior: "instant" });
        if (p < 1) requestAnimationFrame(step);
        else done();
      };
      requestAnimationFrame(step);
    });
  }

  async function hideCaption() {
    if (cap.classList.contains("on")) {
      cap.classList.remove("on");
      await wait(220);
    }
  }

  function placeCaption(rect) {
    // Below the block if it fits, else above it; if neither fits, at the bottom of the screen
    // with the spotlight stopping above the caption, so the text never covers what it describes.
    const w = cap.offsetWidth;
    const h = cap.offsetHeight;
    let left;
    let top;
    spotLimit = null;
    if (!rect) {
      left = (innerWidth - w) / 2;
      top = innerHeight - h - 28;
    } else {
      left = clamp(rect.left, EDGE, innerWidth - w - EDGE);
      if (rect.bottom + GAP + h <= innerHeight - EDGE) top = rect.bottom + GAP;
      else if (rect.top - GAP - h >= EDGE) top = rect.top - GAP - h;
      else {
        top = innerHeight - h - EDGE;
        spotLimit = top - GAP;
      }
    }
    cap.style.left = `${left}px`;
    cap.style.top = `${top}px`;
  }

  async function show(text, rect) {
    cap.textContent = text;
    placeCaption(rect);
    cap.classList.add("on");
  }

  window.guide = {
    // Spotlight on `el`, scroll it (and room for the caption) into view, then show the caption.
    async focus(el, text) {
      await hideCaption();
      target = el;
      started = started && spot.style.opacity === "1";
      spot.style.opacity = "1";
      cap.textContent = text;
      const capH = cap.offsetHeight;
      const r = el.getBoundingClientRect();
      const need = r.height + 2 * PAD + GAP + capH;
      const offset = need <= innerHeight - 2 * EDGE ? (innerHeight - need) / 2 : 64;
      await scrollToY(window.scrollY + r.top - PAD - offset);
      await wait(320); // let the frame settle before the caption appears
      const box = el.getBoundingClientRect();
      await show(text, {
        left: box.left - PAD,
        top: Math.max(box.top - PAD, 6),
        bottom: Math.min(box.bottom + PAD, innerHeight - 6),
      });
    },
    // Caption alone, centred at the bottom; the spotlight fades out.
    async say(text) {
      await hideCaption();
      target = null;
      spot.style.opacity = "0";
      started = false;
      if (text) await show(text, null);
    },
    // Scroll smoothly only if `el` is not comfortably on screen (used before pointing at it).
    async reveal(el) {
      const r = el.getBoundingClientRect();
      if (r.top < 60 || r.bottom > innerHeight - 60) {
        await scrollToY(window.scrollY + r.top - (innerHeight - r.height) / 2);
        await wait(150);
      }
    },
    async scrollBy(px) {
      await scrollToY(window.scrollY + px);
    },
    async scrollTop() {
      await scrollToY(0);
    },
    point(x, y) {
      goal.cx = x;
      goal.cy = y;
    },
    // After a click the cursor moves aside, so it does not sit on the content.
    park() {
      goal.cx = innerWidth - 56;
      goal.cy = clamp(goal.cy, 80, innerHeight - 80);
    },
    async tap() {
      cursor.classList.add("down");
      const ring = document.createElement("div");
      ring.className = "g-ripple";
      ring.style.left = `${now.cx}px`;
      ring.style.top = `${now.cy}px`;
      document.body.appendChild(ring);
      setTimeout(() => ring.remove(), 700);
      await wait(130);
      cursor.classList.remove("down");
    },
    // For clean screenshots.
    setVisible(on) {
      for (const node of [spot, cap, cursor]) node.style.visibility = on ? "" : "hidden";
    },
  };
})();
