"use strict";

(() => {
  const $ = (id) => document.getElementById(id);

  const FRIENDLY_ERROR = "The AI provider didn't answer. Please try again in a moment.";
  const DROP_REASONS = {
    unknown_passage: "pointed to a passage that wasn't given",
    duplicate_passage: "same passage twice",
    wrong_format: "wrong format",
    quote_too_short: "quote too short to check",
    quote_not_in_source: "quote not found in the lesson",
    malformed_idea: "came back incomplete",
  };

  // ---------- helpers ----------

  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  };

  const button = (label, className, onClick) => {
    const b = el("button", className, label);
    b.type = "button";
    if (onClick) b.addEventListener("click", onClick);
    return b;
  };

  const clear = (node) => { while (node.firstChild) node.removeChild(node.firstChild); };
  const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;

  const formatSeconds = (ms) => `${(ms / 1000).toFixed(1)} s`;
  const formatCost = (usd) => `$${usd < 0.001 ? usd.toFixed(4) : usd.toFixed(3)}`;

  // Timestamps without a zone are treated as UTC.
  function parseDate(value) {
    if (!value) return null;
    const hasZone = /(Z|[+-]\d\d:?\d\d)$/.test(value);
    const date = new Date(hasZone ? value : `${value.replace(" ", "T")}Z`);
    return Number.isNaN(date.getTime()) ? null : date;
  }

  function relativeDate(value) {
    const date = parseDate(value);
    if (!date) return "never";
    const days = Math.floor((Date.now() - date.getTime()) / 86400000);
    if (days <= 0) return "today";
    if (days === 1) return "yesterday";
    if (days < 30) return `${days} days ago`;
    if (days < 60) return "about a month ago";
    if (days < 365) return `${Math.round(days / 30)} months ago`;
    return date.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" });
  }

  const absoluteDate = (value) => {
    const date = parseDate(value);
    return date
      ? date.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" })
      : "";
  };

  class ApiError extends Error {}

  function validationMessage(detail) {
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      const parts = detail.map((d) => {
        const field = Array.isArray(d.loc) ? d.loc.filter((p) => p !== "body").join(" ") : "";
        return field ? `${field}: ${d.msg}` : String(d.msg || "");
      }).filter(Boolean);
      if (parts.length) return parts.join("; ");
    }
    return "Please check the values you entered.";
  }

  async function api(path, options) {
    let response;
    try {
      response = await fetch(path, options);
    } catch {
      throw new ApiError(FRIENDLY_ERROR);
    }
    const data = await response.json().catch(() => null);
    if (response.status === 422) {
      throw new ApiError(validationMessage(data && data.detail));
    }
    if (!response.ok) {
      const detail = data && typeof data.detail === "string" ? data.detail : null;
      // 503 (no API key) and 409 (index needs a rebuild) carry a plain-text explanation.
      if ((response.status === 503 || response.status === 409) && detail) throw new ApiError(detail);
      if (response.status >= 500) throw new ApiError(FRIENDLY_ERROR);
      throw new ApiError(detail || `Something went wrong (error ${response.status}).`);
    }
    if (data === null) throw new ApiError(FRIENDLY_ERROR);
    return data;
  }

  const post = (path, body) => api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });

  function showError(box, message) {
    box.textContent = message || "";
    box.hidden = !message;
  }

  // ---------- quote highlighting ----------

  // Lowercase, straighten quotes and dashes, collapse whitespace; keep a map back to the original.
  function normalizeWithMap(text) {
    let out = "";
    const map = [];
    let lastSpace = true;
    for (let i = 0; i < text.length; i += 1) {
      let ch = text[i];
      if (/\s/.test(ch)) {
        if (!lastSpace) { out += " "; map.push(i); lastSpace = true; }
        continue;
      }
      lastSpace = false;
      if (/[‘’‛′]/.test(ch)) ch = "'";
      else if (/[“”‟″]/.test(ch)) ch = '"';
      else if (/[–—−]/.test(ch)) ch = "-";
      const lower = ch.toLowerCase();
      // Only keep 1:1 mappings so the index map stays valid.
      ch = lower.length === 1 ? lower : ch;
      out += ch;
      map.push(i);
    }
    return { text: out, map };
  }

  function findQuote(text, quote) {
    const source = normalizeWithMap(text);
    const needle = normalizeWithMap(quote).text.trim().replace(/^["']+|["']+$/g, "").trim();
    if (!needle) return null;
    // Match only at word boundaries, like the server: no starting or ending mid-word.
    const isWord = (ch) => Boolean(ch) && /[\p{L}\p{N}_]/u.test(ch);
    const needsStart = isWord(needle[0]);
    const needsEnd = isWord(needle[needle.length - 1]);
    // An apostrophe inside a word counts as part of it: "can" must not match "can't".
    const wordAt = (i) => isWord(source.text[i]);
    const glued = (i, step) => wordAt(i) || (source.text[i] === "'" && wordAt(i + step));
    let at = source.text.indexOf(needle);
    while (at >= 0) {
      const end = at + needle.length;
      if (!(needsStart && glued(at - 1, -1)) && !(needsEnd && glued(end, 1))) {
        return { start: source.map[at], end: source.map[at + needle.length - 1] + 1 };
      }
      at = source.text.indexOf(needle, at + 1);
    }
    return null;
  }

  function appendHighlighted(parent, text, quote) {
    const hit = findQuote(text, quote);
    if (!hit) return false;
    parent.appendChild(document.createTextNode(text.slice(0, hit.start)));
    parent.appendChild(el("mark", null, text.slice(hit.start, hit.end)));
    parent.appendChild(document.createTextNode(text.slice(hit.end)));
    return true;
  }

  // ---------- clipboard ----------

  async function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      try { await navigator.clipboard.writeText(text); return true; } catch { /* fall through */ }
    }
    const area = el("textarea", "offscreen");
    area.value = text;
    area.setAttribute("readonly", "");
    area.setAttribute("aria-hidden", "true");
    document.body.appendChild(area);
    area.select();
    let ok = false;
    try { ok = document.execCommand("copy"); } catch { ok = false; }
    document.body.removeChild(area);
    return ok;
  }

  function copyButton(label, getText) {
    const b = button(label, "btn");
    let timer = null;
    b.addEventListener("click", async () => {
      const ok = await copyText(getText());
      b.textContent = ok ? "Copied" : "Copy failed";
      clearTimeout(timer);
      timer = setTimeout(() => { b.textContent = label; }, 2000);
    });
    return b;
  }

  // ---------- idea cards ----------

  // state: { idea, warnings, sourceOpen, busy, error, note }
  function createCard(idea, onUpdate) {
    const state = { idea, warnings: [], sourceOpen: false, busy: null, error: "", node: null };
    state.node = el("article", "card");
    state.update = (next) => {
      if (next) state.idea = next;
      render();
      if (onUpdate) onUpdate(state);
    };

    async function act(kind, path, busyText) {
      if (state.busy) return;
      state.busy = busyText;
      state.error = "";
      render();
      try {
        if (kind === "script") {
          const result = await post(path);
          state.warnings = Array.isArray(result.warnings) ? result.warnings : [];
          state.idea = result.idea || state.idea;
          if (!state.idea.script) state.idea = { ...state.idea, script: result.script, caption: result.caption };
        } else {
          state.idea = await post(path);
        }
      } catch (error) {
        state.error = error.message;
      }
      state.busy = null;
      state.update();
    }

    function render() {
      const { idea: it } = state;
      const node = state.node;
      clear(node);
      node.className = `card${it.status === "used" ? " is-used" : ""}${it.status === "dismissed" ? " is-dismissed" : ""}`;

      if (it.status === "dismissed") {
        node.appendChild(el("span", "dismissed-text", `Dismissed: ${it.title}`));
        const undo = button(state.busy ? "Working…" : "Undo", "btn btn-quiet",
          () => act("reset", `/ideas/${it.id}/reset`, "Working…"));
        undo.disabled = Boolean(state.busy);
        node.appendChild(undo);
        if (state.error) node.appendChild(el("span", "card-error", state.error));
        return;
      }

      const top = el("div", "card-top");
      top.appendChild(el("span", "badge", it.format_label || it.format));
      if (it.status === "used") top.appendChild(el("span", "badge badge-used", "Used"));
      node.appendChild(top);

      node.appendChild(el("h3", "card-title", it.title));
      if (it.hook) node.appendChild(el("p", "hook", it.hook));
      if (Array.isArray(it.outline) && it.outline.length) {
        const list = el("ul", "outline");
        it.outline.forEach((line) => list.appendChild(el("li", null, line)));
        node.appendChild(list);
      }

      if (Array.isArray(it.warnings) && it.warnings.length) {
        const list = el("ul", "warnings");
        list.setAttribute("aria-label", "Check before you use this idea");
        it.warnings.forEach((w) => list.appendChild(el("li", null, `Check: ${w}`)));
        node.appendChild(list);
      }
      node.appendChild(renderSource(it));
      if (it.script) node.appendChild(renderScript(it));

      const actions = el("div", "card-actions");
      const disabled = Boolean(state.busy);
      if (it.status === "used") {
        actions.appendChild(button("Undo", "btn", () => act("reset", `/ideas/${it.id}/reset`, "Undoing…")));
      } else {
        actions.appendChild(button("I'll use this", "btn btn-primary", () => act("use", `/ideas/${it.id}/use`, "Saving…")));
        actions.appendChild(button("Not for me", "btn", () => act("dismiss", `/ideas/${it.id}/dismiss`, "Saving…")));
      }
      actions.appendChild(button(it.script ? "Rewrite script" : "Write script", "btn",
        () => act("script", `/ideas/${it.id}/script`, "Writing the script in your voice…")));
      actions.querySelectorAll("button").forEach((b) => { b.disabled = disabled; });
      node.appendChild(actions);

      if (it.status === "used" && it.used_at) {
        node.appendChild(el("p", "card-meta used-at", `Used ${absoluteDate(it.used_at)}`));
      }
      if (state.busy) {
        const status = el("p", "card-status", state.busy);
        status.setAttribute("role", "status");
        node.appendChild(status);
      }
      if (state.error) {
        const err = el("p", "card-error", state.error);
        err.setAttribute("role", "alert");
        node.appendChild(err);
      }
    }

    function renderSource(it) {
      const src = it.source || {};
      const wrap = el("div", "source");
      const bodyId = `source-${it.id}`;
      const chip = button("", "source-chip", () => {
        state.sourceOpen = !state.sourceOpen;
        render();
        const again = state.node.querySelector(".source-chip");
        if (again) again.focus();
      });
      chip.setAttribute("aria-expanded", String(state.sourceOpen));
      chip.setAttribute("aria-controls", bodyId);
      chip.appendChild(el("span", "caret"));
      chip.lastChild.setAttribute("aria-hidden", "true");
      const path = el("span", "path");
      path.appendChild(el("b", null, src.course || "Course"));
      path.appendChild(el("span", "sep", "›"));
      path.appendChild(document.createTextNode(src.lesson || "Lesson"));
      if (src.section) {
        path.appendChild(el("span", "sep", "›"));
        path.appendChild(document.createTextNode(src.section));
      }
      chip.appendChild(path);
      wrap.appendChild(chip);

      const body = el("div", "source-body");
      body.id = bodyId;
      body.hidden = !state.sourceOpen;
      let highlighted = false;
      if (src.text) {
        body.appendChild(el("p", "quote-label", "From the lesson"));
        const text = el("div", "source-text");
        highlighted = appendHighlighted(text, src.text, it.quote || "");
        if (!highlighted) text.textContent = src.text;
        body.appendChild(text);
      }
      if (!highlighted && it.quote) {
        body.appendChild(el("p", "quote-label", "Quote"));
        body.appendChild(el("blockquote", "quote", `“${it.quote}”`));
      }
      wrap.appendChild(body);

      // The quote is always visible, so the idea is checkable at a glance.
      if (!state.sourceOpen && it.quote) {
        wrap.appendChild(el("blockquote", "quote-alone", `“${it.quote}”`));
      }
      return wrap;
    }

    function renderScript(it) {
      const box = el("div", "script-box");
      box.appendChild(el("h4", null, "Script"));
      if (state.warnings.length) {
        const list = el("ul", "warnings");
        state.warnings.forEach((w) => list.appendChild(el("li", null, w)));
        box.appendChild(list);
      }
      box.appendChild(el("p", "script-text", it.script));
      if (it.caption) {
        box.appendChild(el("h4", null, "Caption"));
        box.appendChild(el("p", "script-text", it.caption));
      }
      const actions = el("div", "script-actions");
      actions.appendChild(copyButton("Copy script", () => it.script));
      if (it.caption) actions.appendChild(copyButton("Copy caption", () => it.caption));
      box.appendChild(actions);
      return box;
    }

    render();
    return state;
  }

  // ---------- Ideas tab ----------

  const form = $("ideas-form");
  const submit = $("ideas-submit");
  const formStatus = $("ideas-status");
  const ideasError = $("ideas-error");
  const summary = $("ideas-summary");
  const list = $("ideas-list");
  let historyLoaded = false;

  async function loadFormats() {
    try {
      const formats = await api("/formats");
      const select = $("f-format");
      formats.forEach((f) => {
        const option = el("option", null, f.label);
        option.value = f.key;
        select.appendChild(option);
      });
    } catch {
      // The form still works with "Any format".
    }
  }

  function renderSummary(data, days) {
    clear(summary);
    const ideas = data.ideas || [];
    const dropped = data.dropped || [];
    if (!ideas.length) {
      const reason = data.empty_reason || (data.model ? "model_returned_none" : "all_recently_used");
      const messages = {
        empty_index: "No lessons are indexed yet. Add files to the library and run ingest.",
        no_topic_match: "Nothing in your lessons matches that topic. Try another word.",
        topic_matches_used: `Every passage that matches this topic was used in the last ${plural(days, "day", "days")} or is too short. Try another topic or lower the number of days.`,
        all_too_short: "Every passage in your lessons is shorter than the minimum length. Add longer material or lower MIN_PASSAGE_CHARS.",
        all_recently_used: `You've used every passage in the last ${plural(days, "day", "days")}. Lower the number of days or add new material.`,
        model_returned_none: "No usable ideas came back this time. Please try again.",
      };
      summary.textContent = messages[reason] || messages.model_returned_none;
    } else {
      const lessons = new Set(ideas.map((i) => `${i.source.course}|${i.source.lesson}`));
      const parts = [`${plural(ideas.length, "idea", "ideas")} from ${plural(lessons.size, "lesson", "lessons")}`];
      if (dropped.length) {
        const reasons = new Set(dropped.map((d) => d.reason));
        const why = reasons.size === 1 ? ` (${DROP_REASONS[dropped[0].reason] || dropped[0].reason})` : "";
        parts.push(`${dropped.length} dropped${why}`);
      }
      if (typeof data.fresh_passages_left === "number") parts.push(`${data.fresh_passages_left} fresh passages available`);
      if (data.latency_ms && typeof data.latency_ms.total === "number") parts.push(formatSeconds(data.latency_ms.total));
      if (data.usage && typeof data.usage.cost_usd === "number") parts.push(formatCost(data.usage.cost_usd));
      summary.textContent = parts.join(" · ");
    }
    if (dropped.length) {
      const details = el("details", "dropped");
      details.appendChild(el("summary", null, `Why ${plural(dropped.length, "idea was", "ideas were")} dropped`));
      const ul = el("ul");
      dropped.forEach((d) => {
        const li = el("li");
        li.appendChild(document.createTextNode(`${d.title || "Untitled"} `));
        li.appendChild(el("span", "why", `— ${DROP_REASONS[d.reason] || d.reason}`));
        ul.appendChild(li);
      });
      details.appendChild(ul);
      summary.appendChild(details);
    }
    summary.hidden = false;
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (submit.disabled) return;
    const format = $("f-format").value;
    const topic = $("f-topic").value.trim();
    const days = parseInt($("f-days").value, 10);
    const body = {
      count: parseInt($("f-count").value, 10),
      format: format || null,
      topic: topic || null,
      skip_used_days: Number.isNaN(days) ? Number($("f-days").defaultValue) || 0 : days,
    };
    const maxCount = Number($("f-count").max) || 30;
    if (Number.isNaN(body.count) || body.count < 1 || body.count > maxCount) {
      showError(ideasError, `Please choose between 1 and ${maxCount} ideas.`);
      return;
    }
    showError(ideasError, "");
    submit.disabled = true;
    formStatus.textContent = "Reading your lessons and writing ideas… this can take a few seconds.";
    try {
      const data = await post("/ideas", body);
      clear(list);
      renderSummary(data, body.skip_used_days);
      (data.ideas || []).forEach((idea) => list.appendChild(createCard(idea, onCardUpdate).node));
      formStatus.textContent = "Done.";
      historyLoaded = false;
    } catch (error) {
      showError(ideasError, error.message);
      formStatus.textContent = "";
    } finally {
      submit.disabled = false;
    }
  });

  function onCardUpdate() {
    // A change on a card makes the history list stale.
    historyLoaded = false;
    if (!$("history-body").hidden) loadHistory();
  }

  // ---------- History ----------

  const historyToggle = $("history-toggle");
  const historyBody = $("history-body");
  const historyList = $("history-list");
  const historyStatus = $("history-status");

  async function loadHistory() {
    historyStatus.textContent = "Loading…";
    try {
      const data = await api("/ideas?status=used&limit=50");
      clear(historyList);
      const ideas = data.ideas || [];
      historyStatus.textContent = ideas.length ? "" : "Nothing marked as used yet.";
      ideas.forEach((idea) => historyList.appendChild(createCard(idea, onHistoryUpdate).node));
      historyLoaded = true;
    } catch (error) {
      historyStatus.textContent = error.message;
    }
  }

  function onHistoryUpdate(state) {
    // Undone ideas stay visible until the next reload so the user can see what happened.
    if (state.idea.status !== "used") historyLoaded = false;
  }

  historyToggle.addEventListener("click", () => {
    const open = historyBody.hidden;
    historyBody.hidden = !open;
    historyToggle.setAttribute("aria-expanded", String(open));
    if (open && !historyLoaded) loadHistory();
  });

  // ---------- Coverage tab ----------

  async function loadCoverage() {
    const status = $("coverage-status");
    status.textContent = "Loading…";
    showError($("coverage-error"), "");
    try {
      const data = await api("/coverage");
      const lessons = data.lessons || [];
      const totals = data.totals || {};
      const total = totals.lessons ?? lessons.length;
      const never = totals.never_used ?? lessons.filter((l) => !l.used).length;
      const totalsBox = $("coverage-totals");
      totalsBox.textContent = `${total - never} of ${plural(total, "lesson", "lessons")} used at least once · ${never} never used`;
      totalsBox.hidden = false;

      const courses = new Map();
      lessons.forEach((l) => {
        if (!courses.has(l.course)) courses.set(l.course, []);
        courses.get(l.course).push(l);
      });
      const root = $("coverage-list");
      clear(root);
      courses.forEach((items, course) => {
        // Never-used lessons first; Array.sort is stable so the course order is kept otherwise.
        const sorted = [...items].sort((a, b) => Number(a.used > 0) - Number(b.used > 0));
        const section = el("section", "course");
        section.appendChild(el("h2", null, course));
        const ul = el("ul", "lessons");
        sorted.forEach((l) => {
          const li = el("li", `lesson${l.used > 0 ? "" : " never"}`);
          const name = el("span", "lesson-name", l.lesson);
          li.appendChild(name);
          if (!(l.used > 0)) li.appendChild(el("span", "badge badge-never", "Never used"));
          const stats = el("div", "lesson-stats");
          [["Passages", l.passages], ["Used", l.used], ["Suggested", l.suggested]].forEach(([label, value]) => {
            const s = el("span", null, `${label} `);
            s.appendChild(el("b", null, String(value)));
            stats.appendChild(s);
          });
          stats.appendChild(el("span", null, `Last used: ${l.last_used_at ? relativeDate(l.last_used_at) : "never"}`));
          li.appendChild(stats);
          ul.appendChild(li);
        });
        section.appendChild(ul);
        root.appendChild(section);
      });
      status.textContent = lessons.length ? "" : "No lessons found.";
    } catch (error) {
      status.textContent = "";
      showError($("coverage-error"), error.message);
    }
  }

  // ---------- Voice tab ----------

  // Minimal Markdown subset built from DOM nodes: ## headings, "- " bullets, paragraphs.
  function renderStyle(text) {
    const root = el("div", "style-sheet");
    let ul = null;
    let para = [];
    const flushPara = () => {
      if (para.length) root.appendChild(el("p", null, para.join(" ")));
      para = [];
    };
    text.split(/\r?\n/).forEach((raw) => {
      const line = raw.trim();
      const heading = /^#{1,6}\s+(.*)$/.exec(line);
      const bullet = /^[-*]\s+(.*)$/.exec(line);
      if (heading) {
        flushPara(); ul = null;
        root.appendChild(el("h3", null, heading[1]));
      } else if (bullet) {
        flushPara();
        if (!ul) { ul = el("ul"); root.appendChild(ul); }
        ul.appendChild(el("li", null, bullet[1]));
      } else if (!line) {
        flushPara(); ul = null;
      } else {
        ul = null;
        para.push(line);
      }
    });
    flushPara();
    return root;
  }

  function chipSection(title, items, extraClass) {
    const section = el("section", "voice-section");
    section.appendChild(el("h2", null, title));
    const ul = el("ul", "chip-list");
    items.forEach((p) => ul.appendChild(el("li", `chip${extraClass ? ` ${extraClass}` : ""}`, p)));
    section.appendChild(ul);
    return section;
  }

  async function loadVoice() {
    const status = $("voice-status");
    status.textContent = "Loading…";
    showError($("voice-error"), "");
    try {
      const data = await api("/voice");
      const root = $("voice-body");
      clear(root);
      const head = el("section", "voice-section");
      head.appendChild(el("h2", null, data.author || "Author"));
      head.appendChild(el("p", "voice-author", "Scripts follow this style sheet and these sample posts."));
      root.appendChild(head);
      if ((data.phrases || []).length) root.appendChild(chipSection("Phrases I use", data.phrases));
      if ((data.never_say || []).length) {
        const section = chipSection("Never say", data.never_say, "chip-never");
        root.appendChild(section);
      }
      if (data.style) {
        const section = el("section", "voice-section");
        section.appendChild(el("h2", null, "Style sheet"));
        section.appendChild(renderStyle(data.style));
        root.appendChild(section);
      }
      if ((data.samples || []).length) {
        const section = el("section", "voice-section");
        section.appendChild(el("h2", null, "Sample posts"));
        const cards = el("div", "cards");
        data.samples.forEach((s) => {
          const card = el("article", "card");
          const top = el("div", "card-top");
          top.appendChild(el("span", "badge", s.format_label || s.format));
          card.appendChild(top);
          card.appendChild(el("h3", "card-title", s.title));
          card.appendChild(el("p", "script-text", s.text));
          cards.appendChild(card);
        });
        section.appendChild(cards);
        root.appendChild(section);
      }
      status.textContent = "";
    } catch (error) {
      status.textContent = "";
      showError($("voice-error"), error.message);
    }
  }

  // ---------- Tabs ----------

  const tabs = Array.from(document.querySelectorAll(".tab"));
  const loaders = { coverage: loadCoverage, voice: loadVoice };

  function selectTab(tab, focus) {
    tabs.forEach((t) => {
      const on = t === tab;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
      $(`panel-${t.dataset.tab}`).hidden = !on;
    });
    if (focus) tab.focus();
    const loader = loaders[tab.dataset.tab];
    if (loader) loader();
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => selectTab(tab, false));
    tab.addEventListener("keydown", (event) => {
      let next = null;
      if (event.key === "ArrowRight") next = tabs[(index + 1) % tabs.length];
      else if (event.key === "ArrowLeft") next = tabs[(index - 1 + tabs.length) % tabs.length];
      else if (event.key === "Home") next = tabs[0];
      else if (event.key === "End") next = tabs[tabs.length - 1];
      if (next) { event.preventDefault(); selectTab(next, true); }
    });
  });

  // ---------- Footer ----------

  function setLimit(input, max, fallbackMax, def, fallbackDef) {
    const m = Number.isInteger(max) && max > 0 ? max : fallbackMax;
    const d = Number.isInteger(def) && def >= 0 ? Math.min(def, m) : fallbackDef;
    input.max = String(m);
    input.defaultValue = String(d);
    input.value = String(d);
  }

  function applyLimits(h) {
    setLimit($("f-count"), h.ideas_max_count, 30, h.ideas_default_count, 15);
    setLimit($("f-days"), h.skip_used_days_max, 365, h.skip_used_days_default, 30);
  }

  async function loadHealth() {
    try {
      const h = await api("/health");
      applyLimits(h);
      const models = [h.llm_model, h.embedding_model].filter(Boolean);
      if (models.length) $("footer-models").textContent = ` · Models: ${models.join(", ")}`;
    } catch {
      // The footer note is optional.
    }
  }

  loadFormats();
  loadHealth();
})();
