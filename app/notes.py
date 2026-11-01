"""Study notes from replay.json: chapters, each piece of writing (or slide element) with its time and
what was said about it. As data (for the player's notes page) or as Markdown with $LaTeX$."""
from __future__ import annotations

DRAWINGS = {"diagram", "graph", "arrow", "scribble", "underline", "box", "figure", "chart", "drawing"}


def _mmss(t: float) -> str:
    t = max(0, int(round(t)))
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def _said(words: list[dict], a: float, b: float, limit: int = 24) -> str:
    ws = [w["w"] for w in words if a <= w["s"] <= b]
    text = " ".join(ws[:limit])
    return text + (" ..." if len(ws) > limit else "")


def _item(read: dict) -> dict | None:
    """{latex, text, drawing} for a reading worth a line in the notes, else None."""
    kind = (read.get("kind") or "").lower()
    latex = (read.get("latex") or "").strip().strip("$")
    text = (read.get("text") or "").strip()
    if latex and kind not in DRAWINGS:
        return {"latex": latex, "text": text, "drawing": False}
    if not text or text.lower() == "unreadable":
        return None
    return {"latex": "", "text": text, "drawing": kind in DRAWINGS}


def notes_data(replay: dict) -> dict:
    """{title, chapters: [{t, title, items: [{t, latex, text, drawing, said}]}]}"""
    words = (replay.get("transcript") or {}).get("words") or []
    chapters = sorted(replay.get("chapters") or [], key=lambda c: c["t"]) or [{"t": 0.0, "title": "Lecture"}]
    title = replay.get("subject") or replay.get("video", {}).get("file", "Lecture")
    entries = []  # (time, item, spoken ranges)
    if replay.get("type") == "slides":
        seen = set()  # a build step repeats the slide's earlier elements
        for st in replay["slides"]["states"]:
            for e in st.get("elements") or []:
                item = _item(e)
                key = " ".join(((item or {}).get("latex") or (item or {}).get("text") or "").lower().split())
                if item and key not in seen:
                    seen.add(key)
                    t = (e.get("spoken") or [{}])[0].get("start", st["t_start"])
                    entries.append((t, item, e.get("spoken") or []))
    else:
        pieces = {p.get("id"): p for p in replay["board"]["pieces"]}
        in_line = set()
        for ln in replay["board"].get("lines") or []:  # a line written in bursts reads as one entry
            item = _item(ln.get("read") or {})
            members = [pieces[i] for i in ln["pieces"] if i in pieces]
            if not item or not members:
                continue
            in_line.update(ln["pieces"])
            sp = sorted((s for p in members for s in (p.get("spoken") or [])), key=lambda s: s["start"])
            if ln.get("pre_existing") and not sp:
                continue
            t = sp[0]["start"] if sp else min(p.get("redraw_at") or p["t_seen"] for p in members)
            entries.append((t, item, sp))
        for p in replay["board"]["pieces"]:
            if p.get("id") in in_line:
                continue
            item = _item(p.get("read") or {})
            if not item:
                continue
            sp = p.get("spoken") or []
            if p.get("pre_existing") and not sp:
                continue  # was on the board before the recording and never discussed
            t = sp[0]["start"] if sp else (p.get("redraw_at") or p["t_seen"])
            entries.append((t, item, sp))
    # A drawing nobody talked about, or a scribble, is noise in notes.
    entries = [e for e in entries if not e[1]["drawing"] or (e[2] and "scribble" not in e[1]["text"].lower())]
    entries.sort(key=lambda e: e[0])

    out = []
    for i, ch in enumerate(chapters):
        end = chapters[i + 1]["t"] if i + 1 < len(chapters) else float("inf")
        items = [{"t": round(float(t), 2), **item, "said": _said(words, sp[0]["start"], sp[0]["end"]) if sp else ""}
                 for t, item, sp in entries if ch["t"] <= t < end or (i == 0 and t < ch["t"])]
        out.append({"t": ch["t"], "title": ch["title"], "items": items})
    return {"title": title, "chapters": out}


def build_notes(replay: dict) -> str:
    data = notes_data(replay)
    out = [f"# {data['title']}", "",
           "_Notes made by Replay Board: each line shows when it was explained, and what was said._", ""]
    for ch in data["chapters"]:
        out.append(f"## {ch['title']} ({_mmss(ch['t'])})")
        out.append("")
        for it in ch["items"]:
            line = f"${it['latex']}$" if it["latex"] else (f"*{it['text']}*" if it["drawing"] else it["text"])
            out.append(f"- **{_mmss(it['t'])}** {line}" + (f" — “{it['said']}”" if it["said"] else ""))
        if not ch["items"]:
            out.append("- (talk only)")
        out.append("")
    return "\n".join(out)
