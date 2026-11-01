"""Study notes from replay.json: chapters, each piece of writing (or slide element) with its time and
what was said about it. Markdown with $LaTeX$, readable on GitHub, Obsidian, Notion..."""
from __future__ import annotations

DRAWINGS = {"diagram", "graph", "arrow", "scribble", "underline", "box", "figure", "chart", "drawing"}


def _mmss(t: float) -> str:
    t = max(0, int(round(t)))
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def _said(words: list[dict], a: float, b: float, limit: int = 24) -> str:
    ws = [w["w"] for w in words if a <= w["s"] <= b]
    text = " ".join(ws[:limit])
    return text + (" ..." if len(ws) > limit else "")


def _item(read: dict) -> str | None:
    kind = (read.get("kind") or "").lower()
    latex = (read.get("latex") or "").strip().strip("$")
    text = (read.get("text") or "").strip()
    if latex and kind not in DRAWINGS:
        return f"${latex}$"
    if not text or text.lower() == "unreadable":
        return None
    return f"*{text}*" if kind in DRAWINGS else text


def build_notes(replay: dict) -> str:
    words = (replay.get("transcript") or {}).get("words") or []
    chapters = sorted(replay.get("chapters") or [], key=lambda c: c["t"]) or [{"t": 0.0, "title": "Lecture"}]
    title = replay.get("subject") or replay.get("video", {}).get("file", "Lecture")
    entries = []  # (time, line)
    if replay.get("type") == "slides":
        seen = set()  # a build step repeats the slide's earlier elements
        for st in replay["slides"]["states"]:
            for e in st.get("elements") or []:
                line = _item(e)
                key = " ".join((line or "").lower().split())
                if line and key not in seen:
                    seen.add(key)
                    t = (e.get("spoken") or [{}])[0].get("start", st["t_start"])
                    entries.append((t, line, e.get("spoken") or []))
    else:
        for p in replay["board"]["pieces"]:
            line = _item(p.get("read") or {})
            if not line:
                continue
            sp = p.get("spoken") or []
            t = sp[0]["start"] if sp else (p.get("redraw_at") or p["t_seen"])
            if p.get("pre_existing") and not sp:
                continue  # was on the board before the recording and never discussed
            entries.append((t, line, sp))
    entries.sort(key=lambda e: e[0])

    out = [f"# {title}", "", "_Notes made by Replay Board: each line shows when it was explained, and what was said._", ""]
    for i, ch in enumerate(chapters):
        end = chapters[i + 1]["t"] if i + 1 < len(chapters) else float("inf")
        mine = [e for e in entries if ch["t"] <= e[0] < end or (i == 0 and e[0] < ch["t"])]
        out.append(f"## {ch['title']} ({_mmss(ch['t'])})")
        out.append("")
        for t, line, sp in mine:
            said = _said(words, sp[0]["start"], sp[0]["end"]) if sp else ""
            out.append(f"- **{_mmss(t)}** {line}" + (f" — “{said}”" if said else ""))
        if not mine:
            out.append("- (talk only)")
        out.append("")
    return "\n".join(out)
