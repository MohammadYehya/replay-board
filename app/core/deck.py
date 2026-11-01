"""Optional PDF deck for slide lectures (SPEC 2, M3): match each slide state to its page; read the crisp
page instead of the video frame; show the page instead of the video's slide when nothing differs.

A build step (bullets still to come) or pen ink on the slide keeps the video's image, so the
board never shows something before the teacher did.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .media import MediaError


def render_pages(pdf: Path, width: int = 1920, max_pages: int = 300) -> list[np.ndarray]:
    """Every page as a BGR image `width` px wide."""
    import pypdfium2 as pdfium

    try:
        doc = pdfium.PdfDocument(str(pdf))
    except Exception as e:  # not a PDF, encrypted, broken...
        raise MediaError(f"could not open the slide deck {Path(pdf).name}: {e}") from None
    pages = []
    try:
        for i in range(min(len(doc), max_pages)):
            page = doc[i]
            w, h = page.get_size()
            bitmap = page.render(scale=width / max(1.0, w))
            img = np.ascontiguousarray(bitmap.to_numpy()[..., :3])
            if bitmap.mode in ("RGB", "RGBA", "RGBX"):
                img = img[..., ::-1]
            pages.append(np.ascontiguousarray(img))
    finally:
        doc.close()
    return pages


def _fit(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _signature(img: np.ndarray, live: list, scale: float) -> np.ndarray:
    g = cv2.cvtColor(cv2.resize(img, (192, int(round(192 * img.shape[0] / img.shape[1]))),
                                interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = g.shape[1] / img.shape[1]
    for x, y, w, h in live:
        g[int(y * k):int((y + h) * k) + 1, int(x * k):int((x + w) * k) + 1] = np.median(g)
    g -= g.mean()
    n = np.linalg.norm(g)
    return g / n if n > 0 else g


def match_pages(states: list, pages: list[np.ndarray], live: list, min_score: float = 0.6) -> dict:
    """{state id: (page index, score)} for states whose best page correlates well enough."""
    if not pages:
        return {}
    out = {}
    sigs = None
    for st in states:
        if st.image is None or st.blank:
            continue
        H, W = st.image.shape[:2]
        if sigs is None:
            sigs = [_signature(_fit(p, (W, H)), live, 1.0) for p in pages]
        s = _signature(st.image, live, 1.0)
        scores = [float((s * q).sum()) for q in sigs]
        i = int(np.argmax(scores))
        if scores[i] >= min_score:
            out[st.id] = (i, scores[i])
    return out


def same_content(state_img: np.ndarray, page_img: np.ndarray, bg, live: list, min_blob: int = 60) -> bool:
    """True if the page shows exactly what the state shows: no mark missing, none extra.

    Judged by blobs, not by share of all ink: a pen underline is tiny next to a chart's bars
    but must still keep the video's slide.
    """
    from .slides import content_mask

    a = content_mask(state_img, bg, live)
    b = content_mask(page_img, bg, live)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    a_d, b_d = cv2.dilate(a.astype(np.uint8), k) > 0, cv2.dilate(b.astype(np.uint8), k) > 0
    for diff in (b & ~a_d, a & ~b_d):  # a bullet still to come; pen ink or a pointer on screen
        n, _, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(diff.astype(np.uint8), k), connectivity=8)
        if n > 1 and st[1:, cv2.CC_STAT_AREA].max() >= min_blob * 4:  # dilation grows a blob ~4x
            return False
    return True


def visible_on(state_img: np.ndarray, box: list, bg, live: list, page_img: np.ndarray) -> bool:
    """Is a page element already on screen in this state (not a bullet still to come)?"""
    from .slides import content_mask

    x, y, w, h = (int(round(v)) for v in box)
    if w <= 0 or h <= 0:
        return False
    a = content_mask(state_img, bg, live)[y:y + h, x:x + w]
    b = content_mask(page_img, bg, live)[y:y + h, x:x + w]
    if b.sum() == 0:
        return True
    return a.sum() >= 0.25 * b.sum()
