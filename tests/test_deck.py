"""Optional PDF deck: matching pages to slide states, and when a page may replace the video's slide."""
import io

import cv2
import numpy as np
import pypdfium2 as pdfium

from app.core.deck import match_pages, render_pages, same_content, visible_on
from app.core.slides import SlideState

WHITE = (255, 255, 255)


def slide(lines, ink=False):
    img = np.full((540, 960, 3), 255, np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(img, text, (60, 120 + 60 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (30, 30, 30), 2, cv2.LINE_AA)
    if ink:
        cv2.line(img, (60, 135), (400, 138), (40, 40, 220), 4, cv2.LINE_AA)
    return img


def make_pdf(path, images):
    pdf = pdfium.PdfDocument.new()
    for img in images:
        h, w = img.shape[:2]
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        page = pdf.new_page(w * 0.75, h * 0.75)
        im = pdfium.PdfImage.new(pdf)
        im.load_jpeg(io.BytesIO(buf.tobytes()), inline=True)
        im.set_matrix(pdfium.PdfMatrix().scale(w * 0.75, h * 0.75))
        page.insert_obj(im)
        page.gen_content()
    pdf.save(str(path))


def test_render_and_match(tmp_path):
    a = slide(["Title one", "first point", "second point"])
    b = slide(["Another slide", "x = 3", "y = 4", "z = 5"])
    make_pdf(tmp_path / "deck.pdf", [a, b])
    pages = render_pages(tmp_path / "deck.pdf", width=960)
    assert len(pages) == 2 and pages[0].shape[1] == 960
    states = [SlideState(0, "slide", 0, 5, 0, image=b), SlideState(1, "slide", 5, 9, 5, image=a)]
    m = match_pages(states, pages, [])
    assert m[0][0] == 1 and m[1][0] == 0 and m[0][1] > 0.9


def test_page_replaces_only_identical_slides():
    full = slide(["Title", "point one", "point two", "point three"])
    build = slide(["Title", "point one", "point two"])  # the third bullet is still to come
    inked = slide(["Title", "point one", "point two", "point three"], ink=True)
    assert same_content(full, full.copy(), WHITE, [])
    assert not same_content(build, full, WHITE, [])
    assert not same_content(inked, full, WHITE, [])


def test_bullets_still_to_come_are_not_visible_yet():
    full = slide(["Title", "point one", "point two", "point three"])
    build = slide(["Title", "point one", "point two"])
    third = [55, 270, 300, 35]  # where "point three" is on the page
    assert visible_on(full, third, WHITE, [], full)
    assert not visible_on(build, third, WHITE, [], full)
