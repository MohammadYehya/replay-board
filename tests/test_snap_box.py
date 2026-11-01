"""Model element boxes are snapped onto the slide's real ink."""
import cv2
import numpy as np

from app.core.slides import content_mask, snap_box


def slide():
    img = np.full((540, 960, 3), 255, np.uint8)
    cv2.putText(img, "First bullet line here", (80, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(img, "Second line, a bit longer", (80, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.rectangle(img, (600, 300), (800, 450), (40, 120, 200), -1)
    return img


def test_text_box_one_line_off_snaps_to_its_line():
    img = slide()
    ink = content_mask(img, (255, 255, 255))
    ys, xs = np.nonzero(ink[180:225, :])
    truth = [xs.min(), 180 + ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1]
    rough = [truth[0] + 12, truth[1] - 22, truth[2] - 30, truth[3] + 4]  # model box: up 22 px, too narrow
    got = snap_box(rough, ink, "bullet")
    assert all(abs(a - b) <= 2 for a, b in zip(got, truth)), (got, truth)


def test_figure_box_tightens_to_ink():
    img = slide()
    ink = content_mask(img, (255, 255, 255))
    got = snap_box([592, 292, 220, 170], ink, "figure")
    assert got == [600, 300, 201, 151]


def test_box_over_empty_space_is_kept():
    ink = content_mask(slide(), (255, 255, 255))
    assert snap_box([300, 470, 100, 30], ink, "text") == [300, 470, 100, 30]


def test_word_at_a_point():
    from app.core.slides import word_at
    ink = content_mask(slide(), (255, 255, 255))
    b = word_at(ink, 100, 140)  # on "First"
    assert b is not None and b[0] <= 100 <= b[0] + b[2] and b[2] < 120
    assert word_at(ink, 400, 480) is None
