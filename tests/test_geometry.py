"""Board geometry: corner ordering, quad from a SAM mask, replay-log rebuild, person-track lookup."""
import numpy as np

from app.core.board import BoardResult, order_corners
from app.core.sam import PersonTrack, quad_from_mask


def test_order_corners_any_input_order():
    pts = [[1815, 860], [55, 160], [55, 865], [1815, 150]]
    c = order_corners(pts)
    assert c.tolist() == [[55, 160], [1815, 150], [1815, 860], [55, 865]]


def test_quad_from_mask_with_teacher_cutout():
    m = np.zeros((300, 500), bool)
    m[40:260, 30:470] = True
    m[120:300, 200:280] = False  # the teacher standing in front cuts a notch
    q = order_corners(quad_from_mask(m))
    expect = np.array([[30, 40], [469, 40], [469, 259], [30, 259]])
    assert np.abs(q - expect).max() <= 3


def test_board_rebuild_from_log():
    W, H = 4, 2
    r = BoardResult(width=W, height=H, mode="white", corners=[], background=(250, 250, 250), t_start=0, t_end=10,
                    fps=3, pieces=[], erases=[],
                    add_t=np.array([1.0, 1.0, 5.0]), add_idx=np.array([0, 1, 0]), add_piece=np.array([1, 1, 2]),
                    add_color=np.array([[0, 0, 0], [0, 0, 0], [255, 0, 0]], np.uint8),
                    rem_t=np.array([3.0]), rem_idx=np.array([0]))
    assert r.ink_at(0.5).sum() == 0
    assert r.ink_at(2.0).reshape(-1).tolist() == [True, True, False, False, False, False, False, False]
    assert r.ink_at(4.0).reshape(-1)[:2].tolist() == [False, True]  # pixel 0 erased at 3 s
    assert r.ink_at(6.0).reshape(-1)[:2].tolist() == [True, True]  # and rewritten at 5 s
    assert r.image_at(6.0)[0, 0].tolist() == [255, 0, 0]


def test_person_track_unions_neighbours():
    tr = PersonTrack(start=10.0, fps=2.0, size=(4, 4))
    a = np.zeros((4, 4), bool); a[0, 0] = True
    b = np.zeros((4, 4), bool); b[3, 3] = True
    tr.put(0, a)
    tr.put(1, b)
    assert tr(10.0).sum() == 1
    m = tr(10.25)  # between samples: both
    assert m[0, 0] and m[3, 3]
    assert tr(20.0) is None
