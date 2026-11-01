"""The fast SAM mask decoder must match meta-sam-parser bit for bit."""
import numpy as np
import pytest
from meta_sam_parser import (CompletedOutcome, SegmentationMask, decode_mask_to_raster,
                             video_segmentation_format)
from meta_sam_parser import _mask_codec as official  # test oracle only: Meta's own encoder

from app.core.meta_api import parse_sam_records
from app.core.sam_codec import decode_one_bit, fast_records


def random_mask(rng, h, w):
    m = np.zeros((h, w), np.uint8)
    for _ in range(rng.integers(1, 5)):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        ry, rx = rng.integers(1, max(2, h // 2)), rng.integers(1, max(2, w // 2))
        yy, xx = np.ogrid[:h, :w]
        m[((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1] = 1
    noise = rng.random((h, w)) < 0.02
    return (m ^ noise).astype(np.uint8)


@pytest.mark.parametrize("seed", range(12))
def test_decoder_matches_meta(seed):
    rng = np.random.default_rng(seed)
    h, w = int(rng.integers(1, 90)), int(rng.integers(1, 120))
    m = random_mask(rng, h, w)
    payload = official._encode_raster(m.reshape(-1).tolist(), w, h)
    ref = np.frombuffer(decode_mask_to_raster(SegmentationMask("one_bit", payload, w, h)), np.uint8)
    fast = decode_one_bit(payload, w, h)
    assert fast is not None and np.array_equal(fast.reshape(-1), ref.astype(bool))
    assert np.array_equal(ref.reshape(h, w), m)


def test_corrupt_payload_is_rejected():
    assert decode_one_bit("!!!!", 4, 4) is None
    assert decode_one_bit("~abc", 4, 4) is None


def test_video_lines_match_meta_parser():
    rng = np.random.default_rng(7)
    lines = []
    for f in (0, 1, 3):
        recs = []
        for oid in (0, 2):
            h, w = 20 + f, 30 + oid
            m = random_mask(rng, h, w)
            p = official._encode_raster(m.reshape(-1).tolist(), w, h)
            x1, y1 = 5 * oid + f, 7
            recs.append(f"{oid}<|box;x1={x1};y1={y1};x2={x1 + w - 1};y2={y1 + h - 1};w=200;h=100;c=0.8|>"
                        f"<|mask;x=0;y=0;c=0.8;data={h},{w},{p}|>")
        lines.append(f"<{f}f>" + ",".join(recs))
    text = "\n".join(lines) + "\n"

    fast = fast_records(text)
    parser = video_segmentation_format().create_parser()
    parser.push(text)
    ref = [r for r in parser.finish(CompletedOutcome()).result.records if r.kind == "mask"]
    assert fast is not None and len(fast) == len(ref) == 6
    for a, b in zip(fast, ref):
        assert a["frame"] == b.frame.frame_index and a["id"] == b.object_id
        assert a["box"] == [b.bounds.left, b.bounds.top, b.bounds.right, b.bounds.bottom]
        exp = np.frombuffer(decode_mask_to_raster(b.mask), np.uint8).reshape(b.mask.height, b.mask.width)
        assert np.array_equal(a["local"], exp.astype(bool))
    # The public entry point gives the same answer.
    recs = parse_sam_records(text, "video")
    assert [r["frame"] for r in recs] == [0, 0, 1, 1, 3, 3]
