"""Fast reader for SAM 3.1 output (one_bit masks).

Meta's meta-sam-parser decodes every mask in pure Python (~5 us per pixel, twice with its
canonical check), which takes minutes for a lecture's worth of person masks. This is a numba
port of its one_bit decoder, plus a regex over the documented token grammar. It is checked
bit-for-bit against meta-sam-parser in tests/test_sam_codec.py, and at run time the first mask
of every reply is re-decoded with Meta's decoder; any mismatch falls back to Meta's parser.
"""
from __future__ import annotations

import re

import numpy as np
from numba import njit

_ALPHABET = "".join(chr(c) for c in range(0x21, 0x7F) if chr(c) not in '"\\,;<>|')
_VALUES = np.full(256, -1, np.int64)
for _i, _c in enumerate(_ALPHABET):
    _VALUES[ord(_c)] = _i

_LINE = re.compile(r"<(\d+)f>")
_RECORD = re.compile(r"(\d+)<\|box;([^|]*)\|><\|mask;([^|]*)\|>")


@njit(cache=True)
def _unpack(codes, values):
    """base85 payload (after the encoding marker) -> bytes. Returns (data, ok)."""
    n = codes.size
    empty = np.zeros(0, np.uint8)
    if n < 5:
        return empty, False
    length = 0
    for i in range(5):
        v = values[codes[i]]
        if v < 0:
            return empty, False
        length = length * 85 + v
    rem = length % 4
    expected = 5 + (length // 4) * 5 + (0 if rem == 0 else rem + 1)
    if n != expected:
        return empty, False
    out = np.zeros(length, np.uint8)
    src = 5
    dst = 0
    full = length - rem
    while dst < full:
        value = 0
        for i in range(5):
            v = values[codes[src + i]]
            if v < 0:
                return empty, False
            value = value * 85 + v
        if value > 0xFFFFFFFF:
            return empty, False
        src += 5
        out[dst] = (value >> 24) & 0xFF
        out[dst + 1] = (value >> 16) & 0xFF
        out[dst + 2] = (value >> 8) & 0xFF
        out[dst + 3] = value & 0xFF
        dst += 4
    if rem > 0:
        value = 0
        for i in range(5):
            if i < rem + 1:
                v = values[codes[src + i]]
                if v < 0:
                    return empty, False
            else:
                v = 84
            value = value * 85 + v
        if value > 0xFFFFFFFF:
            return empty, False
        for i in range(rem):
            out[dst + i] = (value >> (24 - i * 8)) & 0xFF
    return out, True


@njit(cache=True)
def _decode_raster(data, width, height):
    """Adaptive binary range decoder with a 12-pixel context (port of meta_sam_parser)."""
    M32 = 0xFFFFFFFF
    TOP = 1 << 24
    BOTTOM = 1 << 16
    out = np.zeros(width * height, np.uint8)
    if data.size < 4:
        return out, False
    pos_in = 0
    code = 0
    for _ in range(4):
        code = ((code << 8) | data[pos_in]) & M32
        pos_in += 1
    low = 0
    rng = M32
    counts = np.ones(4096 * 2, np.int64)
    pw = width + 4
    padded = np.zeros(pw * (height + 2), np.int64)
    for y in range(height):
        row = (y + 2) * pw + 2
        for x in range(width):
            p = row + x
            q = p - pw
            r = p - 2 * pw
            ctx = (padded[p - 1] | (padded[p - 2] << 1) | (padded[q - 2] << 2) | (padded[q - 1] << 3)
                   | (padded[q] << 4) | (padded[q + 1] << 5) | (padded[q + 2] << 6) | (padded[r - 2] << 7)
                   | (padded[r - 1] << 8) | (padded[r] << 9) | (padded[r + 1] << 10) | (padded[r + 2] << 11))
            off = ctx << 1
            zero = counts[off]
            one = counts[off + 1]
            total = zero + one
            scaled = rng // total
            if scaled == 0:
                return out, False
            value = ((code - low) & M32) // scaled
            if value >= total:
                value = total - 1
            if value < zero:
                bit = 0
                cum = 0
                freq = zero
            else:
                bit = 1
                cum = zero
                freq = one
            low = (low + scaled * cum) & M32
            rng = scaled * freq
            while True:
                if (low ^ ((low + rng) & M32)) < TOP:
                    pass
                elif rng < BOTTOM:
                    rng = ((-low) & M32) & (BOTTOM - 1)
                else:
                    break
                if pos_in >= data.size:
                    return out, False
                code = ((code << 8) | data[pos_in]) & M32
                pos_in += 1
                low = (low << 8) & M32
                rng = (rng << 8) & M32
            padded[p] = bit
            out[y * width + x] = bit
            counts[off + bit] = (counts[off + bit] + 14) & 0xFFFF
            if counts[off] + counts[off + 1] >= 4096:
                a = counts[off] >> 1
                b = counts[off + 1] >> 1
                counts[off] = a if a > 0 else 1
                counts[off + 1] = b if b > 0 else 1
    return out, True


def decode_one_bit(payload: str, width: int, height: int) -> np.ndarray | None:
    """`payload` includes the leading '!' marker. Returns a bool (height, width) array or None."""
    if not payload.startswith("!"):
        return None
    codes = np.frombuffer(payload[1:].encode("ascii", "replace"), np.uint8)
    data, ok = _unpack(codes, _VALUES)
    if not ok:
        return None
    raster, ok = _decode_raster(data, int(width), int(height))
    if not ok:
        return None
    return raster.reshape(height, width).astype(bool)


def _fields(text: str) -> dict:
    out = {}
    for part in text.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def fast_records(text: str) -> list[dict] | None:
    """Parse output lines into records like app.core.meta_api.parse_sam_records. None if unsure."""
    recs = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _LINE.match(line)
        if not m:
            return None
        frame = int(m.group(1))
        body = line[m.end():]
        found = list(_RECORD.finditer(body))
        if sum(len(f.group(0)) for f in found) + max(0, len(found) - 1) != len(body):
            return None  # something we don't understand on this line
        for f in found:
            box, mask = _fields(f.group(2)), _fields(f.group(3))
            try:
                x1, y1, x2, y2 = (float(box[k]) for k in ("x1", "y1", "x2", "y2"))
                h, w, payload = mask["data"].split(",", 2)
                h, w = int(h), int(w)
                if float(mask.get("x", 0)) != 0 or float(mask.get("y", 0)) != 0:
                    return None  # never seen; let Meta's parser handle it
            except (KeyError, ValueError):
                return None
            local = decode_one_bit(payload, w, h)
            if local is None:
                return None
            c = mask.get("c", box.get("c"))
            # Wire x2/y2 are inclusive; boxes here are half-open like meta-sam-parser's bounds.
            recs.append({"frame": frame, "id": f.group(1), "score": float(c) if c is not None else None,
                         "box": [x1, y1, x2 + 1, y2 + 1], "local": local, "_payload": (payload, w, h)})
    return recs
