"""Synthetic bumper crops for the S1 digit reader (design §5).

A random 1-5 digit team number in a bold sans font, white (sometimes outlined) on a red / blue /
black fabric-like bumper, pasted onto a background (a real harvest crop when one is given, else
noise) with the harvester's 25% bumper padding, then perspective, scale, blur, JPEG and an
interlace comb (2026 broadcast video is interlaced). Returns (BGR uint8 crop, label).
"""
from __future__ import annotations

import glob
import io
import os
from functools import lru_cache
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIRS = ("/usr/share/fonts", os.path.expanduser("~/.local/share/fonts"))
_FONT_OK = ("bold", "heavy", "black", "extrabold", "condbold")
_FONT_BAD = ("mono", "serif", "slab", "cjk", "semibold", "compressed", "symbol", "emoji", "math")
_SANS = ("dejavusans", "liberationsans", "freesans", "firasans", "opensans", "notosans-",
         "nimbussans", "ubuntu", "roboto-", "lato", "cantarell", "sourcesans", "inter")

# digit-count mix (real FRC numbers are mostly 4 digits; 5-digit teams exist since 2024)
LEN_P = {1: 0.04, 2: 0.08, 3: 0.20, 4: 0.50, 5: 0.18}
COLOURS = {  # BGR fabric base colours
    "red": [(40, 35, 190), (30, 30, 160), (50, 50, 220), (45, 25, 150)],
    "blue": [(170, 70, 20), (140, 50, 15), (200, 100, 40), (120, 40, 10)],
    "black": [(25, 25, 25), (40, 40, 40)],
}


@lru_cache(maxsize=1)
def font_files() -> Tuple[str, ...]:
    out = []
    for d in FONT_DIRS:
        for f in glob.glob(os.path.join(d, "**", "*.[ot]tf"), recursive=True):
            n = os.path.basename(f).lower()
            if any(s in n for s in _SANS) and any(k in n for k in _FONT_OK) and not any(b in n for b in _FONT_BAD):
                out.append(f)
    return tuple(sorted(set(out)))


@lru_cache(maxsize=256)
def _font(path: str, size: int):
    return ImageFont.truetype(path, size)


def random_number(rng: np.random.Generator) -> str:
    n = int(rng.choice(list(LEN_P), p=list(LEN_P.values())))
    first = str(rng.integers(1, 10))
    if n == 5:  # 5-digit FRC teams are 10000-11999 today; allow a few beyond
        first = "1" if rng.random() < 0.85 else str(rng.integers(2, 10))
    return first + "".join(str(d) for d in rng.integers(0, 10, n - 1))


def _fabric(h: int, w: int, colour: str, rng: np.random.Generator) -> np.ndarray:
    base = np.array(COLOURS[colour][rng.integers(len(COLOURS[colour]))], np.float32)
    base = base * rng.uniform(0.75, 1.25)
    img = np.ones((h, w, 3), np.float32) * base
    # fabric grain + soft horizontal folds + vertical light gradient
    img += rng.normal(0, rng.uniform(3, 12), (h, w, 1))
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    img *= 1 + rng.uniform(-0.25, 0.25) * (yy - 0.5)
    for _ in range(rng.integers(0, 3)):
        y0 = rng.uniform(0, 1)
        img *= 1 - rng.uniform(0.05, 0.25) * np.exp(-((yy - y0) ** 2) / (2 * 0.04 ** 2))
    return np.clip(img, 0, 255)


def _text_layer(text: str, h: int, w: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """White (or light) digits on a transparent layer -> (rgb, alpha) at bumper resolution."""
    fonts = font_files()
    th = int(h * rng.uniform(0.5, 0.88))
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    if fonts:
        font = _font(fonts[rng.integers(len(fonts))], max(8, th))
        bbox = d.textbbox((0, 0), text, font=font)
        tw, tht = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if tw > w * 0.95:  # shrink to fit
            font = _font(font.path, max(8, int(th * w * 0.95 / tw)))
            bbox = d.textbbox((0, 0), text, font=font)
            tw, tht = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (w - tw) * rng.uniform(0.15, 0.85) - bbox[0]
        y = (h - tht) * rng.uniform(0.3, 0.7) - bbox[1]
        stroke = int(rng.integers(0, max(1, th // 14) + 1)) if rng.random() < 0.3 else 0
        d.text((x, y), text, fill=255, font=font, stroke_width=stroke, stroke_fill=255)
        alpha = np.asarray(img, np.float32) / 255
    else:  # no TrueType fonts on this machine: Hershey fallback
        a = np.zeros((h, w), np.uint8)
        scale = th / 30
        (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, max(1, th // 8))
        x = int((w - tw) * rng.uniform(0.15, 0.85))
        cv2.putText(a, text, (max(0, x), int((h + tht) / 2)), cv2.FONT_HERSHEY_DUPLEX, scale, 255,
                    max(1, th // 8))
        alpha = a.astype(np.float32) / 255
    shade = rng.uniform(200, 255)
    rgb = np.ones((h, w, 3), np.float32) * shade
    if rng.random() < 0.15:  # outline digits (dark rim)
        k = np.ones((3, 3), np.uint8)
        rim = cv2.dilate((alpha > 0.5).astype(np.uint8), k, iterations=int(rng.integers(1, 3))).astype(np.float32)
        rgb = rgb * alpha[..., None] + 20 * (1 - alpha[..., None])
        alpha = np.maximum(alpha, rim * 0.9)
    if rng.random() < 0.4:  # italic-ish shear
        sh = rng.uniform(-0.25, 0.1)
        M = np.float32([[1, sh, -sh * h / 2], [0, 1, 0]])
        alpha = cv2.warpAffine(alpha, M, (w, h))
        rgb = cv2.warpAffine(rgb, M, (w, h))
    return rgb, alpha


def interlace_comb(img: np.ndarray, rng: np.random.Generator, shift: Optional[int] = None) -> np.ndarray:
    """Shift every other row horizontally (field motion in interlaced video)."""
    s = int(shift if shift is not None else rng.integers(1, 5)) * (1 if rng.random() < 0.5 else -1)
    out = img.copy()
    out[1::2] = np.roll(img[1::2], s, axis=1)
    return out


def jpeg(img: np.ndarray, q: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def render(rng: np.random.Generator, background: Optional[np.ndarray] = None,
           text: Optional[str] = None, colour: Optional[str] = None) -> Tuple[np.ndarray, str]:
    text = text or random_number(rng)
    colour = colour or str(rng.choice(["red", "blue", "black"], p=[0.45, 0.45, 0.10]))
    bh = 64
    bw = int(bh * rng.uniform(1.3, 3.6))
    bump = _fabric(bh, bw, colour, rng)
    rgb, a = _text_layer(text, bh, bw, rng)
    bump = bump * (1 - a[..., None]) + rgb * a[..., None]
    if rng.random() < 0.3:  # top edge (robot frame / trim) or aluminium angle
        t = int(rng.integers(2, 8))
        bump[:t] = rng.uniform(60, 200)
    # canvas with harvester padding (25% of the box per side) and a little box jitter
    ch, cw = int(bh * 1.5), int(bw * 1.5)
    if background is not None and background.size:
        canvas = cv2.resize(background, (cw, ch), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    else:
        canvas = np.clip(rng.normal(110, 40, (ch, cw, 3)), 0, 255).astype(np.float32)
    y0 = int(bh * 0.25 + rng.uniform(-0.1, 0.1) * bh)
    x0 = int(bw * 0.25 + rng.uniform(-0.08, 0.08) * bw)
    y0, x0 = int(np.clip(y0, 0, ch - bh)), int(np.clip(x0, 0, cw - bw))
    # perspective of the bumper itself (robot turned / camera elevation)
    src = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
    j = lambda s: rng.uniform(-s, s)  # noqa: E731
    yaw = j(0.25) * bh
    dst = np.float32([[x0 + j(0.06) * bw, y0 + yaw + j(0.08) * bh], [x0 + bw + j(0.06) * bw, y0 - yaw + j(0.08) * bh],
                      [x0 + bw + j(0.06) * bw, y0 + bh - yaw * 0.6 + j(0.05) * bh],
                      [x0 + j(0.06) * bw, y0 + bh + yaw * 0.6 + j(0.05) * bh]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(bump, M, (cw, ch))
    mask = cv2.warpPerspective(np.ones((bh, bw), np.float32), M, (cw, ch))[..., None]
    img = canvas * (1 - mask) + warped * mask
    # light occluders (game pieces / poles) that never hide more than a sliver of the digits
    for _ in range(int(rng.integers(0, 3))):
        if rng.random() < 0.5:
            r = int(rng.uniform(0.15, 0.35) * bh)
            side = rng.random() < 0.5
            cx = int(x0 + (rng.uniform(-0.05, 0.12) if side else rng.uniform(0.88, 1.05)) * bw)
            cy = int(y0 + rng.uniform(0.0, 1.0) * bh)
            col = [(0, 220, 240), (40, 200, 40), (200, 120, 230), (30, 120, 250)][rng.integers(4)]
            cv2.circle(img, (cx, cy), r, col, -1)
        else:
            xx = int(rng.uniform(0, cw))
            cv2.line(img, (xx, 0), (xx + int(j(0.2) * ch), ch), (rng.uniform(40, 200),) * 3,
                     int(rng.integers(1, 4)))
    img = np.clip(img, 0, 255).astype(np.uint8)
    # down-scale to a broadcast-like crop width, then blur / comb / JPEG
    out_w = int(np.clip(rng.lognormal(np.log(95), 0.45), 32, 260) * 1.5)
    out_h = max(12, int(round(out_w * ch / cw)))
    img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
    rel = min(1.0, out_w / 150)  # keep small crops legible: degrade in proportion to size
    if rng.random() < 0.6:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.2) * rel + 0.2)
    if rng.random() < 0.2:  # horizontal motion blur
        k = int(rng.integers(3, 4 + int(4 * rel)))
        ker = np.zeros((k, k), np.float32)
        ker[k // 2] = 1.0 / k
        img = cv2.filter2D(img, -1, ker)
    if rng.random() < 0.35:
        img = interlace_comb(img, rng, shift=int(rng.integers(1, 2 + int(3 * rel))))
    img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.75, 1.25), beta=rng.uniform(-25, 25))
    img = jpeg(img, int(rng.integers(35, 92)))
    return img, text
