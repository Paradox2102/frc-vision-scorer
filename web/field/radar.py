"""Top-down field radar: render field + plot (x,y) detections."""
from __future__ import annotations
import cv2
import numpy as np

_TINT = {"blue": (255, 120, 0), "red": (0, 0, 255), None: (180, 180, 180)}

class Radar:
    def __init__(self, field_length: float, field_width: float, px_per_m: float = 40):
        self.L, self.W, self.s = field_length, field_width, px_per_m
        self.w = round(field_length * px_per_m)
        self.h = round(field_width * px_per_m)

    def field_to_pixel(self, xy) -> tuple:
        return (xy[0] * self.s, xy[1] * self.s)

    def render(self, points) -> np.ndarray:
        img = np.full((self.h, self.w, 3), 30, np.uint8)
        cv2.line(img, (self.w // 2, 0), (self.w // 2, self.h), (90, 90, 90), 1)
        cv2.rectangle(img, (0, 0), (self.w - 1, self.h - 1), (90, 90, 90), 1)
        for xy, side in points:
            u, v = self.field_to_pixel(xy)
            cv2.circle(img, (int(u), int(v)), 6, _TINT.get(side), -1)
        return img
