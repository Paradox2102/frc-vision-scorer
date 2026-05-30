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
        # WPILib field coordinates use +Y across the field. For the observer HUD,
        # draw +Y upward on screen so the map matches the main broadcast view.
        return (xy[0] * self.s, (self.W - xy[1]) * self.s)

    def render(self, points) -> np.ndarray:
        img = np.full((self.h, self.w, 3), 235, np.uint8)
        self._draw_field(img)
        for xy, side in points:
            u, v = self.field_to_pixel(xy)
            cv2.circle(img, (int(u), int(v)), 8, (20, 20, 20), -1)
            cv2.circle(img, (int(u), int(v)), 6, _TINT.get(side), -1)
        return img

    def _px(self, x, y):
        return int(round(x * self.s)), int(round(y * self.s))

    def _draw_field(self, img):
        # Carpet and perimeter.
        cv2.rectangle(img, (0, 0), (self.w - 1, self.h - 1), (80, 80, 80), 2)
        cv2.line(img, (self.w // 2, 0), (self.w // 2, self.h), (170, 170, 170), 2)

        # Alliance-side field elements approximated from the 2026 REBUILT field
        # layout. This intentionally omits loose fuel balls.
        for side, color, x0 in (
            ("blue", (210, 95, 20), self.L * 0.26),
            ("red", (25, 25, 210), self.L * 0.74),
        ):
            # Long protected/structure lane.
            lane_w = self.L * 0.055
            p1 = self._px(x0 - lane_w / 2, self.W * 0.18)
            p2 = self._px(x0 + lane_w / 2, self.W * 0.82)
            cv2.rectangle(img, p1, p2, color, -1)
            cv2.rectangle(img, p1, p2, tuple(max(0, c - 55) for c in color), 2)

            # Hub ring / central scoring structure.
            hub_x = self.L * (0.275 if side == "blue" else 0.725)
            hub_y = self.W * 0.50
            center = self._px(hub_x, hub_y)
            radius = max(10, int(self.s * 0.42))
            pts = []
            for k in range(8):
                a = np.pi / 8 + k * np.pi / 4
                pts.append((int(center[0] + radius * np.cos(a)),
                            int(center[1] + radius * np.sin(a))))
            cv2.polylines(img, [np.array(pts, np.int32)], True, (90, 90, 90), 3)
            overlay = img.copy()
            cv2.fillPoly(overlay, [np.array(pts, np.int32)], color)
            cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)

            # Side source/loading area blocks.
            src_x = self.L * (0.04 if side == "blue" else 0.96)
            src_y = self.W * (0.14 if side == "blue" else 0.86)
            sx, sy = self._px(src_x, src_y)
            cv2.rectangle(img, (sx - 18, sy - 22), (sx + 18, sy + 22), color, 3)

        cv2.putText(img, "BLUE", self._px(0.25, 0.45), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (210, 95, 20), 1, cv2.LINE_AA)
        cv2.putText(img, "RED", self._px(self.L - 1.0, 0.45), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (25, 25, 210), 1, cv2.LINE_AA)
