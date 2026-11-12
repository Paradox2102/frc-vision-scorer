"""2025hop corner camera calibrations + the operator's saved clicks, copied from
the real versions (load_versions("2025hop", ...), Nov 11 2026). Tests never read
web/uploads; these are the frozen values.

corner_left  20261111-232802-934489: own lens, fx 740, k1 -0.537, rms 1.65 px
corner_right 20261111-235213-551517: own lens over-fitted from 5 tags, fx 1047,
             k1 -1.195, centre x 7.4 (wrong side of the barge), rms 6.90 px
corner_right 20261111-235354-580753: corner_left's lens borrowed, pose only, rms 12.4 px
"""
import numpy as np

from web.field.field_model import FieldModel

SEASON25 = "2025-reefscape-welded"

HOP_LEFT_SIZE = (965, 347)
HOP_LEFT = {"K": [[739.967700055877, 0.0, 482.5], [0.0, 739.967700055877, 173.5],
                  [0.0, 0.0, 1.0]],
            "D": [-0.5373262887776691, 0.0, 0.0, 0.0, 0.0],
            "rvec": [1.3426460943393659, 1.3867128809160423, -1.0255371138805007],
            "tvec": [-4.251873273284275, -0.22737555519031116, 9.097189557677899],
            "image_size": [965, 347], "crop_region": [0, 0, 965, 347],
            "field_season": SEASON25, "rms_error": 1.6495825784268812, "mode": "pinhole",
            "H": None, "masks": [], "robot_track_top_crop": 0.0}
HOP_LEFT_VID = "20261111-232802-934489"
HOP_LEFT_AUTO = {"12": [[191.64, 62.94], [202.75, 61.6], [202.82, 49.19], [191.42, 50.31]],
                 "13": [[742.72, 92.44], [753.24, 95.32], [754.02, 82.31], [743.49, 79.68]]}
HOP_LEFT_MANUAL = [{"id": 22, "u": 345, "v": 305}, {"id": 20, "u": 580, "v": 320},
                   {"id": 21, "u": 460, "v": 339}]
HOP_LEFT_SOURCE = {"auto_tags": HOP_LEFT_AUTO, "manual_tags": HOP_LEFT_MANUAL,
                   "holdout_ids": [], "timestamp": 125.8, "match_key": "2025hop_qm1",
                   "crop": [0, 733, 965, 347]}

HOP_RIGHT_SIZE = (916, 347)
# corner_right 20261111-235213-551517 (own lens, over-fitted)
HOP_RIGHT_OWN = {"K": [[1046.7759845354103, 0.0, 458.0], [0.0, 1046.7759845354103, 173.5],
                       [0.0, 0.0, 1.0]],
                 "D": [-1.1953016128069358, 0.0, 0.0, 0.0, 0.0],
                 "rvec": [1.347907921995034, -1.4394069838355343, 1.1555257083117023],
                 "tvec": [4.8171901154897645, 4.080822791130419, -6.256367062959997],
                 "image_size": [916, 347], "crop_region": [0, 0, 916, 347],
                 "field_season": SEASON25, "rms_error": 6.899384879932897,
                 "mode": "pinhole", "H": None, "masks": [], "robot_track_top_crop": 0.0}
# corner_right 20261111-235354-580753 (corner_left's lens, pose only)
HOP_RIGHT_BORROWED = {"K": [[739.967700055877, 0.0, 458.0], [0.0, 739.967700055877, 173.5],
                            [0.0, 0.0, 1.0]],
                      "D": [-0.5373262887776691, 0.0, 0.0, 0.0, 0.0],
                      "rvec": [1.3540680661208735, -1.4102433621013863, 1.1138014455069039],
                      "tvec": [4.336340103070344, 3.909185910250252, -8.43535555535214],
                      "image_size": [916, 347], "crop_region": [0, 0, 916, 347],
                      "field_season": SEASON25, "rms_error": 12.372298584228169,
                      "mode": "pinhole", "H": None, "masks": [],
                      "robot_track_top_crop": 0.0}
HOP_RIGHT_AUTO = {"1": [[725.12, 59.5], [737.16, 61.92], [737.89, 47.69], [725.68, 45.37]]}
# the clicks of 235213 / 235354 (the operator re-clicked after 233034)
HOP_RIGHT_MANUAL2 = [{"id": 2, "u": 119, "v": 60}, {"id": 11, "u": 554, "v": 325},
                     {"id": 9, "u": 295, "v": 336}, {"id": 10, "u": 423, "v": 342}]
HOP_RIGHT_SOURCE = {"auto_tags": HOP_RIGHT_AUTO, "manual_tags": HOP_RIGHT_MANUAL2,
                    "holdout_ids": [], "timestamp": 21.3, "match_key": "2025hop_qm1",
                    "crop": [1004, 733, 916, 347]}


def corr_from_source(source, fm=None):
    """{key: (img (N,2), obj (N,3))} from a version source, as solve-tags builds it:
    auto tags -> 4 corners ("a<id>"), manual clicks -> the tag centre ("m<id>")."""
    fm = fm or FieldModel.load(SEASON25)
    corr = {f"a{t}": (np.array(c, float), fm.tag_corners_3d(int(t)))
            for t, c in source["auto_tags"].items()}
    for m in source["manual_tags"]:
        corr[f"m{m['id']}"] = (np.array([[m["u"], m["v"]]], float),
                               fm.tag_corners_3d(int(m["id"])).mean(axis=0).reshape(1, 3))
    return corr
