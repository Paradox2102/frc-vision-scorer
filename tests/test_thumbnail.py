import base64, io
from PIL import Image
from web.app import _thumbnail_bytes


def _jpeg_b64(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 30, 30)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def test_thumbnail_bounds_height_and_returns_jpeg_bytes():
    out = _thumbnail_bytes(_jpeg_b64(400, 300), max_h=128)
    assert isinstance(out, (bytes, bytearray)) and len(out) > 0
    img = Image.open(io.BytesIO(out))
    assert img.height <= 128
    assert img.format == "JPEG"


def test_thumbnail_handles_bad_input():
    assert _thumbnail_bytes(None) is None
    assert _thumbnail_bytes("not-base64!!") is None
