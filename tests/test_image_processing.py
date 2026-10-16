from __future__ import annotations

import io
import sys
import types

import pytest
from PIL import Image

import image_processing as ip


def test_load_image_rejects_corrupt_file():
    with pytest.raises((OSError, ValueError)):
        ip.load_image(io.BytesIO(b"this is not an image"))


def test_remove_background_returns_transparency(monkeypatch):
    fake = types.ModuleType("rembg")
    fake.new_session = lambda name: object()

    def fake_remove(source, session=None, **kwargs):
        result = source.convert("RGBA")
        alpha = Image.new("L", result.size, 255)
        alpha.putpixel((0, 0), 0)
        result.putalpha(alpha)
        return result

    fake.remove = fake_remove
    monkeypatch.setitem(sys.modules, "rembg", fake)
    image = Image.new("RGB", (8, 8), "red")
    result = ip.remove_background(image, model_name="test-model")
    assert result.mode == "RGBA"
    assert result.getchannel("A").getpixel((0, 0)) == 0
    assert result.getchannel("A").getpixel((4, 4)) == 255


def test_composite_background_keeps_foreground_dimensions():
    foreground = Image.new("RGBA", (17, 13), (255, 0, 0, 128))
    background = Image.new("RGB", (40, 30), "white")
    result = ip.composite_background(foreground, background)
    assert result.size == foreground.size
    assert result.mode == "RGBA"


def test_official_sizes_have_positive_dimensions_and_expected_output():
    assert ip.OFFICIAL_SIZES
    for size in ip.OFFICIAL_SIZES.values():
        assert len(size) == 2
        assert all(isinstance(value, int) and value > 0 for value in size)
        result = ip.official_crop(Image.new("RGBA", (100, 140), "blue"), size=size)
        assert result.size == size


@pytest.mark.parametrize("fmt,expected_format", [
    ("PNG", "PNG"), ("JPEG", "JPEG"), ("WebP", "WEBP")
])
def test_image_export_formats(fmt, expected_format):
    image = Image.new("RGBA", (12, 10), (20, 100, 220, 120))
    payload = ip.image_to_bytes(image, fmt)
    with Image.open(io.BytesIO(payload)) as decoded:
        assert decoded.format == expected_format
        assert decoded.size == image.size


def test_brush_edit_can_erase_and_restore_pixels():
    original = Image.new("RGBA", (20, 20), (255, 0, 0, 255))
    cutout = original.copy()
    cutout.putalpha(Image.new("L", cutout.size, 255))
    mask = Image.new("L", cutout.size, 0)
    mask_draw = __import__("PIL.ImageDraw", fromlist=["Draw"]).Draw(mask)
    mask_draw.rectangle((5, 5, 10, 10), fill=255)
    erased = ip.brush_edit(cutout, original, mask, restore=False)
    assert erased.getchannel("A").getpixel((7, 7)) == 0
    restored = ip.brush_edit(erased, original, mask, restore=True)
    assert restored.getchannel("A").getpixel((7, 7)) == 255
