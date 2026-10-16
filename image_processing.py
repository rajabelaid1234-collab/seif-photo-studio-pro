
from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw

MAX_SIDE = 3000
MAX_PIXELS = 25_000_000
PROJECT_VERSION = 1

# Portrait-focused models first: usually better for people and fine hair.
# If a model or its optional alpha-matting dependencies fail, remove_background
# automatically falls back to ordinary rembg removal and then the next model.
MODEL_CHAIN = [
    "birefnet-portrait",
    "u2net_human_seg",
    "isnet-general-use",
    "u2netp",
]

OFFICIAL_SIZES = {
    "Photo officielle 35x45 mm": (413, 531),
    "Photo identité 30x40 mm": (354, 472),
    "Photo carrée": (600, 600),
    "Portrait 4x6 pouces": (1200, 1800),
}


def load_image(source: Any, max_side: int = MAX_SIDE) -> Image.Image:
    """Load an image safely, correct its orientation and limit its size."""
    if isinstance(source, Image.Image):
        image = source.copy()
    else:
        image = Image.open(source)

    image = ImageOps.exif_transpose(image)

    if image.width * image.height > MAX_PIXELS:
        raise ValueError("Image too large. Please use a smaller image.")

    image = image.convert("RGBA")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if not re.fullmatch(r"[0-9a-fA-F]{6}", value):
        raise ValueError("Invalid color. Use a value such as #FFFFFF.")
    return tuple(bytes.fromhex(value))


def safe_name(name: str, default: str = "photo") -> str:
    name = Path(str(name)).stem
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")
    return name[:80] or default


def image_to_bytes(
    image: Image.Image, format: str = "PNG", quality: int = 95
) -> bytes:
    """Export a PIL image as PNG, JPEG or WebP; JPEG composites transparency on white."""
    buffer = io.BytesIO()
    fmt = format.upper()

    if fmt in ("JPG", "JPEG"):
        background = Image.new("RGB", image.size, "white")
        rgba = image.convert("RGBA")
        background.paste(rgba, mask=rgba.getchannel("A"))
        background.save(buffer, "JPEG", quality=quality, optimize=True)
    elif fmt == "WEBP":
        image.save(buffer, "WEBP", quality=quality, method=4)
    elif fmt == "PNG":
        image.save(buffer, "PNG", optimize=True)
    else:
        raise ValueError("Unsupported format. Use PNG, JPEG or WEBP.")

    return buffer.getvalue()


def composite_background(
    foreground: Image.Image, background: Image.Image | str
) -> Image.Image:
    """Place an RGBA foreground over an image or solid hexadecimal color."""
    fg = foreground.convert("RGBA")

    if isinstance(background, str):
        bg = Image.new("RGBA", fg.size, (*hex_to_rgb(background), 255))
    else:
        bg = background.convert("RGBA").resize(
            fg.size, Image.Resampling.LANCZOS
        )

    bg.alpha_composite(fg)
    return bg


def remove_background(
    image: Image.Image, model_name: str | None = None
) -> Image.Image:
    """
    Remove a background with a portrait-first model chain.

    First try alpha matting, which can recover softer hair edges. If alpha
    matting or its optional dependencies fail, fall back to standard rembg
    removal for that model, then try the next model.
    """
    from rembg import new_session, remove

    source = image.convert("RGBA")
    errors = []
    models = [model_name] if model_name else MODEL_CHAIN

    for model in models:
        if not model:
            continue
        try:
            session = new_session(model)
            # Hair-friendly alpha matting. Some environments do not have
            # pymatting installed, so this attempt must never block fallback.
            try:
                result = remove(
                    source,
                    session=session,
                    alpha_matting=True,
                    alpha_matting_foreground_threshold=240,
                    alpha_matting_background_threshold=10,
                    alpha_matting_erode_size=10,
                )
                result = result.convert("RGBA")
                if result.getchannel("A").getextrema() == (255, 255):
                    # Avoid returning a fully opaque image if a backend did
                    # not produce a usable transparency mask.
                    raise RuntimeError("Model returned no transparent mask")
                return result
            except Exception as matting_error:
                errors.append(f"{model} alpha matting: {matting_error}")
                result = remove(source, session=session).convert("RGBA")
                return result
        except Exception as exc:
            errors.append(f"{model}: {exc}")

    raise RuntimeError(
        "Background removal failed. Check model downloads and dependencies. "
        + " | ".join(errors)
    )

def refine_mask(
    mask: Image.Image,
    blur: float = 0.5,
    threshold: int = 0,
) -> Image.Image:
    result = mask.convert("L")
    if threshold:
        result = result.point(
            lambda value: 255 if value >= threshold else 0
        )
    if blur > 0:
        result = result.filter(ImageFilter.GaussianBlur(blur))
    return result


def enhance_image(
    image: Image.Image,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    sharpness: float = 1.0,
) -> Image.Image:
    """Apply conservative photo adjustments."""
    result = image.copy()
    result = ImageEnhance.Brightness(result).enhance(
        max(0.0, brightness)
    )
    result = ImageEnhance.Contrast(result).enhance(
        max(0.0, contrast)
    )
    result = ImageEnhance.Color(result).enhance(
        max(0.0, saturation)
    )
    result = ImageEnhance.Sharpness(result).enhance(
        max(0.0, sharpness)
    )
    return result


def fix_lighting(image: Image.Image, amount: float = 1.08) -> Image.Image:
    """A gentle brightness correction, without changing facial features."""
    return ImageEnhance.Brightness(image).enhance(
        min(1.5, max(0.5, amount))
    )


def fit_background(
    background: Image.Image,
    size: tuple[int, int],
    mode: str = "cover",
) -> Image.Image:
    background = background.convert("RGBA")

    if mode == "contain":
        background.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", size, (255, 255, 255, 255))
        canvas.alpha_composite(
            background,
            ((size[0] - background.width) // 2,
             (size[1] - background.height) // 2),
        )
        return canvas

    return ImageOps.fit(
        background, size, method=Image.Resampling.LANCZOS
    ).convert("RGBA")


def create_solid_background(
    size: tuple[int, int], color: str = "#FFFFFF"
) -> Image.Image:
    return Image.new("RGBA", size, (*hex_to_rgb(color), 255))


def create_gradient_background(
    size: tuple[int, int],
    color1: str = "#FFFFFF",
    color2: str = "#D9DDE3",
    horizontal: bool = False,
) -> Image.Image:
    c1 = np.array(hex_to_rgb(color1), dtype=float)
    c2 = np.array(hex_to_rgb(color2), dtype=float)
    width, height = size

    steps = width if horizontal else height
    gradient = np.linspace(0, 1, max(1, steps))[:, None]
    colors = (c1 + (c2 - c1) * gradient).clip(0, 255).astype(np.uint8)

    if horizontal:
        arr = np.tile(colors[None, :, :], (height, 1, 1))
    else:
        arr = np.tile(colors[:, None, :], (1, width, 1))

    alpha = np.full((height, width, 1), 255, dtype=np.uint8)
    return Image.fromarray(np.concatenate((arr, alpha), axis=2), "RGBA")


def official_crop(
    image: Image.Image,
    size: tuple[int, int] = (413, 531),
    position: tuple[float, float] = (0.5, 0.5),
) -> Image.Image:
    """Crop to a portrait aspect ratio using a normalized focal position."""
    width, height = size
    ratio = width / height
    src = image.convert("RGBA")
    sw, sh = src.size

    if sw / sh > ratio:
        crop_h = sh
        crop_w = int(sh * ratio)
    else:
        crop_w = sw
        crop_h = int(sw / ratio)

    cx = int(min(1, max(0, position[0])) * sw)
    cy = int(min(1, max(0, position[1])) * sh)

    left = max(0, min(sw - crop_w, cx - crop_w // 2))
    top = max(0, min(sh - crop_h, cy - crop_h // 2))

    cropped = src.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize(size, Image.Resampling.LANCZOS)


def build_official_photo(
    person: Image.Image,
    background_color: str = "#FFFFFF",
    size: tuple[int, int] = (413, 531),
    brightness: float = 1.0,
) -> Image.Image:
    """Build a simple official portrait from an already-cut-out person."""
    subject = fix_lighting(person.convert("RGBA"), brightness)
    canvas = create_solid_background(size, background_color)

    subject.thumbnail(
        (int(size[0] * 0.92), int(size[1] * 0.92)),
        Image.Resampling.LANCZOS,
    )

    x = (size[0] - subject.width) // 2
    y = size[1] - subject.height
    canvas.alpha_composite(subject, (x, y))

    return canvas.convert("RGB")


def make_print_sheet(
    image: Image.Image,
    paper_size: tuple[int, int] = (2480, 3508),
    photo_size: tuple[int, int] = (413, 531),
    margin: int = 100,
    gap: int = 30,
) -> Image.Image:
    """Arrange repeated photos on a white print sheet."""
    photo = image.convert("RGB").resize(
        photo_size, Image.Resampling.LANCZOS
    )
    sheet = Image.new("RGB", paper_size, "white")

    cols = max(0, (paper_size[0] - 2 * margin + gap) //
               (photo_size[0] + gap))
    rows = max(0, (paper_size[1] - 2 * margin + gap) //
               (photo_size[1] + gap))

    for row in range(rows):
        for col in range(cols):
            x = margin + col * (photo_size[0] + gap)
            y = margin + row * (photo_size[1] + gap)
            sheet.paste(photo, (x, y))

    return sheet


class History:
    """Small undo/redo history for image or mask states."""

    def __init__(self, initial: Any = None, limit: int = 30):
        self.limit = max(2, limit)
        self.states = []
        self.index = -1
        if initial is not None:
            self.push(initial)

    @staticmethod
    def _copy(value: Any) -> Any:
        return value.copy() if hasattr(value, "copy") else value

    def push(self, value: Any) -> None:
        self.states = self.states[:self.index + 1]
        self.states.append(self._copy(value))
        self.states = self.states[-self.limit:]
        self.index = len(self.states) - 1

    def undo(self) -> Any:
        if self.index > 0:
            self.index -= 1
        return self.current

    def redo(self) -> Any:
        if self.index < len(self.states) - 1:
            self.index += 1
        return self.current

    @property
    def current(self) -> Any:
        if self.index < 0:
            return None
        return self._copy(self.states[self.index])

    @property
    def can_undo(self) -> bool:
        return self.index > 0

    @property
    def can_redo(self) -> bool:
        return 0 <= self.index < len(self.states) - 1


def brush_mask(
    mask: Image.Image,
    x: int,
    y: int,
    radius: int = 15,
    mode: str = "erase",
) -> Image.Image:
    """Paint or erase a circular area on a grayscale mask."""
    result = mask.convert("L")
    draw = ImageDraw.Draw(result)
    value = 0 if mode == "erase" else 255
    if mode not in ("erase", "add", "restore"):
        raise ValueError("mode must be erase, add or restore")
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=value,
    )
    return result


def inpaint_image(
    image: Image.Image,
    mask: Image.Image,
    radius: int = 3,
) -> Image.Image:
    """Remove marked objects using OpenCV inpainting."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Install opencv-python-headless to use inpainting."
        ) from exc

    rgb = np.array(image.convert("RGB"))
    mask_array = np.array(mask.convert("L"))
    result = cv2.inpaint(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        mask_array,
        max(1, radius),
        cv2.INPAINT_TELEA,
    )
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


def save_project(
    path: str | Path,
    images: dict[str, Image.Image],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save images and metadata in a ZIP project."""
    path = Path(path)
    manifest = {
        "version": PROJECT_VERSION,
        "metadata": metadata or {},
        "images": [],
    }

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, image in images.items():
            filename = safe_name(name) + ".png"
            key = f"images/{filename}"
            archive.writestr(key, image_to_bytes(image, "PNG"))
            manifest["images"].append({"name": name, "file": key})

        archive.writestr(
            "project.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )


def load_project(
    path: str | Path,
) -> tuple[dict[str, Image.Image], dict[str, Any]]:
    """Load a ZIP project without extracting files to disk."""
    images = {}

    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()

        if "project.json" not in names:
            raise ValueError("Invalid project: project.json is missing.")

        manifest = json.loads(archive.read("project.json"))
        if manifest.get("version") != PROJECT_VERSION:
            raise ValueError("Unsupported project version.")

        for item in manifest.get("images", []):
            filename = item.get("file", "")
            if filename not in names or not filename.startswith("images/"):
                raise ValueError("Invalid image path in project.")
            if ".." in Path(filename).parts:
                raise ValueError("Unsafe path in project.")

            data = archive.read(filename)
            images[item["name"]] = load_image(io.BytesIO(data))

    return images, manifest.get("metadata", {})


def brush_edit(
    cutout: Image.Image,
    original: Image.Image,
    stroke_mask: Image.Image,
    restore: bool,
) -> Image.Image:
    """Restore or erase the pixels covered by a painted grayscale stroke mask.

    stroke_mask: white pixels indicate the area painted by the user.
    Restore uses original RGB pixels and makes them opaque; erase makes them transparent.
    """
    result = cutout.convert("RGBA").copy()
    mask = stroke_mask.convert("L").resize(result.size, Image.Resampling.NEAREST)
    alpha = result.getchannel("A")
    if restore:
        source = original.convert("RGBA").resize(result.size, Image.Resampling.LANCZOS)
        result = Image.composite(source, result, mask)
        new_alpha = Image.composite(Image.new("L", result.size, 255), alpha, mask)
    else:
        new_alpha = Image.composite(Image.new("L", result.size, 0), alpha, mask)
    result.putalpha(new_alpha)
    return result
