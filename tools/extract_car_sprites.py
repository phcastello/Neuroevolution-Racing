#!/usr/bin/env python3
"""Extract the bundled vehicle sheet into normalized transparent PNG sprites."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as error:
    raise SystemExit(
        "Pillow is required. Install it with: python -m pip install Pillow"
    ) from error


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sprites" / "spriteCarros.jpg"
OUTPUT_DIR = ROOT / "assets" / "cars"
CANVAS_SIZE = (280, 150)
PADDING = 6

# Source-space crop anchors supplied with the project brief.
# Car 06 is intentionally omitted because its source artwork is defective.
CAR_BOUNDS = (
    (1, (83, 90, 223, 168)),
    (2, (313, 90, 452, 166)),
    (3, (542, 85, 705, 178)),
    (4, (83, 232, 223, 305)),
    (5, (312, 228, 452, 303)),
    (7, (82, 368, 220, 444)),
    (8, (313, 371, 447, 445)),
    (9, (542, 365, 680, 446)),
    (10, (82, 504, 225, 589)),
    (11, (313, 508, 455, 583)),
    (12, (542, 507, 730, 599)),
)


def is_background_candidate(pixel: tuple[int, int, int]) -> bool:
    """Accept neutral, light pixels only; connectivity protects white car paint."""
    red, green, blue = pixel
    return min(pixel) >= 165 and max(pixel) - min(pixel) <= 35


def remove_border_connected_background(image: Image.Image) -> Image.Image:
    """Remove border-connected near-white background and source shadows."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    connected = bytearray(width * height)
    queue: deque[tuple[int, int]] = deque()

    def enqueue(x: int, y: int) -> None:
        offset = y * width + x
        if not connected[offset] and is_background_candidate(pixels[x, y]):
            connected[offset] = 1
            queue.append((x, y))

    for x in range(width):
        enqueue(x, 0)
        enqueue(x, height - 1)
    for y in range(height):
        enqueue(0, y)
        enqueue(width - 1, y)

    while queue:
        x, y = queue.popleft()
        if x > 0:
            enqueue(x - 1, y)
        if x + 1 < width:
            enqueue(x + 1, y)
        if y > 0:
            enqueue(x, y - 1)
        if y + 1 < height:
            enqueue(x, y + 1)

    rgba = rgb.convert("RGBA")
    output = rgba.load()
    for y in range(height):
        for x in range(width):
            if not connected[y * width + x]:
                continue
            output[x, y] = (0, 0, 0, 0)
    return rgba


def normalize_sprite(image: Image.Image) -> Image.Image:
    alpha = image.getchannel("A")
    # Ignore faint source shadows/residue when locating the vehicle body.
    alpha_bounds = alpha.point(lambda value: 255 if value >= 128 else 0).getbbox()
    if alpha_bounds is None:
        raise ValueError("background removal produced an empty sprite")
    alpha_bounds = (
        max(0, alpha_bounds[0] - 2),
        max(0, alpha_bounds[1] - 2),
        min(image.width, alpha_bounds[2] + 2),
        min(image.height, alpha_bounds[3] + 2),
    )
    # The complete canvas maps to the fixed physical footprint at runtime, so
    # filling it keeps the visible vehicle aligned with the 28x15 hitbox.
    return image.crop(alpha_bounds).resize(CANVAS_SIZE, Image.Resampling.LANCZOS)


def extract_sprites() -> list[Path]:
    source = Image.open(SOURCE).convert("RGB")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for car_number, (x0, y0, x1, y1) in CAR_BOUNDS:
        crop_box = (
            max(0, x0 - PADDING),
            max(0, y0 - PADDING),
            min(source.width, x1 + PADDING),
            min(source.height, y1 + PADDING),
        )
        sprite = normalize_sprite(remove_border_connected_background(source.crop(crop_box)))
        output = OUTPUT_DIR / f"car_{car_number:02}.png"
        sprite.save(output, format="PNG", optimize=True)
        outputs.append(output)
    create_contact_sheet(outputs)
    return outputs


def create_contact_sheet(outputs: list[Path]) -> None:
    columns = 3
    label_height = 24
    cell_width, cell_height = CANVAS_SIZE[0], CANVAS_SIZE[1] + label_height
    sheet = Image.new("RGBA", (columns * cell_width, 4 * cell_height), (35, 42, 46, 255))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, path in enumerate(outputs):
        column = index % columns
        row = index // columns
        origin = (column * cell_width, row * cell_height)
        sprite = Image.open(path).convert("RGBA")
        sheet.alpha_composite(sprite, origin)
        label = path.stem
        label_box = draw.textbbox((0, 0), label, font=font)
        label_width = label_box[2] - label_box[0]
        draw.text(
            (origin[0] + (cell_width - label_width) // 2, origin[1] + CANVAS_SIZE[1] + 5),
            label,
            fill=(225, 240, 245, 255),
            font=font,
        )
    sheet.save(OUTPUT_DIR / "contact_sheet.png", format="PNG", optimize=True)


def check_outputs() -> None:
    missing = []
    for car_number, _ in CAR_BOUNDS:
        path = OUTPUT_DIR / f"car_{car_number:02}.png"
        if not path.exists():
            missing.append(path.name)
            continue
        with Image.open(path) as image:
            if image.size != CANVAS_SIZE:
                raise SystemExit(f"{path}: expected {CANVAS_SIZE}, found {image.size}")
            if image.mode != "RGBA":
                raise SystemExit(f"{path}: expected RGBA, found {image.mode}")
            if image.getchannel("A").getextrema() == (255, 255):
                raise SystemExit(f"{path}: output has no transparent pixels")
    if missing:
        raise SystemExit(f"missing generated sprites: {', '.join(missing)}")
    removed = OUTPUT_DIR / "car_06.png"
    if removed.exists():
        raise SystemExit(f"{removed}: Car 06 must remain removed; Car 03 replaces it")
    print(f"Validated {len(CAR_BOUNDS)} RGBA sprites at {CANVAS_SIZE[0]}x{CANVAS_SIZE[1]}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate existing outputs instead of regenerating them",
    )
    arguments = parser.parse_args()
    if arguments.check:
        check_outputs()
        return
    outputs = extract_sprites()
    check_outputs()
    print(f"Extracted {len(outputs)} sprites from {SOURCE.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
