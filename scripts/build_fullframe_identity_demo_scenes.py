from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image, ImageDraw, ImageFilter


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERIES = REPO_ROOT / "data" / "device_benchmark" / "queries_manifest.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "device_benchmark" / "fullframe_demo"
DEFAULT_ARTIFACTS = REPO_ROOT / "artifacts" / "v03_demo"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class QueryRow:
    image_path: Path
    expected_device_id: str
    expected_type: str
    condition: str
    tag_visible: bool
    expected_zone: str


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("/", "\\")
    except Exception:
        return str(path).replace("/", "\\")


def resolve_image(raw_path: str, manifest_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidates = [
        REPO_ROOT / path,
        manifest_path.parent / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def load_queries(path: Path) -> List[QueryRow]:
    rows: List[QueryRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_path = resolve_image(row.get("image_path", ""), path)
            if not image_path.exists() or image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            rows.append(
                QueryRow(
                    image_path=image_path,
                    expected_device_id=(row.get("expected_device_id") or "").strip(),
                    expected_type=(row.get("expected_type") or "").strip(),
                    condition=(row.get("condition") or "").strip(),
                    tag_visible=str(row.get("tag_visible") or "").strip().lower() in {"1", "true", "yes"},
                    expected_zone=(row.get("expected_zone") or "").strip(),
                )
            )
    return rows


def choose_queries(rows: List[QueryRow], limit: int) -> List[QueryRow]:
    preferred_ids = ["V-1023", "V-1212", "PG-45"]
    selected: List[QueryRow] = []
    for device_id in preferred_ids:
        candidates = [
            row for row in rows
            if row.expected_device_id == device_id and row.condition in {"clean", "low_light"} and row.tag_visible
        ]
        selected.extend(candidates[:2])
    if len(selected) < limit:
        for row in rows:
            if row in selected:
                continue
            if row.condition in {"clean", "low_light"}:
                selected.append(row)
            if len(selected) >= limit:
                break
    return selected[:limit]


def draw_industrial_background(width: int, height: int, seed: int) -> Image.Image:
    rng = random.Random(seed)
    bg = Image.new("RGB", (width, height), (52, 58, 62))
    draw = ImageDraw.Draw(bg)
    for y in range(0, height, 80):
        shade = 46 + (y // 80) % 3 * 8
        draw.rectangle((0, y, width, y + 42), fill=(shade, shade + 5, shade + 8))
    for _ in range(10):
        x = rng.randint(-60, width - 100)
        y = rng.randint(30, height - 70)
        pipe_w = rng.randint(22, 44)
        color = rng.choice([(104, 112, 114), (126, 121, 108), (88, 101, 110)])
        draw.rounded_rectangle((x, y, x + width // rng.randint(3, 5), y + pipe_w), radius=8, fill=color)
        draw.line((x, y + pipe_w // 2, x + width // 3, y + pipe_w // 2), fill=(170, 174, 166), width=2)
    for _ in range(8):
        x = rng.randint(20, width - 160)
        y = rng.randint(20, height - 120)
        draw.rectangle((x, y, x + rng.randint(60, 150), y + rng.randint(40, 120)), outline=(80, 88, 92), width=3)
    return bg.filter(ImageFilter.GaussianBlur(radius=0.25))


def fit_crop(crop: Image.Image, max_w: int, max_h: int) -> Image.Image:
    crop = crop.convert("RGB")
    scale = min(max_w / crop.width, max_h / crop.height)
    scale = min(max(scale, 0.85), 2.2)
    new_size = (max(1, int(crop.width * scale)), max(1, int(crop.height * scale)))
    return crop.resize(new_size, Image.Resampling.LANCZOS)


def make_scene(row: QueryRow, index: int, out_dir: Path, seed: int) -> Dict[str, str]:
    width, height = 1280, 800
    bg = draw_industrial_background(width, height, seed + index)
    crop = Image.open(row.image_path).convert("RGB")
    crop = fit_crop(crop, max_w=560, max_h=520)
    rng = random.Random(seed + 1000 + index)
    x = rng.randint(280, max(281, width - crop.width - 140))
    y = rng.randint(120, max(121, height - crop.height - 100))

    shadow = Image.new("RGBA", crop.size, (0, 0, 0, 120)).filter(ImageFilter.GaussianBlur(radius=14))
    bg.paste(shadow.convert("RGB"), (x + 12, y + 14))
    bg.paste(crop, (x, y))

    draw = ImageDraw.Draw(bg)
    draw.rectangle((x - 4, y - 4, x + crop.width + 4, y + crop.height + 4), outline=(204, 210, 204), width=2)
    draw.text((24, 22), f"Zone demo frame | expected {row.expected_device_id}", fill=(220, 226, 220))

    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"scene_{index:03d}_{row.expected_device_id}_{row.condition}.jpg"
    out_path = out_dir / name
    bg.save(out_path, quality=94)
    return {
        "image_path": rel(out_path),
        "expected_device_id": row.expected_device_id,
        "expected_type": row.expected_type,
        "condition": "fullframe_" + row.condition,
        "tag_visible": str(row.tag_visible).lower(),
        "expected_zone": row.expected_zone,
        "source_crop": rel(row.image_path),
    }


def write_manifest(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "expected_device_id", "expected_type", "condition", "tag_visible", "expected_zone"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def make_preview(rows: List[Dict[str, str]], out_path: Path) -> None:
    thumbs = []
    for row in rows[:12]:
        image = Image.open(REPO_ROOT / row["image_path"]).convert("RGB")
        image.thumbnail((360, 240))
        tile = Image.new("RGB", (360, 280), (240, 240, 240))
        tile.paste(image, ((360 - image.width) // 2, 0))
        draw = ImageDraw.Draw(tile)
        draw.text((8, 248), f"{row['expected_device_id']} | {row['condition']}", fill=(20, 20, 20))
        thumbs.append(tile)
    if not thumbs:
        return
    cols = 3
    rows_count = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 360, rows_count * 280), (255, 255, 255))
    for idx, tile in enumerate(thumbs):
        sheet.paste(tile, ((idx % cols) * 360, (idx // cols) * 280))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full-frame proxy identity demo scenes for v0.3 API validation.")
    parser.add_argument("--queries-manifest", default=str(DEFAULT_QUERIES))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.queries_manifest).resolve()
    out_dir = Path(args.out).resolve()
    artifact_dir = Path(args.artifacts).resolve()
    if out_dir.exists() and any(out_dir.rglob("*")) and not args.overwrite:
        raise SystemExit(f"Output already exists: {out_dir}. Use --overwrite.")
    if args.overwrite and out_dir.exists():
        for file in out_dir.rglob("*"):
            if file.is_file():
                file.unlink()
    rows = choose_queries(load_queries(manifest_path), args.limit)
    generated = [make_scene(row, idx + 1, out_dir, args.seed) for idx, row in enumerate(rows)]
    scene_manifest = out_dir / "fullframe_manifest.csv"
    write_manifest(scene_manifest, generated)
    metadata_path = artifact_dir / "fullframe_identity_demo_metadata.json"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps({"scenes": generated}, indent=2), encoding="utf-8")
    preview_path = artifact_dir / "fullframe_identity_demo_preview.jpg"
    make_preview(generated, preview_path)

    print("Full-frame identity demo scenes generated.")
    print(f"  scenes: {len(generated)}")
    print(f"  output: {rel(out_dir)}")
    print(f"  manifest: {rel(scene_manifest)}")
    print(f"  preview: {rel(preview_path)}")
    print(f"  metadata: {rel(metadata_path)}")


if __name__ == "__main__":
    main()
