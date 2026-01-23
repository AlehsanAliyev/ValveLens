import argparse
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


MAGIC_SIGNATURES = [
    (b"\xFF\xD8\xFF", ".jpg"),
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"BM", ".bmp"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
]


def detect_extension(path: Path) -> str:
    try:
        with path.open("rb") as f:
            head = f.read(16)
        for sig, ext in MAGIC_SIGNATURES:
            if head.startswith(sig):
                return ext
    except Exception:
        return ""
    return ""


def is_image_readable(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def file_hash(path: Path) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def collect_files(root: Path) -> Dict[str, List[Path]]:
    layout = {}
    for split in ["train", "test", "validation"]:
        for group in ["database", "queries"]:
            folder = root / split / group
            key = f"{split}/{group}"
            if folder.exists():
                layout[key] = [p for p in folder.iterdir() if p.is_file()]
            else:
                layout[key] = []
    return layout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        required=True,
        help="Path to extracted indoor_anony folder",
    )
    parser.add_argument(
        "--copy_with_ext",
        action="store_true",
        help="Copy files without extensions into a new folder with detected extension",
    )
    parser.add_argument(
        "--out_dir",
        default="data_sources/extracted/nyc_indoor_vpr_fixed",
        help="Output folder for copied files with extensions",
    )
    parser.add_argument(
        "--hash_dupes",
        action="store_true",
        help="Compute content hashes to detect duplicates across folders",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    layout = collect_files(root)
    total = 0
    missing_ext = 0
    zero_bytes = 0
    unreadable = 0

    hashes: Dict[str, List[str]] = {}
    seen_names: Dict[str, List[str]] = {}

    for key, files in layout.items():
        print(f"{key}: {len(files)} files")
        total += len(files)

        for path in files:
            if path.stat().st_size == 0:
                zero_bytes += 1
            if path.suffix == "":
                missing_ext += 1
            if not is_image_readable(path):
                unreadable += 1

            name_key = path.name
            seen_names.setdefault(name_key, []).append(key)

            if args.hash_dupes:
                try:
                    digest = file_hash(path)
                    hashes.setdefault(digest, []).append(str(path))
                except Exception:
                    pass

            if args.copy_with_ext and path.suffix == "":
                ext = detect_extension(path)
                if ext:
                    rel = Path(key)
                    out_dir = Path(args.out_dir) / rel
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{path.name}{ext}"
                    if not out_path.exists():
                        out_path.write_bytes(path.read_bytes())

    print(f"Total files: {total}")
    print(f"Missing extensions: {missing_ext}")
    print(f"Zero-byte files: {zero_bytes}")
    print(f"Unreadable images (PIL): {unreadable}")

    dup_names = {k: v for k, v in seen_names.items() if len(v) > 1}
    print(f"Duplicate filenames across folders: {len(dup_names)}")

    if args.hash_dupes:
        dup_hashes = {k: v for k, v in hashes.items() if len(v) > 1}
        print(f"Duplicate file contents across folders (sha1): {len(dup_hashes)}")

    if args.copy_with_ext:
        print(f"Copied files with extensions to: {args.out_dir}")


if __name__ == "__main__":
    main()
