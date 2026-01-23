import argparse
import json
from pathlib import Path

from app.datasets.downloader import download_file, extract_archive, verify_archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data_sources/manifests/cold_sequences.json",
        help="Path to COLD sequences manifest",
    )
    parser.add_argument("--max_sequences", type=int, default=2)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    downloads_dir = Path("data_sources/downloads")
    extracted_root = Path("data_sources/extracted/cold")
    downloads_dir.mkdir(parents=True, exist_ok=True)
    extracted_root.mkdir(parents=True, exist_ok=True)

    count = 0
    for entry in entries:
        if count >= args.max_sequences:
            break
        name = entry.get("name")
        url = entry.get("url")
        if not name or not url or "PASTE_SEQUENCE_URL_HERE" in url:
            print(f"Skipping entry with missing URL: {name}")
            continue

        suffix = ".tar.gz"
        if url.endswith(".zip"):
            suffix = ".zip"
        elif url.endswith(".tgz"):
            suffix = ".tgz"

        out_path = downloads_dir / f"{name}{suffix}"
        extract_dir = extracted_root / name
        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(f"Already extracted: {extract_dir}")
            count += 1
            continue

        if not out_path.exists() or out_path.stat().st_size == 0:
            ok = download_file(url, out_path)
            if not ok:
                continue

        if not verify_archive(out_path):
            print(f"Invalid archive: {out_path}")
            continue

        extracted = extract_archive(out_path, extract_dir)
        if extracted:
            print(f"Extracted to {extract_dir}")
            count += 1


if __name__ == "__main__":
    main()
