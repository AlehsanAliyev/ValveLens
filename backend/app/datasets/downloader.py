import shutil
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional


def download_file(url: str, out_path: Path, headers: Optional[dict] = None) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            out_path.write_bytes(resp.read())
        return out_path.exists() and out_path.stat().st_size > 0
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            print(
                f"Manual download required. Put the file here: {out_path} and re-run."
            )
            return False
        print(f"Download failed: {exc}")
        return False
    except Exception as exc:
        print(f"Download failed: {exc}")
        return False


def verify_archive(out_path: Path) -> bool:
    if not out_path.exists():
        return False
    if out_path.stat().st_size <= 0:
        return False
    suffix = "".join(out_path.suffixes).lower()
    return suffix.endswith(".zip") or suffix.endswith(".tar.gz") or suffix.endswith(".tgz")


def extract_archive(out_path: Path, out_dir: Path) -> Optional[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "".join(out_path.suffixes).lower()
    try:
        if suffix.endswith(".zip"):
            with zipfile.ZipFile(out_path, "r") as zf:
                zf.extractall(out_dir)
        elif suffix.endswith(".tar.gz") or suffix.endswith(".tgz"):
            with tarfile.open(out_path, "r:gz") as tf:
                tf.extractall(out_dir)
        else:
            print(f"Unsupported archive: {out_path}")
            return None
    except Exception as exc:
        print(f"Extraction failed: {exc}")
        return None
    return out_dir
