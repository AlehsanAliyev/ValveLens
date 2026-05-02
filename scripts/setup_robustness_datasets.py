from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

from robustness_utils import REPO_ROOT, ensure_dir


ROBUSTNESS_DIRS = [
    "data/robustness/synthetic",
    "data/robustness/preprocessed",
    "artifacts/robustness",
    "artifacts/robustness/preprocessing_preview",
    "data_sources/downloads/exdark",
    "data_sources/extracted/exdark",
]


def _kaggle_configured() -> bool:
    if shutil.which("kaggle") is None:
        return False
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def _download_exdark() -> Dict[str, str]:
    extracted = REPO_ROOT / "data_sources" / "extracted" / "exdark"
    if not _kaggle_configured():
        return {
            "status": "manual_required",
            "message": (
                "Kaggle CLI is not available or not configured. "
                "Manual download: kaggle datasets download -d "
                "washingtongold/exdark-dataset -p data_sources/downloads/exdark --unzip"
            ),
        }

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        "washingtongold/exdark-dataset",
        "-p",
        str(extracted),
        "--unzip",
    ]
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
    except Exception as exc:
        return {"status": "failed", "message": str(exc)}

    if result.returncode != 0:
        return {
            "status": "failed",
            "message": (result.stderr or result.stdout or "Kaggle download failed").strip(),
        }
    return {"status": "downloaded", "message": str(extracted)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare local folders for ValveLens robustness experiments."
    )
    parser.add_argument(
        "--download-exdark",
        action="store_true",
        help="Try to download ExDARK with the Kaggle CLI if credentials are configured.",
    )
    args = parser.parse_args()

    created: List[str] = []
    for item in ROBUSTNESS_DIRS:
        path = ensure_dir(item)
        created.append(str(path.relative_to(REPO_ROOT)))

    test_images = REPO_ROOT / "data" / "detection" / "combined" / "test" / "images"
    extracted_root = REPO_ROOT / "data_sources" / "extracted"
    openloris_present = [
        name
        for name in ("corridor", "office", "station")
        if (extracted_root / name).exists()
    ]
    nyc_present = (extracted_root / "indoor_anony").exists()
    exdark_root = extracted_root / "exdark"
    exdark_has_files = exdark_root.exists() and any(exdark_root.iterdir())

    if args.download_exdark:
        exdark_status = _download_exdark()
    elif exdark_has_files:
        exdark_status = {"status": "present", "message": str(exdark_root)}
    else:
        exdark_status = {
            "status": "manual_required",
            "message": (
                "ExDARK is optional and not present. To add it manually, configure "
                "Kaggle and run: kaggle datasets download -d washingtongold/exdark-dataset "
                "-p data_sources/downloads/exdark --unzip"
            ),
        }

    print("Created/verified folders:")
    for item in created:
        print(f"  - {item}")
    print(f"Valve/Gauge test images: {'found' if test_images.exists() else 'missing'}")
    print(f"NYC-Indoor-VPR extracted: {'found' if nyc_present else 'missing'}")
    print(
        "OpenLORIS extracted folders: "
        + (", ".join(openloris_present) if openloris_present else "missing")
    )
    print(f"ExDARK status: {exdark_status['status']}")
    print(exdark_status["message"])


if __name__ == "__main__":
    main()
