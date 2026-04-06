from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "detection_training"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO detector on a specified split.")
    parser.add_argument("--weights", required=True, help="Path to trained weights file.")
    parser.add_argument("--data", required=True, help="Path to Ultralytics dataset YAML.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="valvelens_eval")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("WANDB_DISABLED", "true")

    args = parse_args()
    from ultralytics import YOLO, settings  # type: ignore

    settings.update({"wandb": False})
    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    summary = {
        "weights": str(Path(args.weights).resolve()),
        "data": str(Path(args.data).resolve()),
        "split": args.split,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class_maps": [float(v) for v in metrics.box.maps],
        "results_dict": {str(k): float(v) for k, v in metrics.results_dict.items()},
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / f"{args.name}_{args.split}_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
