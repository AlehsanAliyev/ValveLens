import argparse
import csv
from pathlib import Path
from typing import Any, Optional


def _to_float(v: Any) -> Optional[float]:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize exported ValveLens metrics CSV.")
    parser.add_argument("--in", dest="in_path", required=True, help="Path to metrics.csv")
    args = parser.parse_args()

    path = Path(args.in_path)
    if not path.exists():
        raise SystemExit(f"Metrics file not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No rows in metrics file.")
        return

    prompts_vals = [_to_float(r.get("prompts")) for r in rows]
    prompts_vals = [v for v in prompts_vals if v is not None]

    frame_vals = [_to_float(r.get("frames_to_accept")) for r in rows]
    frame_vals = [v for v in frame_vals if v is not None]

    wrong_rows = [r for r in rows if r.get("wrong_selection") in {"1", "true", "True"}]
    wrong_den = [
        r
        for r in rows
        if (r.get("gt_device") or "").strip() and (r.get("accepted_device") or "").strip()
    ]

    avg_prompts = sum(prompts_vals) / len(prompts_vals) if prompts_vals else 0.0
    avg_frames = sum(frame_vals) / len(frame_vals) if frame_vals else 0.0
    wrong_rate = (len(wrong_rows) / len(wrong_den)) if wrong_den else 0.0

    print(f"Tasks: {len(rows)}")
    print(f"Wrong-device rate: {wrong_rate:.4f} ({len(wrong_rows)}/{len(wrong_den)})")
    print(f"Avg prompts/task: {avg_prompts:.3f}")
    print(f"Avg time-to-accept (frames): {avg_frames:.3f}")


if __name__ == "__main__":
    main()
