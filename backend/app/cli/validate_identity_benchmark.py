import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import request as urlrequest
from urllib.error import URLError

from PIL import Image

from app import db
from app.embeddings import Embedder
from app.faiss_store import FaissIndex
from app.ocr import OCRReader, match_enrolled_device_id


DEFAULT_OUT = Path("..") / "artifacts" / "identity_benchmark"


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _to_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _ocr_backend_available(reader: OCRReader) -> bool:
    if reader.backend == "easyocr":
        return True
    if reader.backend == "tesseract":
        return shutil.which("tesseract") is not None
    return False


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _counts(embedder_dim: int) -> Dict[str, int]:
    counts = db.fetch_counts()
    index = FaissIndex("devices", embedder_dim)
    index.load()
    return {
        "devices_count": int(counts.get("devices", 0)),
        "device_refs_count": int(counts.get("device_refs", 0)),
        "device_faiss_size": len(index.meta),
    }


def _load_queries(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "image_path",
            "expected_device_id",
            "expected_type",
            "condition",
            "tag_visible",
            "expected_zone",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                "Queries manifest is missing columns: " + ", ".join(sorted(missing))
            )
        return [dict(row) for row in reader]


def _resolve_image_path(raw_path: str, manifest_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    candidates = [
        (Path.cwd() / path).resolve(),
        (Path(__file__).resolve().parents[3] / path).resolve(),
        (manifest_path.parent / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _reid_check(
    image_path: Path,
    expected_device_id: str,
    embedder: Embedder,
    index: FaissIndex,
    topk: int,
) -> Tuple[bool, bool, Optional[str], Optional[float], List[Dict]]:
    if not index.meta:
        return False, False, None, None, []
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return False, False, None, None, []
    vec = embedder.embed_image(image)
    matches = index.search(vec, topk=max(1, topk))
    compact = [
        {"device_id": meta.get("device_id"), "score": float(score)}
        for meta, score in matches
    ]
    top1_id = compact[0]["device_id"] if compact else None
    top1_score = compact[0]["score"] if compact else None
    top1_ok = bool(expected_device_id and top1_id == expected_device_id)
    topk_ok = any(item.get("device_id") == expected_device_id for item in compact)
    return top1_ok, topk_ok, top1_id, top1_score, compact


def _ocr_check(
    image_path: Path,
    expected_device_id: str,
    enrolled_ids: List[str],
    reader: OCRReader,
) -> Tuple[bool, Optional[str], Optional[float], Optional[str]]:
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return False, None, None, None
    result = reader.read(image)
    text = result.get("text")
    matched = match_enrolled_device_id(text or "", enrolled_ids)
    return (
        bool(expected_device_id and matched == expected_device_id),
        text,
        _safe_float(result.get("conf")),
        matched,
    )


def _api_infer(
    backend_url: str,
    image_path: Path,
) -> Tuple[Optional[str], Optional[str], Optional[float], str]:
    boundary = "----ValveLensIdentityBenchmark"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{image_path.name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8")
    body += image_path.read_bytes()
    body += f"\r\n--{boundary}--\r\n".encode("utf-8")
    url = backend_url.rstrip("/") + "/infer/image"
    req = urlrequest.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urlrequest.urlopen(req, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        return None, None, None, f"api_error:{exc}"
    except Exception as exc:
        return None, None, None, f"api_error:{exc}"

    decision = payload.get("decision") or {}
    selected = decision.get("selected_device") or {}
    return (
        decision.get("status"),
        selected.get("device_id"),
        _safe_float(selected.get("score")),
        decision.get("message") or "",
    )


def _observation_decision_counts() -> Dict[str, int]:
    accepted = 0
    deferred = 0
    for row in db.fetch_observations():
        try:
            payload = json.loads(row.get("payload_json") or "{}")
        except Exception:
            payload = {}
        decision = payload.get("decision") or {}
        status = str(decision.get("status") or "").upper()
        if status == "ACCEPTED" or row.get("final_device_id"):
            accepted += 1
        elif status == "UNCERTAIN" or row.get("policy_action"):
            deferred += 1
    return {"observations_accepted_count": accepted, "observations_deferred_count": deferred}


def _failure_reasons(row: Dict, ocr_backend_available: bool) -> str:
    reasons = []
    if row["file_exists"] != "true":
        reasons.append("missing_file")
    if row["expected_device_exists"] != "true":
        reasons.append("expected_device_missing")
    if row["reid_top1_ok"] != "true":
        reasons.append("reid_top1_miss")
    if (
        ocr_backend_available
        and row["tag_visible"] == "true"
        and row["ocr_exact_match"] != "true"
    ):
        reasons.append("ocr_miss")
    if (not ocr_backend_available) and row["tag_visible"] == "true":
        reasons.append("ocr_unavailable")
    if row.get("api_decision_status") == "UNCERTAIN":
        reasons.append("decision_deferred")
    if row.get("api_selected_device") and row.get("api_selected_device") != row.get("expected_device_id"):
        reasons.append("wrong_api_selection")
    return "|".join(reasons)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a manifest-based ValveLens identity benchmark."
    )
    parser.add_argument("--queries-manifest", required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--backend-url", default=None)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    manifest_path = _resolve(args.queries_manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Queries manifest not found: {manifest_path}")

    db.init_db()
    embedder = Embedder()
    counts = _counts(embedder.dim)
    index = FaissIndex("devices", embedder.dim)
    index.load()
    enrolled_ids = db.fetch_device_ids()
    reader = OCRReader()
    ocr_backend_available = _ocr_backend_available(reader)
    queries = _load_queries(manifest_path)

    rows: List[Dict[str, Any]] = []
    for query in queries:
        image_path = _resolve_image_path(query.get("image_path", ""), manifest_path)
        expected_id = (query.get("expected_device_id") or "").strip()
        tag_visible = _to_bool(query.get("tag_visible"))
        file_exists = image_path.exists()
        expected_device_exists = bool(db.get_device(expected_id)) if expected_id else False

        reid_top1_ok = False
        reid_topk_ok = False
        reid_top1_id = None
        reid_top1_score = None
        reid_matches: List[Dict] = []
        ocr_exact_match = False
        ocr_text = None
        ocr_conf = None
        ocr_matched_id = None
        api_status = ""
        api_selected = ""
        api_score = ""
        api_message = ""

        if file_exists:
            (
                reid_top1_ok,
                reid_topk_ok,
                reid_top1_id,
                reid_top1_score,
                reid_matches,
            ) = _reid_check(image_path, expected_id, embedder, index, args.topk)
            if tag_visible and ocr_backend_available:
                ocr_exact_match, ocr_text, ocr_conf, ocr_matched_id = _ocr_check(
                    image_path,
                    expected_id,
                    enrolled_ids,
                    reader,
                )
            if args.backend_url:
                api_status, api_selected, api_score, api_message = _api_infer(
                    args.backend_url,
                    image_path,
                )

        out_row: Dict[str, Any] = {
            "image_path": str(image_path),
            "expected_device_id": expected_id,
            "expected_type": query.get("expected_type", ""),
            "condition": query.get("condition", ""),
            "tag_visible": str(tag_visible).lower(),
            "expected_zone": query.get("expected_zone", ""),
            "file_exists": str(file_exists).lower(),
            "expected_device_exists": str(expected_device_exists).lower(),
            "reid_top1_ok": str(reid_top1_ok).lower(),
            "reid_topk_ok": str(reid_topk_ok).lower(),
            "reid_top1_id": reid_top1_id or "",
            "reid_top1_score": reid_top1_score if reid_top1_score is not None else "",
            "reid_topk_matches": json.dumps(reid_matches[: args.topk]),
            "ocr_backend_available": str(ocr_backend_available).lower(),
            "ocr_attempted": str(bool(tag_visible and ocr_backend_available and file_exists)).lower(),
            "ocr_exact_match": str(ocr_exact_match).lower() if tag_visible else "",
            "ocr_text": ocr_text or "",
            "ocr_conf": ocr_conf if ocr_conf is not None else "",
            "ocr_matched_id": ocr_matched_id or "",
            "api_decision_status": api_status or "",
            "api_selected_device": api_selected or "",
            "api_selected_score": api_score if api_score is not None else "",
            "api_message": api_message or "",
        }
        out_row["failure_reasons"] = _failure_reasons(out_row, ocr_backend_available)
        rows.append(out_row)
        ocr_display: str | bool = "n/a"
        if tag_visible and not ocr_backend_available:
            ocr_display = "unavailable"
        elif tag_visible:
            ocr_display = ocr_exact_match
        print(
            f"{image_path.name}: exists={file_exists} expected={expected_id} "
            f"reid_top1={reid_top1_id} reid_topk_ok={reid_topk_ok} "
            f"ocr_ok={ocr_display}"
        )

    total = len(rows)
    missing_files = sum(1 for row in rows if row["file_exists"] != "true")
    reid_eval = [row for row in rows if row["file_exists"] == "true"]
    reid_top1_hits = sum(1 for row in reid_eval if row["reid_top1_ok"] == "true")
    reid_topk_hits = sum(1 for row in reid_eval if row["reid_topk_ok"] == "true")
    visible_tag_eval = [
        row for row in rows if row["file_exists"] == "true" and row["tag_visible"] == "true"
    ]
    ocr_eval = [row for row in visible_tag_eval if row["ocr_attempted"] == "true"]
    ocr_hits = sum(1 for row in ocr_eval if row["ocr_exact_match"] == "true")
    api_eval = [row for row in rows if row["api_decision_status"]]
    accepted = sum(1 for row in api_eval if row["api_decision_status"] == "ACCEPTED")
    deferred = sum(1 for row in api_eval if row["api_decision_status"] == "UNCERTAIN")
    api_error_count = sum(
        1
        for row in rows
        if str(row.get("api_message") or "").startswith("api_error:")
    )
    api_reason_counts = Counter(
        str(row.get("api_message") or row.get("api_decision_status") or "unknown")
        for row in api_eval
    )
    if not ocr_backend_available and visible_tag_eval:
        ocr_status = "unavailable"
        ocr_rate: Optional[float] = None
    elif ocr_eval and ocr_hits > 0:
        ocr_status = "tested_with_matches"
        ocr_rate = ocr_hits / len(ocr_eval)
    elif ocr_eval:
        ocr_status = "tested_no_matches"
        ocr_rate = 0.0
    else:
        ocr_status = "not_tested"
        ocr_rate = None

    obs_counts = _observation_decision_counts()
    summary = {
        **counts,
        "total_query_images": total,
        "missing_files": missing_files,
        "expected_devices_missing": sum(
            1 for row in rows if row["expected_device_exists"] != "true"
        ),
        "reid_top1_accuracy": reid_top1_hits / len(reid_eval) if reid_eval else 0.0,
        "reid_topk_accuracy": reid_topk_hits / len(reid_eval) if reid_eval else 0.0,
        "ocr_backend": reader.backend or "none",
        "ocr_backend_available": ocr_backend_available,
        "ocr_visible_tag_images": len(visible_tag_eval),
        "ocr_attempted_images": len(ocr_eval),
        "ocr_exact_matches": ocr_hits,
        "ocr_visible_tag_exact_match_rate": ocr_rate,
        "ocr_status": ocr_status,
        "api_tested": bool(args.backend_url),
        "accepted_count": accepted,
        "deferred_count": deferred,
        "api_evaluated_count": len(api_eval),
        "api_error_count": api_error_count,
        "at_least_one_accepted": accepted > 0,
        "top_api_decision_reasons": dict(api_reason_counts.most_common(5)),
        **obs_counts,
        "failure_reason_counts": {},
    }
    for row in rows:
        for reason in str(row.get("failure_reasons") or "").split("|"):
            if not reason:
                continue
            summary["failure_reason_counts"][reason] = (
                summary["failure_reason_counts"].get(reason, 0) + 1
            )

    out_dir = _resolve(args.out)
    json_path = out_dir / "identity_benchmark_summary.json"
    csv_path = out_dir / "identity_benchmark_summary.csv"
    _write_json(json_path, {"summary": summary, "rows": rows})
    _write_csv(csv_path, rows)

    print("\nIdentity benchmark summary:")
    print(json.dumps(summary, indent=2))
    print(f"Summary JSON: {json_path}")
    print(f"Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
