import argparse
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from PIL import Image

from app import db
from app.embeddings import Embedder
from app.faiss_store import FaissIndex
from app.ocr import OCRReader, match_enrolled_device_id
from app.pipeline import InferencePipeline


def _status_line(ok: bool, label: str, detail: str = "") -> str:
    prefix = "PASS" if ok else "FAIL"
    suffix = f" - {detail}" if detail else ""
    return f"[{prefix}] {label}{suffix}"


def _check_counts(embedder_dim: int) -> Dict[str, int]:
    counts = db.fetch_counts()
    device_index = FaissIndex("devices", embedder_dim)
    device_index.load()
    return {
        "devices_count": int(counts.get("devices", 0)),
        "device_refs_count": int(counts.get("device_refs", 0)),
        "device_faiss_size": len(device_index.meta),
    }


def _check_ocr(
    image_path: Optional[str],
    expected_id: Optional[str],
    enrolled_ids: List[str],
) -> Optional[Dict]:
    if not image_path:
        return None
    path = Path(image_path)
    if not path.exists():
        return {"ok": False, "detail": f"image not found: {path}"}

    reader = OCRReader()
    result = reader.read(Image.open(path).convert("RGB"))
    text = result.get("text") or ""
    matched_id = match_enrolled_device_id(text, enrolled_ids)
    expected_ok = not expected_id or matched_id == expected_id
    ok = bool(matched_id) and expected_ok
    return {
        "ok": ok,
        "text": text,
        "conf": float(result.get("conf") or 0.0),
        "matched_id": matched_id,
        "expected_id": expected_id,
        "detail": f"text={text!r}, matched={matched_id}, expected={expected_id}",
    }


def _check_reid(
    image_path: Optional[str],
    expected_id: Optional[str],
    embedder: Embedder,
    topk: int,
) -> Optional[Dict]:
    if not image_path:
        return None
    path = Path(image_path)
    if not path.exists():
        return {"ok": False, "detail": f"image not found: {path}"}

    index = FaissIndex("devices", embedder.dim)
    index.load()
    if not index.meta:
        return {"ok": False, "detail": "device FAISS index is empty"}

    image = Image.open(path).convert("RGB")
    vec = embedder.embed_image(image)
    matches = index.search(vec, topk=max(1, topk))
    compact = [
        {"device_id": meta.get("device_id"), "score": float(score)}
        for meta, score in matches
    ]
    top1 = compact[0]["device_id"] if compact else None
    expected_ok = not expected_id or top1 == expected_id
    return {
        "ok": bool(top1) and expected_ok,
        "top1": top1,
        "expected_id": expected_id,
        "matches": compact,
        "detail": f"top1={top1}, expected={expected_id}, matches={compact[:topk]}",
    }


def _check_inference(image_path: Optional[str], expected_id: Optional[str]) -> Optional[Dict]:
    if not image_path:
        return None
    path = Path(image_path)
    if not path.exists():
        return {"ok": False, "detail": f"image not found: {path}"}

    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        return {"ok": False, "detail": f"could not read image: {path}"}

    pipeline = InferencePipeline()
    response = pipeline.process_frame(
        frame,
        input_type="image",
        source=path.name,
        session_id="identity-demo-validation",
    )
    selected = response.decision.selected_device
    selected_id = selected.device_id if selected else None
    expected_ok = not expected_id or selected_id == expected_id
    ok = response.decision.status == "ACCEPTED" and bool(selected_id) and expected_ok
    return {
        "ok": ok,
        "status": response.decision.status,
        "selected_id": selected_id,
        "expected_id": expected_id,
        "action": response.decision.action,
        "message": response.decision.message,
        "detail": (
            f"status={response.decision.status}, selected={selected_id}, "
            f"expected={expected_id}, action={response.decision.action}"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate whether local data is ready for a v0.3 identity demo."
    )
    parser.add_argument("--ocr-image", default=None, help="Optional OCR tag test image.")
    parser.add_argument("--ocr-expected-id", default=None, help="Expected OCR device ID.")
    parser.add_argument("--reid-image", default=None, help="Optional ReID test image.")
    parser.add_argument("--reid-expected-id", default=None, help="Expected ReID top-1 ID.")
    parser.add_argument("--infer-image", default=None, help="Optional end-to-end inference image.")
    parser.add_argument("--infer-expected-id", default=None, help="Expected accepted device ID.")
    parser.add_argument("--topk", type=int, default=5, help="ReID top-k matches to print.")
    args = parser.parse_args()

    db.init_db()
    embedder = Embedder()
    counts = _check_counts(embedder.dim)
    enrolled_ids = db.fetch_device_ids()

    checks = [
        (
            counts["devices_count"] > 0,
            "devices_count > 0",
            str(counts["devices_count"]),
        ),
        (
            counts["device_refs_count"] > 0,
            "device_refs_count > 0",
            str(counts["device_refs_count"]),
        ),
        (
            counts["device_faiss_size"] > 0,
            "device_faiss_size > 0",
            str(counts["device_faiss_size"]),
        ),
    ]

    print("ValveLens v0.3 identity demo validation")
    for ok, label, detail in checks:
        print(_status_line(ok, label, detail))

    ocr_result = _check_ocr(args.ocr_image, args.ocr_expected_id, enrolled_ids)
    if ocr_result is not None:
        print(_status_line(bool(ocr_result["ok"]), "OCR enrolled-ID match", ocr_result["detail"]))

    reid_result = _check_reid(
        args.reid_image,
        args.reid_expected_id,
        embedder,
        args.topk,
    )
    if reid_result is not None:
        print(_status_line(bool(reid_result["ok"]), "ReID top match", reid_result["detail"]))

    infer_result = _check_inference(args.infer_image, args.infer_expected_id)
    if infer_result is not None:
        print(_status_line(bool(infer_result["ok"]), "end-to-end ACCEPTED decision", infer_result["detail"]))

    all_required_ok = all(ok for ok, _, _ in checks)
    optional_results = [ocr_result, reid_result, infer_result]
    optional_ok = all(bool(item["ok"]) for item in optional_results if item is not None)
    print(_status_line(all_required_ok and optional_ok, "overall identity demo readiness"))


if __name__ == "__main__":
    main()
