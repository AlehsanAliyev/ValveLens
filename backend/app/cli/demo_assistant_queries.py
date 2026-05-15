import argparse
import csv
import json
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app import db
from app.routes.ask import AskRequest, ask


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "artifacts" / "v05_assistant_demo"
DEFAULT_TARGET_DEVICE_ID = "V-1023"
BASE_QUESTIONS = [
    "What is this?",
    "Where is {target_device_id}?",
    "Why are you uncertain?",
    "What should I do next?",
    "Which devices are visible?",
    "What tag did you read?",
]


def _safe_json_load(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _latest_observation_id(session_id: Optional[str] = None) -> Optional[str]:
    rows = db.fetch_observations()
    if session_id:
        filtered = []
        for row in rows:
            payload = _safe_json_load(row.get("payload_json"))
            source = ((payload.get("input") or {}).get("source")) or row.get("source_name")
            if source == session_id:
                filtered.append(row)
        rows = filtered
    if not rows:
        return None
    return rows[-1].get("obs_id")


def _post_ask(
    backend_url: str,
    question: str,
    observation_id: str,
    session_id: Optional[str],
    selected_detection_id: Optional[str],
    use_vlm: bool,
) -> Dict[str, Any]:
    payload = {
        "question": question,
        "observation_id": observation_id,
        "session_id": session_id,
        "selected_detection_id": selected_detection_id,
        "use_vlm": use_vlm,
    }
    req = urllib.request.Request(
        f"{backend_url.rstrip('/')}/ask",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return {
            "answer": f"Ask request failed: {exc}",
            "confidence": 0.0,
            "mode": "error",
            "evidence_used": [],
            "recommended_next_action": "Check backend URL and observation ID.",
            "uncertainty_reason": str(exc),
        }


def _local_ask(
    question: str,
    observation_id: str,
    session_id: Optional[str],
    selected_detection_id: Optional[str],
    use_vlm: bool,
) -> Dict[str, Any]:
    return ask(
        AskRequest(
            question=question,
            observation_id=observation_id,
            session_id=session_id,
            selected_detection_id=selected_detection_id,
            use_vlm=use_vlm,
        )
    )


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "observation_id",
        "question",
        "answer",
        "confidence",
        "mode",
        "evidence_used",
        "recommended_next_action",
        "uncertainty_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    summary = payload["summary"]
    rows = payload["questions"]
    lines = [
        "# ValveLens v0.5 Assistant Demo Report",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Summary",
        "",
        f"- observation_ids: `{', '.join(summary.get('observation_ids') or [str(summary.get('observation_id'))])}`",
        f"- backend_url: `{summary.get('backend_url') or 'local route call'}`",
        f"- questions_tested: `{summary.get('questions_tested')}`",
        f"- vlm_requested: `{summary.get('vlm_requested')}`",
        f"- vlm_modes_seen: `{', '.join(summary.get('modes_seen') or [])}`",
        "",
        "## Example Answers",
        "",
    ]
    for row in rows:
        vlm_status = (row.get("raw_response") or {}).get("vlm_status")
        lines.extend(
            [
                f"### {row['question']}",
                "",
                f"- mode: `{row.get('mode')}`",
                f"- confidence: `{row.get('confidence')}`",
                f"- evidence_used: `{', '.join(row.get('evidence_used') or [])}`",
                f"- recommended_next_action: `{row.get('recommended_next_action')}`",
                f"- uncertainty_reason: `{row.get('uncertainty_reason') or ''}`",
                f"- vlm_status: `{vlm_status or ''}`",
                "",
                row.get("answer") or "",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_thesis_section(path: Path, payload: Dict[str, Any]) -> None:
    rows = payload.get("questions") or []
    examples = []
    for row in rows[:6]:
        examples.extend(
            [
                f"### {row.get('question')}",
                "",
                row.get("answer") or "",
                "",
                f"Evidence used: `{', '.join(row.get('evidence_used') or [])}`.",
                "",
            ]
        )
    lines = [
        "# Thesis Section: Evidence-Aware Interactive Assistant",
        "",
        "## Evidence-aware assistant design",
        "",
        "ValveLens v0.5.1 adds an interactive assistant layer on top of the perception pipeline. The assistant does not replace the detector, OCR, ReID, fusion module, or uncertainty policy. It converts the latest stored inference observation into compact structured evidence and answers operator questions from that evidence.",
        "",
        "The evidence object includes zone candidates, detections, image-space boxes, detector confidence, OCR text, parsed device IDs, ReID matches, fused identity, final decision status, uncertainty reasons, image-quality diagnostics, and optional user selection context.",
        "",
        "## Rule-based mode",
        "",
        "The default assistant is deterministic and rule-based. This is the thesis/demo reliability path because each answer can be traced to explicit fields in the observation: decision, detections, OCR, ReID, quality, and zone evidence. When evidence is weak, the assistant reports uncertainty and returns the same next action produced by the ValveLens policy.",
        "",
        "## VLM-gated mode",
        "",
        "A VLM pathway exists as a scaffold, but provider execution is gated by `backend/app/config.yaml` and environment credentials. The default configuration keeps `assistant.enable_vlm: false`. If VLM use is requested without a configured provider, the route falls back to rule-based answers and records the VLM status in the response.",
        "",
        "## Why the VLM does not replace perception",
        "",
        "A VLM is allowed only to explain ValveLens evidence in natural language. It must not invent device IDs, override an `UNCERTAIN` policy decision, claim calibrated confidence, or identify devices outside the enrolled evidence. The source of truth remains the perception stack: zone retrieval, detector, OCR, ReID, fusion, and policy.",
        "",
        "## Example questions and answers",
        "",
        *examples,
        "## Limitations",
        "",
        "- Location is image-relative unless the scene has calibrated facility coordinates.",
        "- VLM provider execution is disabled by default and has not been used as a source of perception evidence.",
        "- Identity acceptance is validated on a controlled proxy benchmark, not on real repeated physical device photos.",
        "- OCR remains condition-sensitive and depends on a working local OCR backend.",
        "- ReID is retrieval-based and depends on representative enrolled references.",
        "",
        "Safe thesis claim: ValveLens implements an evidence-aware assistant interface that explains current perception results and uncertainty from structured backend evidence.",
        "",
        "Unsafe thesis claim: the assistant should not be described as a blind visual-language model that independently recognizes industrial devices from raw images.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed v0.5 assistant demo questions.")
    parser.add_argument("--observation-id", default=None)
    parser.add_argument("--observation-ids", nargs="*", default=None)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--selected-detection-id", default=None)
    parser.add_argument("--backend-url", default=None)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--use-vlm", action="store_true")
    parser.add_argument("--target-device-id", default=DEFAULT_TARGET_DEVICE_ID)
    parser.add_argument("--include-candidates", action="store_true")
    parser.add_argument("--questions", nargs="*", default=None)
    args = parser.parse_args()

    db.init_db()
    observation_ids = args.observation_ids or ([args.observation_id] if args.observation_id else [])
    if not observation_ids:
        latest = _latest_observation_id(args.session_id)
        if latest:
            observation_ids = [latest]
    if not observation_ids:
        raise SystemExit("No observation found. Run inference first or pass --observation-id.")

    questions = args.questions or [
        item.format(target_device_id=args.target_device_id) for item in BASE_QUESTIONS
    ]
    if args.include_candidates:
        questions.append("What are the top candidates?")

    rows: List[Dict[str, Any]] = []
    for observation_id in observation_ids:
        for question in questions:
            if args.backend_url:
                response = _post_ask(
                    args.backend_url,
                    question,
                    observation_id,
                    args.session_id,
                    args.selected_detection_id,
                    args.use_vlm,
                )
            else:
                response = _local_ask(
                    question,
                    observation_id,
                    args.session_id,
                    args.selected_detection_id,
                    args.use_vlm,
                )
            rows.append(
                {
                    "observation_id": observation_id,
                    "question": question,
                    "answer": response.get("answer"),
                    "confidence": response.get("confidence"),
                    "mode": response.get("mode"),
                    "evidence_used": response.get("evidence_used") or [],
                    "recommended_next_action": response.get("recommended_next_action"),
                    "uncertainty_reason": response.get("uncertainty_reason"),
                    "raw_response": response,
                }
            )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "observation_id": observation_ids[0] if len(observation_ids) == 1 else None,
            "observation_ids": observation_ids,
            "session_id": args.session_id,
            "selected_detection_id": args.selected_detection_id,
            "backend_url": args.backend_url,
            "questions_tested": len(rows),
            "vlm_requested": args.use_vlm,
            "modes_seen": sorted({str(row.get("mode")) for row in rows if row.get("mode")}),
        },
        "questions": rows,
    }
    (out_dir / "assistant_demo_report.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    _write_csv(out_dir / "example_questions.csv", rows)
    _write_markdown(out_dir / "assistant_demo_report.md", payload)
    _write_thesis_section(out_dir / "thesis_assistant_section.md", payload)

    print(f"Observations: {', '.join(observation_ids)}")
    print(f"Report: {out_dir / 'assistant_demo_report.md'}")
    print(f"JSON: {out_dir / 'assistant_demo_report.json'}")
    print(f"CSV: {out_dir / 'example_questions.csv'}")
    print(f"Thesis section: {out_dir / 'thesis_assistant_section.md'}")


if __name__ == "__main__":
    main()
