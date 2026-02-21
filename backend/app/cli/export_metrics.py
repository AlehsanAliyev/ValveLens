import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app import db


def _safe_json_load(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _to_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _parse_time(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _derive_session_id(obs: Dict[str, Any]) -> str:
    source_name = obs.get("source_name")
    if source_name:
        return str(source_name)
    payload = _safe_json_load(obs.get("payload_json"))
    source = ((payload.get("input") or {}).get("source")) if payload else None
    return str(source or "unknown_session")


def _extract_frame_index(obs: Dict[str, Any]) -> Optional[int]:
    payload = _safe_json_load(obs.get("payload_json"))
    frame_index = ((payload.get("input") or {}).get("frame_index")) if payload else None
    return _to_int(frame_index)


def _extract_zone(obs: Dict[str, Any]) -> tuple[Optional[str], Optional[float]]:
    zone_top1 = obs.get("zone_top1")
    zone_conf = _to_float(obs.get("zone_conf"))
    if zone_top1 is not None and zone_conf is not None:
        return str(zone_top1), float(zone_conf)
    payload = _safe_json_load(obs.get("payload_json"))
    zone = payload.get("zone") or {}
    top1 = zone.get("top1") or {}
    return (
        str(top1.get("zone_id")) if top1.get("zone_id") else None,
        _to_float(top1.get("score")),
    )


def _extract_selected(obs: Dict[str, Any]) -> tuple[Optional[str], Optional[float]]:
    if obs.get("final_device_id"):
        return str(obs.get("final_device_id")), _to_float(obs.get("final_conf"))
    payload = _safe_json_load(obs.get("payload_json"))
    decision = payload.get("decision") or {}
    selected = decision.get("selected_device") or {}
    return (
        str(selected.get("device_id")) if selected.get("device_id") else None,
        _to_float(selected.get("score")),
    )


def _extract_policy_action(obs: Dict[str, Any]) -> str:
    if obs.get("policy_action"):
        return str(obs.get("policy_action"))
    payload = _safe_json_load(obs.get("payload_json"))
    decision = payload.get("decision") or {}
    return str(decision.get("action") or "NONE")


def _is_accepted(obs: Dict[str, Any]) -> bool:
    device_id, _ = _extract_selected(obs)
    if device_id:
        return True
    payload = _safe_json_load(obs.get("payload_json"))
    decision = payload.get("decision") or {}
    return str(decision.get("status") or "").upper() == "ACCEPTED"


def _load_gt(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    data = _safe_json_load(path.read_text(encoding="utf-8"))
    out: Dict[str, str] = {}
    for k, v in data.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k] = v
    return out


def _final_action_for_task(
    accepted_obs: Optional[Dict[str, Any]],
    task_obs: List[Dict[str, Any]],
    feedback_map: Dict[str, List[Dict[str, Any]]],
) -> str:
    terminal_obs = accepted_obs or task_obs[-1]
    terminal_obs_id = str(terminal_obs.get("obs_id"))
    feedback_events = feedback_map.get(terminal_obs_id, [])
    if feedback_events:
        return str(feedback_events[-1].get("feedback_type", "NONE")).upper()
    return _extract_policy_action(terminal_obs)


def _build_task_row(
    session_id: str,
    task_index: int,
    task_obs: List[Dict[str, Any]],
    feedback_map: Dict[str, List[Dict[str, Any]]],
    gt_map: Dict[str, str],
) -> Dict[str, Any]:
    accepted_obs = next((obs for obs in task_obs if _is_accepted(obs)), None)
    terminal_obs = accepted_obs or task_obs[-1]
    accepted_device, selected_conf = _extract_selected(terminal_obs)
    zone_top1, zone_conf = _extract_zone(terminal_obs)
    policy_action = _extract_policy_action(terminal_obs)

    prompts = sum(1 for obs in task_obs if _extract_policy_action(obs) != "NONE")

    first_frame = next(
        (_extract_frame_index(obs) for obs in task_obs if _extract_frame_index(obs) is not None),
        None,
    )
    accepted_frame = _extract_frame_index(accepted_obs) if accepted_obs else None
    if accepted_obs:
        if first_frame is not None and accepted_frame is not None:
            frames_to_accept: Any = max(0, accepted_frame - first_frame)
        else:
            frames_to_accept = max(0, len(task_obs) - 1)
    else:
        frames_to_accept = ""

    first_time = _parse_time(task_obs[0].get("created_at"))
    accepted_time = _parse_time(accepted_obs.get("created_at")) if accepted_obs else None
    if first_time and accepted_time:
        time_to_accept_seconds: Any = max(
            0.0, (accepted_time - first_time).total_seconds()
        )
    else:
        time_to_accept_seconds = ""

    gt_device = gt_map.get(session_id, "")
    wrong_selection = (
        int(bool(gt_device and accepted_device and accepted_device != gt_device))
        if accepted_device
        else ""
    )

    feedback_events = []
    for obs in task_obs:
        feedback_events.extend(
            [
                str(item.get("feedback_type"))
                for item in feedback_map.get(str(obs.get("obs_id")), [])
                if item.get("feedback_type")
            ]
        )

    return {
        "session_id": session_id,
        "task_id": f"{session_id}_task_{task_index:03d}",
        "accepted_device": accepted_device or "",
        "gt_device": gt_device,
        "action": _final_action_for_task(accepted_obs, task_obs, feedback_map),
        "prompts": prompts,
        "frames_to_accept": frames_to_accept,
        "wrong_selection": wrong_selection,
        "zone_top1": zone_top1 or "",
        "zone_conf": zone_conf if zone_conf is not None else "",
        "selected_conf": selected_conf if selected_conf is not None else "",
        "policy_action": policy_action,
        "time_to_accept_seconds": time_to_accept_seconds,
        "feedback_events": "|".join(feedback_events),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export interaction-aware experiment metrics from sqlite."
    )
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument(
        "--gt",
        default=str(Path("backend/data/gt_sessions.json")),
        help="Optional session ground-truth mapping JSON file",
    )
    args = parser.parse_args()

    db.init_db()
    observations = db.fetch_observations()
    feedback_rows = db.fetch_feedback_rows()
    gt_map = _load_gt(Path(args.gt))

    feedback_by_obs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in feedback_rows:
        feedback_by_obs[str(row.get("obs_id"))].append(row)

    obs_by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for obs in observations:
        session_id = _derive_session_id(obs)
        obs_by_session[session_id].append(obs)

    rows: List[Dict[str, Any]] = []
    for session_id, session_obs in sorted(obs_by_session.items(), key=lambda x: x[0]):
        sorted_session = sorted(
            session_obs,
            key=lambda obs: (
                _parse_time(obs.get("created_at")) or datetime.min,
                _extract_frame_index(obs) if _extract_frame_index(obs) is not None else -1,
                str(obs.get("obs_id")),
            ),
        )
        start = 0
        task_index = 1
        while start < len(sorted_session):
            end = start
            while end < len(sorted_session) and not _is_accepted(sorted_session[end]):
                end += 1
            if end >= len(sorted_session):
                task_obs = sorted_session[start:]
                if task_obs:
                    rows.append(
                        _build_task_row(
                            session_id, task_index, task_obs, feedback_by_obs, gt_map
                        )
                    )
                break
            task_obs = sorted_session[start : end + 1]
            rows.append(
                _build_task_row(
                    session_id, task_index, task_obs, feedback_by_obs, gt_map
                )
            )
            task_index += 1
            start = end + 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "session_id",
        "task_id",
        "accepted_device",
        "gt_device",
        "action",
        "prompts",
        "frames_to_accept",
        "wrong_selection",
        "zone_top1",
        "zone_conf",
        "selected_conf",
        "policy_action",
        "time_to_accept_seconds",
        "feedback_events",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} task rows to {out_path}")


if __name__ == "__main__":
    main()
