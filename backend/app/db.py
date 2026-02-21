import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

import numpy as np


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "valvelens.db"


def _utc_now() -> str:
    return datetime.utcnow().isoformat()


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(row["name"]) for row in rows}


def _ensure_column(
    conn: sqlite3.Connection, table: str, column: str, definition_sql: str
) -> None:
    columns = _table_columns(conn, table)
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition_sql}")


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS zones (
            zone_id TEXT PRIMARY KEY,
            name TEXT,
            description TEXT,
            created_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS zone_keyframes (
            keyframe_id TEXT PRIMARY KEY,
            zone_id TEXT,
            image_path TEXT,
            embedding_type TEXT,
            embedding BLOB,
            created_at TEXT,
            FOREIGN KEY(zone_id) REFERENCES zones(zone_id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS devices (
            device_id TEXT PRIMARY KEY,
            zone_id TEXT,
            device_type TEXT,
            description TEXT,
            created_at TEXT,
            FOREIGN KEY(zone_id) REFERENCES zones(zone_id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS device_refs (
            ref_id TEXT PRIMARY KEY,
            device_id TEXT,
            image_path TEXT,
            embedding_type TEXT,
            embedding BLOB,
            created_at TEXT,
            FOREIGN KEY(device_id) REFERENCES devices(device_id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS observations (
            obs_id TEXT PRIMARY KEY,
            created_at TEXT,
            input_type TEXT,
            source_name TEXT,
            zone_top1 TEXT,
            zone_conf REAL,
            final_device_id TEXT,
            final_conf REAL,
            policy_action TEXT,
            payload_json TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            fb_id TEXT PRIMARY KEY,
            created_at TEXT,
            obs_id TEXT,
            session_id TEXT,
            feedback_type TEXT,
            data_json TEXT,
            FOREIGN KEY(obs_id) REFERENCES observations(obs_id)
        )
        """
    )
    _ensure_column(conn, "feedback", "session_id", "TEXT")
    conn.commit()
    conn.close()


def create_zone(name: str, description: str) -> str:
    zone_id = str(uuid4())
    conn = get_conn()
    conn.execute(
        "INSERT INTO zones (zone_id, name, description, created_at) VALUES (?, ?, ?, ?)",
        (zone_id, name, description, _utc_now()),
    )
    conn.commit()
    conn.close()
    return zone_id


def get_zone_by_name(name: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM zones WHERE name = ?", (name,)).fetchone()
    conn.close()
    return dict(row) if row else None


def fetch_zone_name_map() -> Dict[str, str]:
    conn = get_conn()
    rows = conn.execute("SELECT zone_id, name FROM zones").fetchall()
    conn.close()
    return {row["zone_id"]: row["name"] for row in rows}


def zone_keyframe_exists(image_path: str) -> bool:
    conn = get_conn()
    row = conn.execute(
        "SELECT 1 FROM zone_keyframes WHERE image_path = ?", (image_path,)
    ).fetchone()
    conn.close()
    return row is not None


def add_zone_keyframe(
    zone_id: str, image_path: str, embedding_type: str, embedding: bytes
) -> str:
    keyframe_id = str(uuid4())
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO zone_keyframes
        (keyframe_id, zone_id, image_path, embedding_type, embedding, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (keyframe_id, zone_id, image_path, embedding_type, embedding, _utc_now()),
    )
    conn.commit()
    conn.close()
    return keyframe_id


def create_device(
    device_id: str, zone_id: str, device_type: str, description: str
) -> str:
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO devices (device_id, zone_id, device_type, description, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (device_id, zone_id, device_type, description, _utc_now()),
    )
    conn.commit()
    conn.close()
    return device_id


def add_device_ref(
    device_id: str, image_path: str, embedding_type: str, embedding: bytes
) -> str:
    ref_id = str(uuid4())
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO device_refs
        (ref_id, device_id, image_path, embedding_type, embedding, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (ref_id, device_id, image_path, embedding_type, embedding, _utc_now()),
    )
    conn.commit()
    conn.close()
    return ref_id


def insert_observation(payload: Dict[str, Any]) -> None:
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO observations
        (obs_id, created_at, input_type, source_name, zone_top1, zone_conf,
         final_device_id, final_conf, policy_action, payload_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload["obs_id"],
            payload["created_at"],
            payload["input_type"],
            payload["source_name"],
            payload["zone_top1"],
            payload["zone_conf"],
            payload["final_device_id"],
            payload["final_conf"],
            payload["policy_action"],
            json.dumps(payload["payload_json"], default=_json_default),
        ),
    )
    conn.commit()
    conn.close()


def insert_feedback(
    obs_id: str,
    feedback_type: str,
    data_json: Dict[str, Any],
    session_id: Optional[str] = None,
) -> str:
    fb_id = str(uuid4())
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO feedback
        (fb_id, created_at, obs_id, session_id, feedback_type, data_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (fb_id, _utc_now(), obs_id, session_id, feedback_type, json.dumps(data_json)),
    )
    conn.commit()
    conn.close()
    return fb_id


def fetch_zone_keyframes() -> List[Dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM zone_keyframes").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def fetch_device_refs() -> List[Dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT device_refs.*, devices.zone_id as zone_id
        FROM device_refs
        LEFT JOIN devices ON device_refs.device_id = devices.device_id
        """
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def fetch_device_ids() -> List[str]:
    conn = get_conn()
    rows = conn.execute("SELECT device_id FROM devices").fetchall()
    conn.close()
    return [row["device_id"] for row in rows]


def get_device(device_id: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM devices WHERE device_id = ?", (device_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def fetch_counts() -> Dict[str, int]:
    conn = get_conn()
    cur = conn.cursor()
    counts = {}
    for table in ["zones", "zone_keyframes", "devices", "device_refs", "observations", "feedback"]:
        cur.execute(f"SELECT COUNT(*) as count FROM {table}")
        counts[table] = int(cur.fetchone()["count"])
    conn.close()
    return counts


def count_zones() -> int:
    conn = get_conn()
    row = conn.execute("SELECT COUNT(*) AS c FROM zones").fetchone()
    conn.close()
    return int(row["c"]) if row else 0


def count_zone_keyframes() -> int:
    conn = get_conn()
    row = conn.execute("SELECT COUNT(*) AS c FROM zone_keyframes").fetchone()
    conn.close()
    return int(row["c"]) if row else 0


def fetch_observation(obs_id: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM observations WHERE obs_id = ?", (obs_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def update_observation(
    obs_id: str,
    payload_json: Dict[str, Any],
    final_device_id: Optional[str],
    final_conf: Optional[float],
    policy_action: Optional[str],
) -> None:
    conn = get_conn()
    conn.execute(
        """
        UPDATE observations
        SET payload_json = ?, final_device_id = ?, final_conf = ?, policy_action = ?
        WHERE obs_id = ?
        """,
        (
            json.dumps(payload_json, default=_json_default),
            final_device_id,
            final_conf,
            policy_action,
            obs_id,
        ),
    )
    conn.commit()
    conn.close()


def fetch_observations() -> List[Dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT *
        FROM observations
        ORDER BY datetime(created_at) ASC, created_at ASC
        """
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def fetch_feedback_rows() -> List[Dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT *
        FROM feedback
        ORDER BY datetime(created_at) ASC, created_at ASC
        """
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]
