import sqlite3
from pathlib import Path


def main() -> None:
    db_path = Path("backend/data/valvelens.db")
    print("DB exists:", db_path.exists())
    if not db_path.exists():
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    print("\nTables:")
    for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        print("-", row["name"])

    print("\nCounts:")
    for table in ["zones", "zone_keyframes", "devices", "device_refs", "observations", "feedback"]:
        count = cur.execute(f"SELECT COUNT(*) as c FROM {table}").fetchone()["c"]
        print(f"{table}: {count}")

    print("\nSample zones:")
    for row in cur.execute(
        "SELECT zone_id, name, description, created_at FROM zones LIMIT 5"
    ):
        print(dict(row))

    print("\nSample zone_keyframes:")
    for row in cur.execute(
        "SELECT keyframe_id, zone_id, image_path, embedding_type, created_at FROM zone_keyframes LIMIT 5"
    ):
        print(dict(row))

    print("\nSample observations:")
    for row in cur.execute(
        "SELECT obs_id, input_type, source_name, zone_top1, zone_conf, final_device_id, final_conf, policy_action, created_at FROM observations LIMIT 5"
    ):
        print(dict(row))

    conn.close()


if __name__ == "__main__":
    main()
