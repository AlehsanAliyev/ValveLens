import argparse
import json
import sys
import uuid
from pathlib import Path
from urllib import request


def _get_json(url: str) -> dict:
    with request.urlopen(url, timeout=10) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))


def _post_multipart(url: str, file_path: Path) -> dict:
    boundary = f"----valvelens-{uuid.uuid4().hex}"
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8")
    footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = header + file_path.read_bytes() + footer
    req = request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Content-Length", str(len(body)))
    with request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:8000")
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    base = args.base.rstrip("/")
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(2)

    try:
        status = _get_json(f"{base}/debug/status")
    except Exception as exc:
        print(f"Failed to reach /debug/status: {exc}")
        sys.exit(2)

    print("Debug status:", status)

    try:
        response = _post_multipart(f"{base}/infer/image", image_path)
    except Exception as exc:
        print(f"Failed to run /infer/image: {exc}")
        sys.exit(2)

    zone_candidates = response.get("zone", {}).get("candidates", [])
    detections = response.get("detections", [])

    print(f"Zone candidates: {len(zone_candidates)}")
    print(f"Detections: {len(detections)}")

    if len(detections) == 0:
        print("Smoke test failed: no detections returned.")
        sys.exit(1)
    if len(zone_candidates) == 0 and status.get("counts", {}).get("zones", 0) > 0:
        print("Smoke test failed: zones exist but no zone candidates returned.")
        sys.exit(1)

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
