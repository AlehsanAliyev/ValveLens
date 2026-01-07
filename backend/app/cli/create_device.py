import argparse

from app import db


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", required=True)
    parser.add_argument("--zone_id", required=True)
    parser.add_argument("--type", required=True)
    parser.add_argument("--desc", default="")
    args = parser.parse_args()

    db.init_db()
    device_id = db.create_device(args.device_id, args.zone_id, args.type, args.desc)
    print(device_id)


if __name__ == "__main__":
    main()
