import argparse

from app import db


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--desc", default="")
    args = parser.parse_args()
    db.init_db()
    zone_id = db.create_zone(args.name, args.desc)
    print(zone_id)


if __name__ == "__main__":
    main()
