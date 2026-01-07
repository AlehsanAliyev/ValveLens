from app import db


def main() -> None:
    db.init_db()
    print("Database initialized.")


if __name__ == "__main__":
    main()
