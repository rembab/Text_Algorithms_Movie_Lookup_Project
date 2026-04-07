import lancedb


class Database:
    def __init__(self, db_path: str = "../data/database"):
        self.db = lancedb.connect(db_path)

        self.plot_table = self.db.open_table("plots")

        print(f"Found {self.plot_table.count_rows()} movie records.")
