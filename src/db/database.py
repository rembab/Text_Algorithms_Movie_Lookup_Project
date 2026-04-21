import lancedb
from sentence_transformers import SentenceTransformer


class Database:
    def __init__(self, db_path: str = "./data/database"):
        self.db = lancedb.connect(db_path)
        self.plot_table = self.db.open_table("plots")

        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def search_movies(self, user_query: str, num_results: int = 5):
        query_vector = self.model.encode(user_query)

        results_df = self.plot_table.search(query_vector).limit(num_results)

        return results_df
