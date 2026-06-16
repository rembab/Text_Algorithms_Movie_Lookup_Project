import lancedb
from sentence_transformers import SentenceTransformer, CrossEncoder
import pandas as pd
import numpy as np


import lancedb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder


class Database:
    def __init__(self, db_path: str = "./data/database"):
        self.db = lancedb.connect(db_path)
        self.plot_table = self.db.open_table("plots")

        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")

        print("Loading Re-ranker...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def search_movies(self, user_query: str, num_results: int = 24):
        query_vector = self.model.encode(user_query)

        raw_results_df = (
            self.plot_table.search(query_vector)
            .distance_type("cosine")
            .limit(50)
            .to_pandas()
        )

        if raw_results_df.empty:
            return pd.DataFrame(
                columns=["Title", "Release Year", "Genre", "Plot", "Similarity"]
            )

        pairs = [
            [user_query, row["combined_text"]] for _, row in raw_results_df.iterrows()
        ]
        raw_scores = self.reranker.predict(pairs)

        confidence_scores = 1 / (1 + np.exp(-raw_scores))
        raw_results_df["Confidence"] = confidence_scores

        best_results_df = raw_results_df.sort_values(
            by="Confidence", ascending=False
        ).head(num_results)

        best_results_df["Similarity"] = (best_results_df["Confidence"] * 100).round(
            1
        ).astype(str) + "%"

        return best_results_df[["Title", "Release Year", "Genre", "Plot", "Similarity"]]
