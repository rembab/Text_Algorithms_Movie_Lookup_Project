import pandas as pd
import lancedb
from sentence_transformers import SentenceTransformer


def build_vector_database(
    csv_path: str = "../../data/wiki_movie_plots_deduped.csv",
    db_path: str = "../../data/database",
    transformer_model: str = "BAAI/bge-small-en-v1.5",
):
    db = lancedb.connect(db_path)

    print(f"Loading csv {csv_path}...")
    dataset = pd.read_csv(csv_path)
    dataset.drop(
        columns=[
            "Release Year",
            "Origin/Ethnicity",
            "Director",
            "Cast",
            "Genre",
            "Wiki Page",
        ],
        inplace=True,
    )

    dataset = dataset.head(2000)

    print(f"Loading ai slop: {transformer_model}...")
    model = SentenceTransformer(transformer_model)

    print("Generating vectors")
    plots_list = dataset["Plot"].tolist()
    embeddings = model.encode(plots_list, show_progress_bar=True)

    dataset["vector"] = embeddings.tolist()

    print("Saving database")
    table = db.create_table("plots", data=dataset, mode="overwrite")

    print(f"Loaded {table.count_rows()} rows into the database.")


if __name__ == "__main__":
    build_vector_database()
