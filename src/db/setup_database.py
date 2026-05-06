import pandas as pd
import lancedb
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import re


def cosine_similarity(a, b):
    """Calculates cosine similarity between two 1D numpy arrays."""
    if norm(a) == 0 or norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (norm(a) * norm(b))


GENRE_SPLIT_RE = re.compile(r"[,|\[\]/]")


def combine_genres(genres: List[str]) -> str:
    cleaned = {
        part.strip().title()
        for g in genres
        if pd.notna(g)
        for part in GENRE_SPLIT_RE.split(str(g))
        if part.strip() and part.strip().lower() != "unknown"
    }

    return ", ".join(sorted(cleaned)) if cleaned else "Unknown"


def smart_combine_plots(
    plots: List[str],
    model: SentenceTransformer,
    threshold: float = 0.85,
) -> str:
    valid_plots = [str(p).strip() for p in plots if pd.notna(p) and str(p).strip()]

    if not valid_plots:
        return ""
    if len(valid_plots) == 1:
        return valid_plots[0]

    valid_plots = sorted(valid_plots, key=len, reverse=True)

    embeddings = model.encode(valid_plots, batch_size=32, normalize_embeddings=True)
    embeddings = np.array(embeddings)

    selected_indices = []
    used = np.zeros(len(valid_plots), dtype=bool)

    for i in range(len(valid_plots)):
        if used[i]:
            continue

        selected_indices.append(i)

        sims = embeddings @ embeddings[i]

        used |= sims >= threshold

    return "\n\n".join(valid_plots[i] for i in selected_indices)


def build_vector_database(
    csv_configs: List[Dict[str, str]],
    db_path: str = "./data/database",
    transformer_model: str = "BAAI/bge-small-en-v1.5",
    similarity_threshold: float = 0.85,
):
    db = lancedb.connect(db_path)

    print(f"Loading transformer model: {transformer_model}...")
    model = SentenceTransformer(transformer_model)

    all_dfs = []

    for config in csv_configs:
        path = config["path"]
        print(f"Loading csv {path}...")
        df = pd.read_csv(path)

        rename_map = {}
        if config.get("title_col"):
            rename_map[config["title_col"]] = "Title"
        if config.get("year_col"):
            rename_map[config["year_col"]] = "Release Year"
        if config.get("genre_col"):
            rename_map[config["genre_col"]] = "Genre"
        if config.get("plot_col"):
            rename_map[config["plot_col"]] = "Plot"

        df = df.rename(columns=rename_map)

        if "Release Year" not in df.columns:
            df["Release Year"] = "Unknown"
        if "Genre" not in df.columns:
            df["Genre"] = "Unknown"

        df = df[["Title", "Release Year", "Genre", "Plot"]]
        all_dfs.append(df)

    print("Concatenating datasets...")
    dataset = pd.concat(all_dfs, ignore_index=True)

    dataset["Release Year"] = (
        dataset["Release Year"]
        .astype(str)
        .str.extract(r"(\d{4})", expand=False)
        .fillna("Unknown")
    )

    print("Merging duplicates based on semantic distance...")

    grouped = dataset.groupby(["Title", "Release Year"])
    total_groups = grouped.ngroups

    progress_counter = [0]

    def merge_group(group):
        genres = group["Genre"].tolist()
        plots = group["Plot"].tolist()

        result = pd.Series(
            {
                "Genre": combine_genres(genres),
                "Plot": smart_combine_plots(plots, model, similarity_threshold),
            }
        )

        progress_counter[0] += 1

        current = min(progress_counter[0], total_groups)

        percent = (current / total_groups) * 100
        bar_length = 40
        filled_length = int(bar_length * current // total_groups)
        bar = "█" * filled_length + "-" * (bar_length - filled_length)

        print(
            f"\rProgress: |{bar}| {percent:.1f}% ({current}/{total_groups})",
            end="",
            flush=True,
        )

        return result

    grouped_dataset = grouped.apply(merge_group).reset_index()
    print()

    print("Combining fields for rich vectorization...")
    grouped_dataset["combined_text"] = (
        "Title: "
        + grouped_dataset["Title"]
        + ". "
        + "Release Year: "
        + grouped_dataset["Release Year"]
        + ". "
        + "Genre: "
        + grouped_dataset["Genre"]
        + ". "
        + "Plot: "
        + grouped_dataset["Plot"]
    )

    print("Generating final vectors...")
    text_list = grouped_dataset["combined_text"].tolist()
    embeddings = model.encode(text_list, show_progress_bar=True)

    grouped_dataset["vector"] = embeddings.tolist()

    print("Saving database...")
    table = db.create_table("plots", data=grouped_dataset, mode="overwrite")

    print(f"Loaded {table.count_rows()} unique rows into the database.")


if __name__ == "__main__":
    sources = [
        {
            "path": "./data/wiki_movie_plots_deduped.csv",
            "title_col": "Title",
            "year_col": "Release Year",
            "genre_col": "Genre",
            "plot_col": "Plot",
        },
        {
            "path": "./data/imdb_movie_keyword.csv",
            "title_col": "movie_title",
            "year_col": "year",
            "genre_col": "genre",
            "plot_col": "synopsis",
        },
        {
            "path": "./data/MovieVerse.csv",
            "title_col": "movie_name",
            "year_col": "year",
            "genre_col": "movie_genres",
            "plot_col": "movie_summary",
        },
    ]

    build_vector_database(csv_configs=sources)
