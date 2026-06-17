import argparse
import os
import re
import sys
import math
from typing import List, Dict, Optional, Sequence

import lancedb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Allow `import config` whether this script is run from the project root
# (`python src/db/setup_database.py`) or from within `src/`.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


GENRE_SPLIT_RE = re.compile(r"[,|\[\]/]")
# Strip the Python-list noise found in some keyword columns: ['a', 'b'].
TERM_CLEAN_RE = re.compile(r"[\[\]'\"]")
# Title normalisation for cross-source de-duplication.
TITLE_PUNCT_RE = re.compile(r"[^\w\s]")
TITLE_ARTICLE_RE = re.compile(r"^(the|a|an)\s+")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    """Canonical key so the same film across sources merges despite case,
    punctuation, or a leading article (e.g. 'The Matrix' == 'matrix')."""
    text = str(title).lower().strip()
    text = TITLE_PUNCT_RE.sub(" ", text)
    text = TITLE_ARTICLE_RE.sub("", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def combine_genres(genres: List[str]) -> str:
    cleaned = {
        part.strip().title()
        for g in genres
        if pd.notna(g)
        for part in GENRE_SPLIT_RE.split(str(g))
        if part.strip() and part.strip().lower() != "unknown"
    }

    return ", ".join(sorted(cleaned)) if cleaned else "Unknown"


def combine_terms(values: Sequence[str], max_items: int = 25) -> str:
    """Union comma-separated terms (cast, keywords, ...) preserving order.

    Used to merge supplementary fields across duplicate rows of the same movie.
    """
    seen = []
    seen_lower = set()
    for value in values:
        if pd.isna(value):
            continue
        for part in str(value).split(","):
            term = TERM_CLEAN_RE.sub("", part).strip()
            if not term or term.lower() in ("unknown", "nan"):
                continue
            key = term.lower()
            if key not in seen_lower:
                seen_lower.add(key)
                seen.append(term)
    return ", ".join(seen[:max_items])


def smart_combine_plots(
    plots: List[str],
    embeddings: List[np.ndarray],
    threshold: float = config.SIMILARITY_THRESHOLD,
) -> str:
    """Merge plot variants, dropping near-duplicates using precomputed
    (normalised) embeddings so no per-group model calls are needed."""
    pairs = [
        (str(p).strip(), e)
        for p, e in zip(plots, embeddings)
        if pd.notna(p) and str(p).strip()
    ]

    if not pairs:
        return ""
    if len(pairs) == 1:
        return pairs[0][0]

    # Prefer longer (richer) plots when collapsing duplicates.
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    texts = [p for p, _ in pairs]
    emb = np.array([e for _, e in pairs])

    selected_indices = []
    used = np.zeros(len(texts), dtype=bool)

    for i in range(len(texts)):
        if used[i]:
            continue
        selected_indices.append(i)
        sims = emb @ emb[i]  # cosine, embeddings are normalised
        used |= sims >= threshold

    return "\n\n".join(texts[i] for i in selected_indices)


def _normalize_year(series: pd.Series) -> pd.Series:
    """Extract a plausible 4-digit release year, falling back to 'Unknown'."""
    years = series.astype(str).str.extract(r"(\d{4})", expand=False)
    numeric = pd.to_numeric(years, errors="coerce")
    valid = numeric.between(1880, 2030)
    return years.where(valid).fillna("Unknown")


def _take_newest_by_year(
    df: pd.DataFrame, limit: int, year_col: str = "Release Year"
) -> pd.DataFrame:
    """Keep the `limit` rows with the highest release years (newest first).

    Rows without a parseable year are sorted last so known-year films win the
    head() cut when testing on a subset of the Wikipedia corpus.
    """
    years = pd.to_numeric(
        df[year_col].astype(str).str.extract(r"(\d{4})", expand=False),
        errors="coerce",
    )
    sort_key = years.fillna(0)
    return (
        df.assign(_sort_year=sort_key)
        .sort_values("_sort_year", ascending=False)
        .head(limit)
        .drop(columns="_sort_year")
        .reset_index(drop=True)
    )


def _load_source(source: Dict[str, str]) -> pd.DataFrame:
    path = source["path"]
    print(f"Loading csv {path}...")
    df = pd.read_csv(path)

    # Merge any keyword columns into a single column before renaming.
    keyword_cols = [c for c in source.get("keyword_cols", []) if c in df.columns]
    if keyword_cols:
        df["Keywords"] = df[keyword_cols].astype(str).agg(", ".join, axis=1)

    rename_map = {}
    for key, target in (
        ("title_col", "Title"),
        ("year_col", "Release Year"),
        ("genre_col", "Genre"),
        ("plot_col", "Plot"),
        ("cast_col", "Cast"),
        ("director_col", "Director"),
    ):
        if source.get(key):
            rename_map[source[key]] = target
    df = df.rename(columns=rename_map)

    defaults = {
        "Release Year": "Unknown",
        "Genre": "Unknown",
        "Plot": "",
        "Cast": "",
        "Director": "",
        "Keywords": "",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    df = df[["Title", "Release Year", "Genre", "Plot", "Cast", "Director", "Keywords"]]

    limit_newest = source.get("limit_newest")
    if limit_newest is not None:
        before = len(df)
        df = _take_newest_by_year(df, limit_newest)
        print(
            f"  Test subset: kept {len(df)} newest rows (of {before}) "
            f"from {path}"
        )

    return df


def _build_combined_text(row) -> str:
    # Short, high-signal metadata first; the (capped) plot last. The cap keeps
    # the whole string within the embedding model's 512-token window so the
    # plot isn't silently truncated. The full plot is still stored for display.
    parts = [
        f"Title: {row['Title']}",
        f"Release Year: {row['Release Year']}",
        f"Genre: {row['Genre']}",
    ]
    if row["Cast"]:
        parts.append(f"Cast: {row['Cast']}")
    if row["Director"] and row["Director"] != "Unknown":
        parts.append(f"Director: {row['Director']}")
    if row["Keywords"]:
        parts.append(f"Keywords: {row['Keywords']}")
    plot = str(row["Plot"])[: config.MAX_PLOT_CHARS_FOR_EMBEDDING]
    parts.append(f"Plot: {plot}")
    return ". ".join(parts)


def build_vector_database(
    csv_configs: List[Dict[str, str]],
    db_path: str = config.DB_PATH,
    transformer_model: str = config.EMBEDDING_MODEL,
    similarity_threshold: float = config.SIMILARITY_THRESHOLD,
):
    db = lancedb.connect(db_path)

    print(f"Loading transformer model: {transformer_model}...")
    model = SentenceTransformer(transformer_model)

    all_dfs = [_load_source(source) for source in csv_configs]

    print("Concatenating datasets...")
    dataset = pd.concat(all_dfs, ignore_index=True)
    dataset["Release Year"] = _normalize_year(dataset["Release Year"])

    # Encode every plot ONCE up front, in batches, instead of re-encoding
    # inside each duplicate group. Normalised so we can use a plain dot
    # product for cosine similarity during de-duplication.
    print("Encoding plots for de-duplication...")
    plot_texts = dataset["Plot"].fillna("").astype(str).tolist()
    plot_embeddings = model.encode(
        plot_texts,
        batch_size=config.ENCODE_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    dataset["_plot_emb"] = list(plot_embeddings)

    # Group on a normalised title (+ year) so the same film across the three
    # sources merges despite cosmetic title differences.
    dataset["_title_key"] = dataset["Title"].map(normalize_title)

    print("Merging duplicates based on semantic distance...")
    grouped = dataset.groupby(["_title_key", "Release Year"])
    total_groups = grouped.ngroups
    progress_counter = [0]

    def merge_group(group):
        # Most frequent original title as the canonical display title.
        canonical_title = group["Title"].astype(str).value_counts().idxmax()
        result = pd.Series(
            {
                "Title": canonical_title,
                "Genre": combine_genres(group["Genre"].tolist()),
                "Plot": smart_combine_plots(
                    group["Plot"].tolist(),
                    group["_plot_emb"].tolist(),
                    similarity_threshold,
                ),
                "Cast": combine_terms(group["Cast"].tolist()),
                "Director": combine_terms(group["Director"].tolist()),
                "Keywords": combine_terms(group["Keywords"].tolist()),
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
    grouped_dataset["combined_text"] = grouped_dataset.apply(
        _build_combined_text, axis=1
    )

    print("Generating final vectors...")
    embeddings = model.encode(
        grouped_dataset["combined_text"].tolist(),
        batch_size=config.ENCODE_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    grouped_dataset["vector"] = embeddings.tolist()

    # Transient build-only columns must not be persisted.
    grouped_dataset = grouped_dataset.drop(
        columns=["_plot_emb", "_title_key"], errors="ignore"
    )

    print("Saving database...")
    table = db.create_table(config.TABLE_NAME, data=grouped_dataset, mode="overwrite")
    row_count = table.count_rows()
    print(f"Loaded {row_count} unique rows into the database.")

    # An ANN index keeps search fast as the corpus grows. Wrapped defensively
    # so a build never fails just because indexing parameters are unsupported.
    if row_count >= 256:
        try:
            print("Building vector index...")
            num_partitions = max(1, int(math.sqrt(row_count)))
            table.create_index(
                metric="cosine",
                num_partitions=num_partitions,
                num_sub_vectors=96,
            )
            print("Vector index created.")
        except Exception as exc:  # noqa: BLE001 - indexing is best-effort
            print(f"Index creation skipped ({exc}). Search will use a flat scan.")

    # A full-text (BM25) index enables hybrid retrieval at query time, which
    # complements vectors on proper nouns / distinctive keywords.
    try:
        print("Building full-text index...")
        table.create_fts_index("combined_text", replace=True)
        print("Full-text index created.")
    except Exception as exc:  # noqa: BLE001 - FTS is best-effort
        print(f"Full-text index skipped ({exc}). Hybrid search will be disabled.")


def _wiki_source(limit_newest: Optional[int] = None) -> Dict:
    source = {
        "path": config.WIKI_CSV_PATH,
        "title_col": "Title",
        "year_col": "Release Year",
        "genre_col": "Genre",
        "plot_col": "Plot",
        "cast_col": "Cast",
        "director_col": "Director",
    }
    if limit_newest is not None:
        source["limit_newest"] = limit_newest
    return source


def _all_sources() -> List[Dict]:
    return [
        _wiki_source(),
        {
            "path": "./data/imdb_movie_keyword.csv",
            "title_col": "movie_title",
            "year_col": "year",
            "genre_col": "genre",
            "plot_col": "synopsis",
            "cast_col": "cast",
            "keyword_cols": ["Key-Bert", "Yake"],
        },
        {
            "path": "./data/MovieVerse.csv",
            "title_col": "movie_name",
            "year_col": "year",
            "genre_col": "movie_genres",
            "plot_col": "movie_summary",
        },
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the LanceDB vector store from movie plot CSVs."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Fast test build: Wikipedia plots only, limited to the newest "
            f"{config.TEST_WIKI_ROW_LIMIT:,} rows (override with --wiki-limit)."
        ),
    )
    parser.add_argument(
        "--wiki-limit",
        type=int,
        default=config.TEST_WIKI_ROW_LIMIT,
        metavar="N",
        help=(
            "With --test, number of newest Wikipedia rows to include "
            f"(default: {config.TEST_WIKI_ROW_LIMIT:,})."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.test:
        limit = max(1, args.wiki_limit)
        print(
            f"TEST MODE: building from Wikipedia only "
            f"({limit:,} newest movies by release year)."
        )
        sources = [_wiki_source(limit_newest=limit)]
    else:
        sources = _all_sources()

    build_vector_database(csv_configs=sources)
