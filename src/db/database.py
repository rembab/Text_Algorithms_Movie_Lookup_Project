from typing import Optional

import lancedb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder

import config


RESULT_COLUMNS = ["Title", "Release Year", "Genre", "Plot", "Confidence"]


class DatabaseNotReadyError(RuntimeError):
    """Raised when the vector store has not been built yet."""


class Database:
    def __init__(self, db_path: str = config.DB_PATH):
        self.db = lancedb.connect(db_path)
        try:
            self.plot_table = self.db.open_table(config.TABLE_NAME)
        except Exception as exc:  # table missing / DB never built
            raise DatabaseNotReadyError(
                "Movie database not found. Run "
                "`python src/db/setup_database.py` to build it first."
            ) from exc

        print("Loading embedding model...")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)

        print("Loading re-ranker...")
        self.reranker = CrossEncoder(config.RERANKER_MODEL)

    def search_movies(
        self,
        user_query: str,
        num_results: int = config.NUM_RESULTS,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        genre: Optional[str] = None,
    ):
        empty = pd.DataFrame(columns=RESULT_COLUMNS)

        query = user_query.strip()
        if not query:
            return empty

        # BGE is an instruction model: prefix the (short) query for asymmetric
        # query -> passage retrieval. Documents were embedded without it.
        query_vector = self.model.encode(
            config.BGE_QUERY_INSTRUCTION + query, normalize_embeddings=True
        )

        candidates = self._retrieve(query, query_vector)
        if candidates.empty:
            return empty

        candidates = self._apply_filters(candidates, min_year, max_year, genre)
        if candidates.empty:
            return empty

        # Stage 2: precise re-ranking. Trim the passage to stay within the
        # cross-encoder's 512-token window so the plot isn't cut off.
        pairs = [
            [query, str(row["combined_text"])[: config.RERANK_MAX_CHARS]]
            for _, row in candidates.iterrows()
        ]
        raw_scores = self.reranker.predict(pairs)

        candidates = candidates.copy()
        candidates["_confidence"] = 1 / (1 + np.exp(-np.asarray(raw_scores)))

        best = candidates.sort_values(by="_confidence", ascending=False).head(
            num_results
        )
        best["Confidence"] = (
            (best["_confidence"] * 100).round(1).astype(str) + "%"
        )

        return best[RESULT_COLUMNS]

    def similar_movies(
        self,
        title: str,
        year: str,
        k: int = config.SIMILAR_MOVIES_COUNT,
    ) -> pd.DataFrame:
        """Return the k nearest neighbors by stored embedding (excluding self)."""
        empty = pd.DataFrame(columns=["Title", "Release Year", "Genre", "Plot", "Similarity"])

        row = self._find_movie_row(title, year)
        if row is None:
            return empty

        vector = row["vector"]
        neighbors = (
            self.plot_table.search(vector)
            .distance_type("cosine")
            .limit(k + 1)
            .to_pandas()
        )

        if neighbors.empty:
            return empty

        year_str = str(year)
        mask = ~(
            (neighbors["Title"].astype(str) == str(title))
            & (neighbors["Release Year"].astype(str) == year_str)
        )
        neighbors = neighbors[mask].head(k)

        # Cosine distance in LanceDB is 1 - similarity for normalized vectors.
        neighbors = neighbors.copy()
        neighbors["Similarity"] = (
            ((1 - neighbors["_distance"].clip(0, 2)) * 100).round(1).astype(str) + "%"
        )

        return neighbors[["Title", "Release Year", "Genre", "Plot", "Similarity"]]

    def _find_movie_row(self, title: str, year: str):
        """Look up a single movie row by title + year."""
        year_str = str(year)
        safe_title = str(title).replace("'", "''")

        try:
            matches = (
                self.plot_table.search()
                .where(f"Title = '{safe_title}' AND `Release Year` = '{year_str}'")
                .limit(1)
                .to_pandas()
            )
            if not matches.empty:
                return matches.iloc[0]
        except Exception:
            pass

        # Fallback when the filter syntax fails (special characters, etc.).
        try:
            by_title = (
                self.plot_table.search()
                .where(f"Title = '{safe_title}'")
                .limit(20)
                .to_pandas()
            )
            if not by_title.empty:
                year_match = by_title[
                    by_title["Release Year"].astype(str) == year_str
                ]
                if not year_match.empty:
                    return year_match.iloc[0]
        except Exception:
            pass

        return None

    def _retrieve(self, query: str, query_vector) -> pd.DataFrame:
        """Hybrid retrieval: union of vector hits and (optionally) BM25
        full-text hits. The cross-encoder later re-ranks the merged pool, so a
        plain union is enough — no fragile score fusion needed."""
        vector_df = (
            self.plot_table.search(query_vector)
            .distance_type("cosine")
            .limit(config.RETRIEVAL_LIMIT)
            .to_pandas()
        )
        frames = [vector_df]

        if config.ENABLE_FULLTEXT:
            try:
                fts_df = (
                    self.plot_table.search(query, query_type="fts")
                    .limit(config.RETRIEVAL_LIMIT)
                    .to_pandas()
                )
                frames.append(fts_df)
            except Exception:
                # No FTS index (older DB build) -> vector-only retrieval.
                pass

        combined = pd.concat(frames, ignore_index=True)
        # Each movie has a unique combined_text; use it to de-duplicate the union.
        combined = combined.drop_duplicates(subset=["combined_text"]).reset_index(
            drop=True
        )
        return combined

    @staticmethod
    def _apply_filters(
        df: pd.DataFrame,
        min_year: Optional[int],
        max_year: Optional[int],
        genre: Optional[str],
    ) -> pd.DataFrame:
        out = df

        if min_year is not None or max_year is not None:
            years = pd.to_numeric(out["Release Year"], errors="coerce")
            mask = years.notna()  # rows with an "Unknown" year are excluded when filtering
            if min_year is not None:
                mask &= years >= min_year
            if max_year is not None:
                mask &= years <= max_year
            out = out[mask]

        if genre:
            out = out[
                out["Genre"].astype(str).str.contains(genre, case=False, na=False)
            ]

        return out
