# Movie Lookup — Semantic Search

Find a movie when you only vaguely remember its plot. Describe what you
recall (e.g. *"a man relives the same day over and over"*) and the app returns
the most likely matches, ranked by confidence.

## How it works

A two-stage semantic search pipeline:

1. **Hybrid retrieval** — your query is embedded with `BAAI/bge-small-en-v1.5`
   (with the BGE query instruction prefix) and matched against a
   [LanceDB](https://lancedb.com/) vector store via cosine similarity. When a
   full-text index is present, BM25 keyword hits are unioned in to catch
   proper nouns and distinctive terms.
2. **Re-ranking** — the merged candidates are re-scored with a CrossEncoder
   (`ms-marco-MiniLM-L-6-v2`) for higher precision, and the best results are
   shown with a confidence percentage.

Optional **year-range and genre filters** narrow the candidates before ranking.

## Project layout

```
src/
├── config.py            # Central config: paths, models, limits, UI constants
├── main.py              # Entry point; wires the UI to the database
├── ui/chat.py           # Flet GUI (search box, results table, plot dialog)
└── db/
    ├── database.py      # Runtime search: embed -> vector search -> re-rank
    └── setup_database.py # One-time build of the vector DB from CSV sources
data/                    # CSV datasets + generated vector database
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Place the dataset CSVs in `data/`:

- `wiki_movie_plots_deduped.csv`
- `imdb_movie_keyword.csv`
- `MovieVerse.csv`

## Build the database (one-time)

```bash
python src/db/setup_database.py
```

**Fast test build** (Wikipedia only, 10,000 newest movies by release year):

```bash
python src/db/setup_database.py --test
python src/db/setup_database.py --test --wiki-limit 5000
```

This merges the sources, de-duplicates movies semantically (normalising titles
so the same film merges across datasets), folds in cast and keyword signals,
embeds everything, and builds both an ANN vector index and a BM25 full-text
index. It is compute-heavy — a CUDA GPU speeds it up considerably.

> Rebuild the database whenever you change the embedding model, the
> `combined_text` fields, or upgrade from an older build (the full-text index
> and richer fields require a fresh build).

## Run the app

```bash
python src/main.py
```

The window opens immediately and shows *"Loading models..."* while the models
load in the background; the search box enables once they are ready.

## Configuration

Tweak models, search limits, the de-duplication threshold, and UI colors in
`src/config.py`. The embedding model and DB path must match between the build
step and runtime.
