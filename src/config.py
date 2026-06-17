"""Central configuration for the Movie Lookup project.

Keeping these values in one place avoids drift between the database build
step (`setup_database.py`) and the runtime search (`database.py`). The
embedding model and DB path in particular MUST match between build and
query time, or search silently degrades.
"""

# --- Storage ---
DB_PATH = "./data/database"
TABLE_NAME = "plots"
WIKI_CSV_PATH = "./data/wiki_movie_plots_deduped.csv"

# --- Database build (testing) ---
TEST_WIKI_ROW_LIMIT = 10_000  # used with `setup_database.py --test`

# --- Models ---
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Search behaviour ---
RETRIEVAL_LIMIT = 50      # candidates pulled from the vector store before re-ranking
NUM_RESULTS = 5           # final results shown to the user
RESULTS_GRID_MAX = 24     # results requested to fill the paginated card grid (UI)
SIMILAR_MOVIES_COUNT = 6  # neighbors shown in the detail panel
SIMILARITY_THRESHOLD = 0.85  # plots more similar than this are treated as duplicates
ENCODE_BATCH_SIZE = 64
ENABLE_FULLTEXT = True    # combine BM25 full-text hits with vector hits (hybrid retrieval)

# BGE v1.5 is an instruction model: short queries should be prefixed for
# asymmetric (query -> passage) retrieval. Documents are embedded WITHOUT it.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Caps to stay within the 512-token limit of the embedding / re-ranker models.
# The full plot is still stored for display; only the embedded/re-ranked text
# is trimmed. ~1500 chars of plot + short metadata fits comfortably in 512 tokens.
MAX_PLOT_CHARS_FOR_EMBEDDING = 1500
RERANK_MAX_CHARS = 2000

# --- UI ---
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800

COLOR_BG = "#5b5f72"
COLOR_SURFACE = "#4c5060"
COLOR_BORDER = "#3b3e4a"
COLOR_TEXT = "#FFFFFF"
COLOR_ACCENT = "#a8ffb2"
FONT_FAMILY = "Consolas"
