import os
import sys
import threading

import flet as ft
import pandas as pd

# Ensure imports work when launched as `python src/main.py`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from ui.chat import ChatScreen
from db.database import Database, DatabaseNotReadyError

EMPTY_RESULTS = pd.DataFrame(
    columns=["Title", "Release Year", "Genre", "Plot", "Confidence"]
)
EMPTY_SIMILAR = pd.DataFrame(
    columns=["Title", "Release Year", "Genre", "Plot", "Similarity"]
)


def main(page: ft.Page):
    state = {"db": None}

    def submit(text: str, filters: dict):
        db = state["db"]
        if db is None:
            chat.set_app_status("Models are still loading, please wait...")
            return

        try:
            results_df = db.search_movies(
                text,
                num_results=config.RESULTS_GRID_MAX,
                min_year=filters.get("min_year"),
                max_year=filters.get("max_year"),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Search failed: {exc}")
            results_df = EMPTY_RESULTS
        chat.update_results(results_df)

    def similar(title: str, year: str):
        db = state["db"]
        if db is None:
            return EMPTY_SIMILAR
        return db.similar_movies(title, year)

    chat = ChatScreen(page, submit, similar)
    chat.set_app_status("Loading models...")

    def load_db():
        try:
            state["db"] = Database()
            chat.set_app_status(None)
            print("Models loaded — app is ready.")
        except DatabaseNotReadyError as exc:
            chat.set_app_status(str(exc))
            print(exc)
        except Exception as exc:  # noqa: BLE001
            chat.set_app_status(f"Failed to load models: {exc}")
            print(f"Failed to load models: {exc}")

    threading.Thread(target=load_db, daemon=True).start()


if __name__ == "__main__":
    print("Starting Movie Lookup (web view)...")
    print("Your browser should open automatically. Keep this terminal open.")
    # port=0 picks a free port so a leftover instance can't block startup.
    # ft.run(main=main, view=ft.AppView.WEB_BROWSER, port=0)
    ft.run(main=main, view=ft.AppView.FLET_APP, port=0)
