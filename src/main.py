import flet as ft
from ui.chat import ChatScreen
from db.database import Database


def main(page: ft.Page):
    db = Database()

    def print_best_matches(text):
        chat.set_results_text(
            f"My best guesses:\n {db.search_movies(text).to_pandas()[["Title", "_distance"]].to_string()}"
        )

    chat = ChatScreen(page, print_best_matches)


if __name__ == "__main__":
    ft.run(main=main)
