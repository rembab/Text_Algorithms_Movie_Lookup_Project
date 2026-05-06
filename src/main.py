import flet as ft
from ui.chat import ChatScreen
from db.database import Database


def main(page: ft.Page):
    db = Database()

    def print_best_matches(text):
        results_df = db.search_movies(text)

        chat.update_results(results_df)

    chat = ChatScreen(page, print_best_matches)


if __name__ == "__main__":
    ft.run(main=main)
