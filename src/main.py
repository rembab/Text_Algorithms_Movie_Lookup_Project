import flet as ft
from ui.chat import ChatScreen
from db.database import Database


def main(page: ft.Page):
    db = Database()

    def print_best_matches(text):
        print(f"Best movie matches for {text}")
        print(db.search_movies(text).to_pandas().Title.to_string())

    ChatScreen(page, print_best_matches)


if __name__ == "__main__":
    ft.run(main=main)
