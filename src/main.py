import flet as ft
from ui.chat import ChatScreen
from db.database import Database


def main(page: ft.Page):
    db = Database()

    def print_best_matches(text):
        results = db.search_movies(text).to_pandas()[["Title", "_distance"]]
        results["Similarity"] = 1 - results["_distance"]

        title_width = results["Title"].str.len().max() + 2
        formatted_lines = [
            f"{row['Title']:<{title_width}}{row['Similarity']:>8.4f}"
            for _, row in results.iterrows()
        ]

        chat.set_results_text(
            "My best guesses:\n" + f"{"Title":<{title_width-10}}{"Similarity":>18}\n" + "\n".join(formatted_lines)
        )

    chat = ChatScreen(page, print_best_matches)


if __name__ == "__main__":
    ft.run(main=main)
