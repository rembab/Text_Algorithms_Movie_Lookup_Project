import flet as ft
from ui.chat import ChatScreen
from db.database import Database


def say_meow(text):
    print(f"meow {text} meow")


def main(page: ft.Page):
    app = ChatScreen(page, say_meow)
    db = Database()


if __name__ == "__main__":
    ft.app(target=main)
