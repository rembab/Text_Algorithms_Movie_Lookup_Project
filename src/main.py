import flet as ft
from ui.chat import ChatScreen


def say_meow(text):
    print(f"meow {text} meow")


def main(page: ft.Page):
    app = ChatScreen(page, say_meow)


if __name__ == "__main__":
    ft.app(target=main)
