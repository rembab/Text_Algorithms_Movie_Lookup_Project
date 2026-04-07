from typing import Callable
import flet as ft


class ChatScreen:
    def __init__(self, page: ft.Page, submit_fun: Callable):
        self.page = page

        self.setup_page()

        self.submit_fun = submit_fun

    def setup_page(self):
        self.page.window.width = 800
        self.page.window.height = 800
        self.page.bgcolor = "#5b5f72"
        self.page.theme = ft.Theme(
            text_theme=ft.TextTheme(
                body_medium=ft.TextStyle(size=22),  # Normal text
                label_large=ft.TextStyle(size=20),  # Ex. buttons
            )
        )
        self.init_components()
        self.page.add(self.main_layout)

    def init_components(self):
        self.header_text = ft.Text(
            value="I remember a movie where...",
            text_align=ft.TextAlign.CENTER,
        )

        self.input_field = ft.TextField(width=300, multiline=True, min_lines=10)

        self.submit_button = ft.Button(
            content="imabutton",
            bgcolor="#FFFFFF",
            color="#000000",
            on_click=self.on_click_submit,
        )

        self.results_text = ft.Text(
            value="",
            text_align=ft.TextAlign.START,
        )

        self.main_layout = ft.Column(
            controls=[
                self.header_text,
                self.input_field,
                self.submit_button,
                self.results_text,
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            height=800,
            width=800,
        )

    def on_click_submit(self):
        user_input = self.input_field.value

        # call the function passed on class init
        self.submit_fun(user_input)

        # clear input field
        self.input_field.value = ""
        self.page.update()

    def set_results_text(self, text):
        self.results_text.value = text
