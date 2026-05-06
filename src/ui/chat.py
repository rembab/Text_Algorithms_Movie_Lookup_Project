from typing import Callable
import flet as ft
import pandas as pd


class ChatScreen:
    def __init__(self, page: ft.Page, submit_fun: Callable):
        self.page = page
        self.submit_fun = submit_fun
        self.setup_page()

    def setup_page(self):
        self.page.window.width = 800
        self.page.window.height = 800
        self.page.bgcolor = "#5b5f72"
        self.page.theme = ft.Theme(
            text_theme=ft.TextTheme(
                body_medium=ft.TextStyle(size=22),
                label_large=ft.TextStyle(size=20),
            )
        )
        self.init_components()
        self.page.add(self.main_layout)

    def init_components(self):
        self.header_text = ft.Text(
            value="I remember a movie where...",
            text_align=ft.TextAlign.CENTER,
            font_family="Consolas",
            weight=ft.FontWeight.BOLD,
        )

        self.input_field = ft.TextField(
            width=600, multiline=True, min_lines=4, max_lines=8, border_color="#FFFFFF"
        )

        self.submit_button = ft.Button(
            content=ft.Text("Submit query"),
            bgcolor="#FFFFFF",
            color="#000000",
            on_click=self.on_click_submit,
            style=ft.ButtonStyle(
                text_style=ft.TextStyle(
                    font_family="Consolas", weight=ft.FontWeight.BOLD
                ),
            ),
        )

        self.results_table = ft.DataTable(
            columns=[
                ft.DataColumn(
                    ft.Text("Title", font_family="Consolas", weight=ft.FontWeight.BOLD)
                ),
                ft.DataColumn(
                    ft.Text("Year", font_family="Consolas", weight=ft.FontWeight.BOLD),
                    numeric=True,
                ),
                ft.DataColumn(
                    ft.Text(
                        "Similarity", font_family="Consolas", weight=ft.FontWeight.BOLD
                    ),
                    numeric=True,
                ),
            ],
            rows=[],
            bgcolor="#4c5060",
            border=ft.border.all(1, "#3b3e4a"),
            border_radius=10,
            column_spacing=40,
        )

        self.table_container = ft.Column(
            controls=[self.results_table],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )

        self.main_layout = ft.Column(
            controls=[
                self.header_text,
                self.input_field,
                self.submit_button,
                ft.Divider(height=20, color="transparent"),
                self.table_container,
            ],
            alignment=ft.MainAxisAlignment.START,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            expand=True,
        )

    def on_click_submit(self, e):
        user_input = self.input_field.value
        if not user_input.strip():
            return

        self.submit_fun(user_input)

        self.input_field.value = ""
        self.page.update()

    def update_results(self, results_df: pd.DataFrame):
        self.results_table.rows.clear()

        for index, row in results_df.iterrows():

            title = str(row.get("Title", "Unknown"))
            year = str(row.get("Release Year", "Unknown"))

            similarity_val = row.get("Similarity", 0)
            similarity_str = f"{similarity_val:.4f}"

            table_row = ft.DataRow(
                cells=[
                    ft.DataCell(ft.Text(title, font_family="Consolas", size=16)),
                    ft.DataCell(ft.Text(year, font_family="Consolas", size=16)),
                    ft.DataCell(
                        ft.Text(
                            similarity_str,
                            font_family="Consolas",
                            size=16,
                            color="#a8ffb2",
                        )
                    ),
                ]
            )
            self.results_table.rows.append(table_row)

        self.page.update()
