from typing import Callable
import flet as ft
import pandas as pd
import asyncio

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
        # --- Pre-define the Dialog (Bulletproof Approach) ---
        self.dialog_title = ft.Text(
            font_family="Consolas", weight=ft.FontWeight.BOLD, color="#FFFFFF"
        )
        self.dialog_content = ft.Text(
            font_family="Consolas", size=16, color="#FFFFFF", selectable=True
        )

        self.plot_dialog = ft.AlertDialog(
            bgcolor="#4c5060",
            title=self.dialog_title,
            content=ft.Container(
                content=ft.Column(
                    controls=[self.dialog_content],
                    scroll=ft.ScrollMode.AUTO,
                    tight=True,
                ),
                height=400,
                width=600,
            ),
            actions=[
                ft.TextButton(
                    "Close",
                    on_click=lambda _: self.close_dialog(),
                    style=ft.ButtonStyle(color="#a8ffb2"),
                )
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.page.overlay.append(self.plot_dialog)

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
            bgcolor="#4c5060",
            border=ft.border.all(1, "#3b3e4a"),
            border_radius=10,
        )

        self.main_layout = ft.Column(
            controls=[
                self.header_text,
                self.input_field,
                self.submit_button,
                ft.Divider(height=20, color="transparent"),
                ft.Column([self.results_table], scroll=ft.ScrollMode.AUTO, expand=True),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            expand=True,
        )

    def close_dialog(self):
        self.plot_dialog.open = False
        self.page.update()

    def show_movie_details(self, e, title, year, plot):
        e.control.selected = False

        self.dialog_title.value = f"{title} ({year})"
        self.dialog_content.value = plot if plot else "No plot available."

        self.plot_dialog.open = True
        self.page.update()

    async def on_click_submit(self, e):
        query = self.input_field.value.strip()
        if not query:
            return
        
        self.submit_button.content = ft.Row([
            ft.ProgressRing(width=16, height=16),
            ft.Text("Searching...")
        ], tight=True)
        self.submit_button.disabled = True
        self.page.update()

        # Moves the query to another thread so that the UI updates
        await asyncio.to_thread(self.submit_fun, query)

    def update_results(self, results_df: pd.DataFrame):
        self.results_table.rows.clear()

        for _, row in results_df.iterrows():
            title = str(row.get("Title", "Unknown"))
            year = str(row.get("Release Year", "Unknown"))
            plot = str(row.get("Plot", ""))
            sim = row.get("Similarity", 0)
            sim_str = f"{sim:.4f}" if isinstance(sim, (int, float)) else str(sim)

            self.results_table.rows.append(
                ft.DataRow(
                    # Passing 'e' allows us to unselect the row inside the handler
                    on_select_change=lambda e, t=title, y=year, p=plot: self.show_movie_details(
                        e, t, y, p
                    ),
                    cells=[
                        ft.DataCell(ft.Text(title, font_family="Consolas")),
                        ft.DataCell(ft.Text(year, font_family="Consolas")),
                        ft.DataCell(
                            ft.Text(sim_str, font_family="Consolas", color="#a8ffb2")
                        ),
                    ],
                )
            )

        self.submit_button.content = ft.Text("Submit query")
        self.page.update()
