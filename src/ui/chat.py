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
            content=ft.Text("Submit query", text_align=ft.TextAlign.CENTER),
            bgcolor="#FFFFFF",
            color="#000000",
            on_click=self.on_click_submit,
            width=170,
        )

        self.results_grid = ft.GridView(
            expand=True,
            runs_count=3,
            max_extent=280,
            child_aspect_ratio=1.1,
            spacing=14,
            run_spacing=14,
            padding=ft.padding.all(8),
        )

        self.main_layout = ft.Column(
            controls=[
                self.header_text,
                self.input_field,
                self.submit_button,
                ft.Divider(height=20, color="transparent"),
                self.results_grid,
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            expand=True,
        )

    def close_dialog(self):
        self.plot_dialog.open = False
        self.page.update()

    def show_movie_details(self, e, title, year, plot):
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
        ], tight=True, alignment=ft.MainAxisAlignment.CENTER)
        self.submit_button.disabled = True
        self.page.update()

        # Moves the query to another thread so that the UI updates
        await asyncio.to_thread(self.submit_fun, query)

    def _sim_color(self, sim: float) -> str:
        # Interpolates between red, yellow and green
        t = max(0.0, min(1.0, float(sim)))
        if t < 0.5:
            r, g = 220, int(220 * (t / 0.5))
        else:
            r, g = int(220 * (1 - (t - 0.5) / 0.5)), 200
        return f"#{r:02x}{g:02x}40"

    def _build_card(self, title: str, year: str, plot: str, sim: float) -> ft.Container:
        sim_pct = sim * 100
        bar_color = self._sim_color(sim)
        plot_preview = (plot[:160] + "…") if len(plot) > 160 else plot

        # sim_bar = ft.Container(
        #     width=10,
        #     border_radius=ft.border_radius.all(4),
        #     bgcolor="#3b3e4a",
        #     content=ft.Column(
        #         controls=[
        #             ft.Container(expand=True),          # empty top spacer
        #             ft.Container(
        #                 bgcolor=bar_color,
        #                 border_radius=ft.border_radius.all(4),
        #                 height=None,
        #                 expand=int(round(sim_pct)),     # proportional fill
        #             ),
        #         ],
        #         spacing=0,
        #         expand=True,
        #     ),
        #     expand=False,
        # )
        sim_bar = ft.Container(
            width=10,
            border_radius=ft.border_radius.all(4),
            bgcolor="#3b3e4a",
            content=ft.Column(
                controls=[
                    ft.Container(expand=int(100 - round(sim_pct))),
                    ft.Container(
                        bgcolor=bar_color,
                        border_radius=ft.border_radius.all(4),
                        expand=int(round(sim_pct)) or 1,
                    ),
                ],
                spacing=0,
                expand=True,
            ),
            expand=False,
        )

        card_body = ft.Column(
            controls=[
                ft.Text(
                    f"{title} ({year})",
                    font_family="Consolas",
                    weight=ft.FontWeight.BOLD,
                    size=13,
                    color="#FFFFFF",
                    max_lines=2,
                    overflow=ft.TextOverflow.ELLIPSIS,
                ),
                ft.Row(
                    controls=[
                        ft.Container(
                            width=8,
                            height=8,
                            border_radius=ft.border_radius.all(4),
                            bgcolor=bar_color,
                        ),
                        ft.Text(
                            f"Similarity: {sim_pct:.1f}%",
                            font_family="Consolas",
                            size=12,
                            color=bar_color if bar_color else "#a8ffb2",
                        ),
                    ],
                    spacing=6,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Divider(height=6, color="#3b3e4a"),
                ft.Text(
                    plot_preview,
                    font_family="Consolas",
                    size=11,
                    color="#c8cad4",
                    max_lines=5,
                    overflow=ft.TextOverflow.ELLIPSIS,
                    expand=True,
                ),
                ft.Text(
                    "Show full summary →",
                    font_family="Consolas",
                    size=11,
                    color="#a8ffb2",
                ),
            ],
            spacing=6,
            expand=True,
        )

        return ft.Container(
            content=ft.Row(
                controls=[card_body, sim_bar],
                spacing=8,
                expand=True,
                vertical_alignment=ft.CrossAxisAlignment.STRETCH,
            ),
            bgcolor="#4c5060",
            border=ft.border.all(1, "#3b3e4a"),
            border_radius=10,
            padding=ft.padding.all(12),
            expand=True,
            on_click=lambda e, t=title, y=year, p=plot: self.show_movie_details(e, t, y, p),
            ink=True,
        )

    def update_results(self, results_df: pd.DataFrame):
        self.results_grid.controls.clear()

        for _, row in results_df.iterrows():
            title = str(row.get("Title", "Unknown"))
            year = str(row.get("Release Year", "Unknown"))
            plot = str(row.get("Plot", ""))
            sim = row.get("Similarity", 0)
            sim_val = float(sim.strip('%')) / 100 if isinstance(sim, str) else 0.0

            self.results_grid.controls.append(self._build_card(title, year, plot, sim_val))

        self.submit_button.content = ft.Text("Submit query", text_align=ft.TextAlign.CENTER)
        self.submit_button.disabled = False
        self.page.update()
