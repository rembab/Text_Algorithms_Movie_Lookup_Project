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
        self.page.window.min_width = 800
        self.page.window.min_height = 800
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
        # --- Pagination state ---
        self._all_cards: list = []
        self._current_page: int = 0
        self._total_pages: int = 4
        self._cards_per_page: int = 6

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

        self.results_grid = ft.Column(
            controls=[],
            spacing=14,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )

        self._prev_btn = ft.IconButton(
            icon=ft.Icons.CHEVRON_LEFT,
            icon_color="#FFFFFF",
            icon_size=28,
            disabled=True,
            on_click=self._on_prev_page,
        )
        self._next_btn = ft.IconButton(
            icon=ft.Icons.CHEVRON_RIGHT,
            icon_color="#FFFFFF",
            icon_size=28,
            disabled=True,
            on_click=self._on_next_page,
        )
        self._page_dots = ft.Row(controls=[], spacing=8)
        self.pagination_bar = ft.Row(
            controls=[self._prev_btn, self._page_dots, self._next_btn],
            alignment=ft.MainAxisAlignment.CENTER,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            visible=False,
        )

        self.main_layout = ft.Column(
            controls=[
                self.header_text,
                self.input_field,
                self.submit_button,
                ft.Divider(height=20, color="transparent"),
                self.results_grid,
                ft.Divider(height=10, color="transparent"),
                self.pagination_bar,
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
            width=240,
            height=200,
            on_click=lambda e, t=title, y=year, p=plot: self.show_movie_details(e, t, y, p),
            ink=True,
        )

    def _on_prev_page(self, e):
        if self._current_page > 0:
            self._current_page -= 1
            self._render_current_page()

    def _on_next_page(self, e):
        max_page = min(self._total_pages, (len(self._all_cards) + self._cards_per_page - 1) // self._cards_per_page) - 1
        if self._current_page < max_page:
            self._current_page += 1
            self._render_current_page()

    def _update_pagination_controls(self):
        num_pages = min(self._total_pages, (len(self._all_cards) + self._cards_per_page - 1) // self._cards_per_page)
        num_pages = max(num_pages, 1)

        self._prev_btn.disabled = self._current_page == 0
        self._next_btn.disabled = self._current_page >= num_pages - 1

        self._page_dots.controls.clear()
        for i in range(num_pages):
            is_active = i == self._current_page
            self._page_dots.controls.append(
                ft.Container(
                    width=10 if is_active else 8,
                    height=10 if is_active else 8,
                    border_radius=ft.border_radius.all(5),
                    bgcolor="#a8ffb2" if is_active else "#3b3e4a",
                    border=ft.border.all(1, "#a8ffb2" if is_active else "#5b6070"),
                    on_click=lambda e, idx=i: self._go_to_page(idx),
                )
            )

        self.pagination_bar.visible = num_pages > 1

    def _go_to_page(self, idx: int):
        self._current_page = idx
        self._render_current_page()

    def _render_current_page(self):
        self.results_grid.controls.clear()

        start = self._current_page * self._cards_per_page
        page_cards = self._all_cards[start: start + self._cards_per_page]

        # Pad to 6 so the grid is always full
        while len(page_cards) < self._cards_per_page:
            page_cards.append(ft.Container(width=240, height=200))

        for i in range(0, self._cards_per_page, 3):
            self.results_grid.controls.append(
                ft.Row(
                    controls=page_cards[i:i + 3],
                    spacing=14,
                    alignment=ft.MainAxisAlignment.CENTER,
                )
            )

        self._update_pagination_controls()
        self.page.update()

    def update_results(self, results_df: pd.DataFrame):
        max_results = self._total_pages * self._cards_per_page  # 24

        self._all_cards = []
        for _, row in results_df.head(max_results).iterrows():
            title = str(row.get("Title", "Unknown"))
            year = str(row.get("Release Year", "Unknown"))
            plot = str(row.get("Plot", ""))
            sim = row.get("Similarity", 0)
            sim_val = float(sim.strip('%')) / 100 if isinstance(sim, str) else 0.0
            self._all_cards.append(self._build_card(title, year, plot, sim_val))

        self._current_page = 0
        self._render_current_page()

        self.submit_button.content = ft.Text("Submit query", text_align=ft.TextAlign.CENTER)
        self.submit_button.disabled = False
        self.page.update()