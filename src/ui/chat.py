from typing import Callable, Optional
import flet as ft
import pandas as pd
import asyncio

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 900
CARD_HEIGHT = 200
GRID_COLUMNS = 3
CARDS_PER_PAGE = 6
DIALOG_WIDTH = 780
LABEL_GREY = "#8b919a"
LABEL_GREEN = "#a8ffb2"
FIELD_FONT = "Consolas"


class ChatScreen:
    def __init__(
        self,
        page: ft.Page,
        submit_fun: Callable,
        similar_fun: Callable,
    ):
        self.page = page
        self.submit_fun = submit_fun
        self.similar_fun = similar_fun
        self.setup_page()

    def setup_page(self):
        self.page.window.width = WINDOW_WIDTH
        self.page.window.height = WINDOW_HEIGHT
        self.page.window.min_width = WINDOW_WIDTH
        self.page.window.min_height = WINDOW_HEIGHT
        self.page.bgcolor = "#0d0f17"
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.theme = ft.Theme(
            text_theme=ft.TextTheme(
                body_medium=ft.TextStyle(size=22, color="#FFFFFF"),
                label_large=ft.TextStyle(size=20, color="#FFFFFF"),
            )
        )
        self.init_components()
        self.page.add(self.main_layout)

    def init_components(self):
        self._result_rows: list[dict] = []
        self._current_page: int = 0
        self._total_pages: int = 4
        self._cards_per_page: int = CARDS_PER_PAGE
        self._selected_key: Optional[tuple[str, str]] = None
        self._focused_field: Optional[ft.TextField] = None

        self.header_text = ft.Text(
            value="Semantic Search Movie Lookup",
            text_align=ft.TextAlign.CENTER,
            font_family="Consolas",
            weight=ft.FontWeight.BOLD,
            color="#FFFFFF",
        )

        self.input_field = ft.TextField(
            label="I remember a movie where...",
            width=600,
            multiline=True,
            min_lines=4,
            max_lines=8,
            align_label_with_hint=True,
        )

        self.min_year_field = ft.TextField(
            label="From year",
            width=130,
            keyboard_type=ft.KeyboardType.NUMBER,
        )
        self.max_year_field = ft.TextField(
            label="To year",
            width=130,
            keyboard_type=ft.KeyboardType.NUMBER,
        )

        for field in (self.input_field, self.min_year_field, self.max_year_field):
            self._bind_label_behavior(field)
        self.filters_row = ft.Row(
            controls=[self.min_year_field, self.max_year_field],
            alignment=ft.MainAxisAlignment.CENTER,
        )

        self.submit_button = ft.Button(
            content=ft.Text(
                "Submit query", text_align=ft.TextAlign.CENTER, color="#FFFFFF"
            ),
            bgcolor="#1f8f5b",
            color="#FFFFFF",
            on_click=self.on_click_submit,
            width=170,
            disabled=True,
        )

        self.status_spinner = ft.ProgressRing(
            visible=False, width=18, height=18, color="#a8ffb2"
        )
        self.status_text = ft.Text(
            value="",
            font_family="Consolas",
            size=14,
            color="#a8ffb2",
            text_align=ft.TextAlign.CENTER,
        )
        self.status_row = ft.Row(
            controls=[self.status_spinner, self.status_text],
            alignment=ft.MainAxisAlignment.CENTER,
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
                self.filters_row,
                self.submit_button,
                self.status_row,
                ft.Divider(height=20, color="transparent"),
                # Fixed height sized to fit two full rows of cards so the whole
                # grid sits in view (no internal scroll / "bobbing"). Two rows of
                # CARD_HEIGHT plus the inter-row spacing, with a little breathing room.
                ft.Container(
                    content=self.results_grid,
                    height=2 * CARD_HEIGHT + 40,
                ),
                ft.Divider(height=10, color="transparent"),
                self.pagination_bar,
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            expand=True,
        )

        # --- Movie detail dialog (plot + similar flashcards) ---
        self.dialog_title = ft.Text(
            font_family="Consolas", weight=ft.FontWeight.BOLD, color="#FFFFFF"
        )
        self.dialog_meta = ft.Text(font_family="Consolas", size=13, color="#FFFFFF")
        self.dialog_plot = ft.Text(
            font_family="Consolas", size=14, color="#FFFFFF", selectable=True
        )
        self.similar_section_title = ft.Text(
            value="Similar movies",
            font_family="Consolas",
            weight=ft.FontWeight.BOLD,
            size=14,
            color="#FFFFFF",
        )
        self.similar_grid = ft.Column(controls=[], spacing=10)
        self.similar_loading = ft.ProgressRing(width=24, height=24, color="#a8ffb2")

        self.plot_dialog = ft.AlertDialog(
            bgcolor="#171a26",
            modal=True,
            title=self.dialog_title,
            content=ft.Container(
                width=DIALOG_WIDTH,
                height=560,
                content=ft.Column(
                    controls=[
                        self.dialog_meta,
                        ft.Divider(height=8, color="#2b3142"),
                        self.dialog_plot,
                        ft.Divider(height=12, color="#2b3142"),
                        self.similar_section_title,
                        self.similar_loading,
                        self.similar_grid,
                    ],
                    scroll=ft.ScrollMode.AUTO,
                    spacing=8,
                    expand=True,
                ),
            ),
            actions=[
                ft.TextButton(
                    "Close",
                    on_click=lambda _: self.close_dialog(),
                    style=ft.ButtonStyle(color="#FFFFFF"),
                )
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.page.overlay.append(self.plot_dialog)

    def _bind_label_behavior(self, field: ft.TextField):
        field.border_color = "#2b3142"
        field.focused_border_color = LABEL_GREEN
        field.focus_color = LABEL_GREEN
        field.color = "#FFFFFF"
        field.on_focus = self._on_field_focus
        field.on_blur = self._on_field_blur
        field.on_change = self._on_field_change
        self._refresh_field_label(field)

    def _refresh_field_label(self, field: ft.TextField):
        has_value = bool((field.value or "").strip())
        focused = field is self._focused_field
        floated = has_value or focused
        field.label_style = ft.TextStyle(
            color=LABEL_GREEN if floated else LABEL_GREY,
            font_family=FIELD_FONT,
        )

    def _on_field_focus(self, e: ft.ControlEvent):
        self._focused_field = e.control
        self._refresh_field_label(e.control)
        self.page.update()

    def _on_field_blur(self, e: ft.ControlEvent):
        self._focused_field = None
        self._refresh_field_label(e.control)
        self.page.update()

    def _on_field_change(self, e: ft.ControlEvent):
        self._refresh_field_label(e.control)
        self.page.update()

    def close_dialog(self):
        self.plot_dialog.open = False
        self._selected_key = None
        self._render_current_page()
        self.page.update()

    def set_app_status(self, message: Optional[str]):
        """Show startup status. `None` means ready."""
        loading = message is not None
        self.status_spinner.visible = loading
        self.status_text.value = message or ""
        self.submit_button.disabled = loading
        self.page.update()

    @staticmethod
    def _parse_year(field: ft.TextField) -> Optional[int]:
        value = (field.value or "").strip()
        return int(value) if value.isdigit() else None

    async def on_click_submit(self, e):
        query = self.input_field.value.strip()
        if not query:
            return

        filters = {
            "min_year": self._parse_year(self.min_year_field),
            "max_year": self._parse_year(self.max_year_field),
        }

        self.submit_button.content = ft.Row(
            [
                ft.ProgressRing(width=16, height=16, color="#FFFFFF"),
                ft.Text("Searching...", color="#FFFFFF"),
            ],
            tight=True,
            alignment=ft.MainAxisAlignment.CENTER,
        )
        self.submit_button.disabled = True
        self.page.update()

        await asyncio.to_thread(self.submit_fun, query, filters)

    async def select_movie(
        self,
        title: str,
        year: str,
        plot: str,
        genre: str = "Unknown",
        confidence: str = "",
    ):
        self._selected_key = (title, year)

        self.dialog_title.value = f"{title} ({year})"
        meta_parts = []
        if genre and genre != "Unknown":
            meta_parts.append(f"Genre: {genre}")
        if confidence:
            meta_parts.append(f"Match: {confidence}")
        self.dialog_meta.value = (
            "  ·  ".join(meta_parts) if meta_parts else f"Year: {year}"
        )
        self.dialog_plot.value = plot if plot else "No plot available."

        self.similar_loading.visible = True
        self.similar_grid.controls.clear()
        self.plot_dialog.open = True
        self._render_current_page()
        self.page.update()

        similar_df = await asyncio.to_thread(self.similar_fun, title, year)
        self.similar_loading.visible = False
        self._render_similar_cards(similar_df)
        self.page.update()

    def _sim_color(self, sim: float) -> str:
        t = max(0.0, min(1.0, float(sim)))
        if t < 0.5:
            r, g = 220, int(220 * (t / 0.5))
        else:
            r, g = int(220 * (1 - (t - 0.5) / 0.5)), 200
        return f"#{r:02x}{g:02x}40"

    def _is_selected(self, title: str, year: str) -> bool:
        return self._selected_key == (title, year)

    def _card_border(self, selected: bool) -> ft.Border:
        if selected:
            return ft.Border.all(2, "#a8ffb2")
        return ft.Border.all(1, "#2b3142")

    @staticmethod
    def _movie_dict_from_row(row) -> dict:
        """Normalise a search/similar dataframe row into the shared card dict."""
        title = str(row.get("Title", "Unknown"))
        year = str(row.get("Release Year", "Unknown"))
        plot = str(row.get("Plot", ""))
        genre = str(row.get("Genre", "Unknown"))
        confidence = str(row.get("Confidence", row.get("Similarity", "")))
        sim_raw = row.get("Confidence", row.get("Similarity", 0))
        if isinstance(sim_raw, str):
            sim_val = float(sim_raw.strip("%")) / 100 if sim_raw.strip("%") else 0.0
        else:
            sim_val = float(sim_raw or 0)
        return {
            "title": title,
            "year": year,
            "plot": plot,
            "genre": genre,
            "confidence": confidence,
            "sim": sim_val,
        }

    def _build_card(self, movie: dict) -> ft.Container:
        title = movie["title"]
        year = movie["year"]
        plot = movie["plot"]
        sim = movie.get("sim", 0.0)
        selected = self._is_selected(title, year)

        sim_pct = sim * 100
        bar_color = self._sim_color(sim)
        plot_preview = (plot[:160] + "…") if len(plot) > 160 else plot

        sim_bar = ft.Container(
            width=10,
            border_radius=ft.BorderRadius.all(4),
            bgcolor="#2b3142",
            content=ft.Column(
                controls=[
                    ft.Container(expand=int(100 - round(sim_pct))),
                    ft.Container(
                        bgcolor=bar_color,
                        border_radius=ft.BorderRadius.all(4),
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
                            border_radius=ft.BorderRadius.all(4),
                            bgcolor=bar_color,
                        ),
                        ft.Text(
                            f"Similarity: {sim_pct:.1f}%",
                            font_family="Consolas",
                            size=12,
                            color="#FFFFFF",
                        ),
                    ],
                    spacing=6,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Divider(height=6, color="#2b3142"),
                ft.Text(
                    plot_preview,
                    font_family="Consolas",
                    size=11,
                    color="#FFFFFF",
                    max_lines=5,
                    overflow=ft.TextOverflow.ELLIPSIS,
                    expand=True,
                ),
                ft.Text(
                    "Show full summary →",
                    font_family="Consolas",
                    size=11,
                    color="#FFFFFF",
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
            bgcolor="#171a26" if not selected else "#1a2e24",
            border=self._card_border(selected),
            border_radius=10,
            padding=ft.Padding.all(12),
            width=240,
            height=CARD_HEIGHT,
            on_click=lambda e, m=movie: self.page.run_task(
                self.select_movie,
                m["title"],
                m["year"],
                m["plot"],
                m.get("genre", "Unknown"),
                m.get("confidence", ""),
            ),
            ink=True,
        )

    def _render_similar_cards(self, similar_df: pd.DataFrame):
        self.similar_grid.controls.clear()
        if similar_df is None or similar_df.empty:
            self.similar_grid.controls.append(
                ft.Text(
                    "No similar movies found.",
                    font_family="Consolas",
                    size=12,
                    color="#FFFFFF",
                )
            )
            return

        cards = [
            self._build_card(self._movie_dict_from_row(row))
            for _, row in similar_df.iterrows()
        ]
        for i in range(0, len(cards), GRID_COLUMNS):
            self.similar_grid.controls.append(
                ft.Row(
                    controls=cards[i : i + GRID_COLUMNS],
                    spacing=14,
                    alignment=ft.MainAxisAlignment.CENTER,
                )
            )

    def _on_prev_page(self, e):
        if self._current_page > 0:
            self._current_page -= 1
            self._render_current_page()

    def _on_next_page(self, e):
        max_page = (
            min(
                self._total_pages,
                (len(self._result_rows) + self._cards_per_page - 1)
                // self._cards_per_page,
            )
            - 1
        )
        if self._current_page < max_page:
            self._current_page += 1
            self._render_current_page()

    def _update_pagination_controls(self):
        num_pages = min(
            self._total_pages,
            (len(self._result_rows) + self._cards_per_page - 1) // self._cards_per_page,
        )
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
                    border_radius=ft.BorderRadius.all(5),
                    bgcolor="#a8ffb2" if is_active else "#2b3142",
                    border=ft.Border.all(1, "#a8ffb2" if is_active else "#2b3142"),
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
        page_movies = self._result_rows[start : start + self._cards_per_page]

        cards = [self._build_card(m) for m in page_movies]
        while len(cards) < self._cards_per_page:
            cards.append(ft.Container(width=240, height=CARD_HEIGHT))

        for i in range(0, self._cards_per_page, GRID_COLUMNS):
            self.results_grid.controls.append(
                ft.Row(
                    controls=cards[i : i + GRID_COLUMNS],
                    spacing=14,
                    alignment=ft.MainAxisAlignment.CENTER,
                )
            )

        self._update_pagination_controls()
        self.page.update()

    def update_results(self, results_df: pd.DataFrame):
        max_results = self._total_pages * self._cards_per_page

        self._result_rows = []
        for _, row in results_df.head(max_results).iterrows():
            self._result_rows.append(self._movie_dict_from_row(row))

        self._current_page = 0
        self._render_current_page()

        self.submit_button.content = ft.Text(
            "Submit query", text_align=ft.TextAlign.CENTER, color="#FFFFFF"
        )
        self.submit_button.disabled = False
        self.page.update()
