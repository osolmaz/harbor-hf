from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Literal, Protocol, runtime_checkable

from harbor_hf_space.config import SpaceConfig
from harbor_hf_space.data import (
    AnonymousHubReader,
    DatasetLoader,
    PresentationError,
    Snapshot,
)
from harbor_hf_space.views import (
    VIEW_COLUMNS,
    Cell,
    ViewFilters,
    ViewSet,
    build_views,
    summary,
)

_VIEW_NAMES = tuple(VIEW_COLUMNS)
_EMPTY_TABLES = tuple([] for _ in _VIEW_NAMES)
SPACE_CSS = """
.gradio-container { max-width: 1500px !important; margin: 0 auto; }
.prose h1 { font-size: 1.65rem; font-weight: 600; letter-spacing: -0.02em; }
.prose h2, .prose h3 { font-weight: 550; }
.quiet-note { color: var(--body-text-color-subdued); }
.table-wrap { min-height: 28rem; }
"""


class Launchable(Protocol):
    def launch(self, *, show_error: bool, css: str) -> object: ...


@runtime_checkable
class Clickable(Protocol):
    def click(
        self,
        fn: Callable[..., tuple[object, ...]],
        *,
        inputs: Sequence[object],
        outputs: Sequence[object],
        api_name: Literal[False],
    ) -> object: ...


class SnapshotLoader(Protocol):
    def load(self) -> Snapshot: ...


def create_app(
    *, config: SpaceConfig | None = None, loader: SnapshotLoader | None = None
) -> Launchable:
    import gradio as gr

    try:
        selected_config = SpaceConfig.from_env() if config is None else config
    except ValueError as error:
        with gr.Blocks(title="Harbor results configuration") as app:
            gr.Markdown("# Harbor results")
            gr.Markdown(
                "This read-only Space is not configured. Set the public index "
                f"Dataset environment variable.\n\n`{error}`"
            )
        return app

    selected_loader = (
        DatasetLoader(selected_config, AnonymousHubReader())
        if loader is None
        else loader
    )
    refresh_views = partial(_refresh_views, selected_loader)

    with gr.Blocks(title=selected_config.title) as app:
        gr.Markdown(f"# {selected_config.title}")
        gr.Markdown(
            "Read-only views over normalized Harbor result Datasets. Labels show "
            "both completion scope and selection kind; partial, composite, and "
            "manual results are never presented as ordinary complete runs.",
            elem_classes=["quiet-note"],
        )
        with gr.Row():
            result = gr.Dropdown(
                choices=[
                    ("All results", "all"),
                    ("Complete", "complete"),
                    ("Partial", "partial"),
                    ("Ordinary", "ordinary"),
                    ("Composite", "composite"),
                    ("Manual", "manual"),
                    ("Complete · ordinary", "complete:ordinary"),
                    ("Complete · composite", "complete:composite"),
                    ("Complete · manual", "complete:manual"),
                    ("Partial · ordinary", "partial:ordinary"),
                    ("Partial · composite", "partial:composite"),
                    ("Partial · manual", "partial:manual"),
                ],
                value="all",
                label="Result label",
            )
            campaign = gr.Textbox(label="Campaign ID", placeholder="Exact match")
            run = gr.Textbox(label="Run ID", placeholder="Exact match")
            search = gr.Textbox(
                label="Search", placeholder="Benchmark, model, agent, or ID"
            )
            refresh = gr.Button("Refresh", variant="secondary")
        status = gr.Markdown("Loading public result index…")
        outputs: list[object] = [status]
        for name in _VIEW_NAMES:
            with gr.Tab(name.title()):
                table = gr.Dataframe(
                    headers=list(VIEW_COLUMNS[name]),
                    value=[],
                    interactive=False,
                    wrap=True,
                    elem_classes=["table-wrap"],
                )
                outputs.append(table)
        inputs = [result, campaign, run, search]
        if not isinstance(refresh, Clickable):
            raise RuntimeError("the Gradio button does not expose click events")
        refresh.click(
            refresh_views,
            inputs=inputs,
            outputs=outputs,
            api_name=False,
        )
        app.load(
            refresh_views,
            inputs=inputs,
            outputs=outputs,
            api_name=False,
        )
    return app.queue(default_concurrency_limit=1, max_size=8)


def _refresh_views(
    loader: SnapshotLoader,
    result: str,
    campaign: str,
    run: str,
    search: str,
) -> tuple[object, ...]:
    try:
        snapshot = loader.load()
        views = build_views(
            snapshot,
            ViewFilters(
                result=result,
                campaign=campaign,
                run=run,
                search=search,
            ),
        )
        tables = tuple(_table_values(views, name) for name in _VIEW_NAMES)
        return (summary(snapshot, views), *tables)
    except (PresentationError, OSError, ValueError) as error:
        return (
            "**Dataset read failed.** No result rows are displayed.  "
            f"\n`{type(error).__name__}: {error}`",
            *_EMPTY_TABLES,
        )


def _table_values(views: ViewSet, name: str) -> list[list[Cell]]:
    columns = VIEW_COLUMNS[name]
    return [[record[column] for column in columns] for record in views.table(name)]
