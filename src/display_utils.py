"""
Display utilities: styled pandas tables and coloured parameter panels.

These helpers are used across the package to produce consistent, presentation-
quality output in Jupyter notebooks and the terminal. The styling here is
deliberately minimal (CSS + ANSI) so that it renders well both in the thesis
appendix screenshots and in plain console execution.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from IPython.display import HTML, display


# ── Pandas styler presets ──────────────────────────────────────────────

_CAPTION_STYLE = [
    {
        "selector": "caption",
        "props": [
            ("color", "#2c3e50"),
            ("font-size", "18px"),
            ("font-weight", "bold"),
            ("text-align", "left"),
            ("padding", "10px"),
        ],
    }
]


def styled_stats(
    df: pd.DataFrame,
    caption: str = "Data Statistics",
    cmap: str = "Blues",
    fmt: str = "{:.4f}",
) -> Any:
    """
    Return a Styler object matching the thesis' Data Statistics look:
    bold left-aligned caption and a blue gradient background.

    Parameters
    ----------
    df : pd.DataFrame
        Numeric frame to display. For descriptive-statistics tables the index
        holds metric names (mean, std, ...) and columns hold tenors.
    caption : str
        Table caption shown above the grid.
    cmap : str
        matplotlib colormap for the background gradient.
    fmt : str
        Python format string applied to every cell.
    """
    return (
        df.style.set_caption(caption)
        .set_table_styles(_CAPTION_STYLE)
        .background_gradient(cmap=cmap, axis=None)
        .format(fmt)
    )


# ── Parameter / section panels ─────────────────────────────────────────

def _fmt_value(v: Any) -> str:
    """Format a parameter value for compact display."""
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:.6f}" if abs(v) < 1 else f"{v:.4f}"
    if isinstance(v, np.ndarray):
        if v.ndim == 1 and v.size <= 6:
            return "[" + ", ".join(f"{x:.4f}" for x in v) + "]"
        return f"ndarray  shape={v.shape}  mean={v.mean():.4f}  std={v.std():.4f}"
    return str(v)


_PARAMS_CSS = """
<style>
.pbox-wrap {{
    display: table;
    margin: 10px 0 16px 0;
}}
.pbox {{
    font-family: "SF Mono", "Fira Mono", Consolas, monospace;
    font-size: 12.5px;
    border-top: 2px solid {accent};
    border-left: 0.5px solid #d0d7de;
    border-right: 0.5px solid #d0d7de;
    border-bottom: 0.5px solid #d0d7de;
    min-width: 320px;
}}
.pbox .ph {{
    background: {accent};
    color: #ffffff;
    padding: 5px 14px 4px 14px;
    font-size: 11.5px;
    font-weight: 500;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    white-space: nowrap;
}}
.pbox .ps {{
    background: #f6f8fa;
    color: #57606a;
    padding: 3px 14px;
    font-size: 11px;
    border-bottom: 0.5px solid #d0d7de;
    white-space: nowrap;
}}
.pbox table {{
    border-collapse: collapse;
    width: auto;
}}
.pbox tr:nth-child(even) td {{ background: #f6f8fa; }}
.pbox tr:nth-child(odd)  td {{ background: #ffffff; }}
.pbox td {{
    padding: 4px 14px 4px 14px;
    color: #1f2328;
    white-space: nowrap;
    border: none;
    text-align: left;
}}
.pbox td.k {{
    color: #57606a;
    font-weight: 400;
    padding-right: 32px;
}}
.pbox td.v {{
    font-weight: 500;
    text-align: right;
}}
</style>
"""


def print_params_box(
    params: dict,
    title: str,
    model_key: str = "default",
    subtitle: str | None = None,
) -> None:
    """Render a compact, terminal-style parameter table in Jupyter."""
    accent = '#0969da'
    css = _PARAMS_CSS.format(accent=accent)

    rows = "".join(
        f'<tr><td class="k">{k}</td><td class="v">{_fmt_value(v)}</td></tr>'
        for k, v in params.items()
        if not k.startswith("_")
    )
    sub_html = f'<div class="ps">{subtitle}</div>' if subtitle else ""
    html = f"""{css}
<div class="pbox-wrap">
<div class="pbox">
  <div class="ph">{title}</div>
  {sub_html}
  <table>{rows}</table>
</div>
</div>"""
    display(HTML(html))


# ── Results comparison table ───────────────────────────────────────────

def results_table(
    results: dict[str, dict[str, float]],
    caption: str = "Model Comparison",
    higher_is_better: set[str] | None = None,
    fmt: str = "{:.4f}",
) -> Any:
    """
    Render a side-by-side comparison of models across metrics, with best
    values highlighted in bold.

    Parameters
    ----------
    results : dict
        {model_name: {metric_name: value, ...}, ...}.
    caption : str
        Table title.
    higher_is_better : set of metric names
        Metrics where larger is better (e.g. 'Kupiec POF p-value', 'Coverage').
        Metrics not listed here are assumed to be 'lower is better'.
    fmt : str
        Cell format string.
    """
    higher_is_better = higher_is_better or set()
    df = pd.DataFrame(results).T  # rows = models, cols = metrics

    def _highlight_best(col: pd.Series) -> list[str]:
        best = col.max() if col.name in higher_is_better else col.min()
        return [
            "font-weight: 700; color: #0969da;" if v == best else ""
            for v in col
        ]

    return (
        df.style.set_caption(caption)
        .set_table_styles(_CAPTION_STYLE)
        .apply(_highlight_best, axis=0)
        .format(fmt)
    )