"""Table components for the dashboard."""

import re
from typing import Sequence

from dash import html
import dash_bootstrap_components as dbc


def _get_strategy_group_order() -> dict[str, int]:
    """Build strategy group order from the StrategyRegistry.

    Derives ordering from ``StrategyRegistry.list_strategies()`` (base names in
    insertion order), so new strategies are picked up automatically without
    needing to maintain a hardcoded dict.
    """
    from conformal_relevance.icl.strategies import StrategyRegistry

    order: dict[str, int] = {"": 0}  # icl0 baseline always first
    for i, name in enumerate(StrategyRegistry.list_strategies(), start=1):
        order[name] = i
    # Local strategies (not in registry) sort after registry strategies
    n = len(order)
    for j, name in enumerate(["bm25", "knn"], start=n):
        order[name] = j
    # MoE ensembles sort after all individual strategies
    n = len(order)
    for k in range(2, 7):  # moe2 through moe6
        order[f"moe{k}"] = n + k - 2
    order["moe"] = n + 10  # legacy entries without K, sort last
    return order


def _strategy_sort_key(label: str) -> tuple[int, int]:
    """Return (strategy_group_order, icl_n) for sorting performance matrix rows."""
    m = re.match(r"icl(\d+)(?:_(.+))?$", label)
    if not m:
        return (99, 0)
    n = int(m.group(1))
    strategy = m.group(2) or ""
    order = _get_strategy_group_order()
    return (order.get(strategy, 50), n)


def create_runs_table(runs_data: Sequence[dict]) -> html.Div:
    """Create a table displaying runs.

    Args:
        runs_data: List of run dictionaries.

    Returns:
        Dash HTML component.
    """
    if not runs_data:
        return html.P("No runs to display", className="text-muted")

    header = html.Thead(html.Tr([
        html.Th("Run ID"),
        html.Th("Name"),
        html.Th("Dataset"),
        html.Th("Model"),
        html.Th("Strategy"),
        html.Th("Status"),
        html.Th("Mean AP"),
        html.Th("Samples"),
        html.Th("Created"),
    ]))

    rows = []
    for run in runs_data[:50]:  # Limit to 50 rows
        status_class = {
            "completed": "text-success",
            "failed": "text-danger",
            "running": "text-warning",
            "pending": "text-secondary",
        }.get(run.get("status", ""), "")

        mean_ap = run.get("mean_ap")
        mean_ap_str = f"{mean_ap:.3f}" if mean_ap is not None else "N/A"

        created_at = run.get("created_at", "")
        if hasattr(created_at, "strftime"):
            created_at = created_at.strftime("%Y-%m-%d %H:%M")

        rows.append(html.Tr([
            html.Td(html.A(
                f"#{run.get('run_id', 'N/A')}",
                href=f"/run?id={run.get('run_id')}",
            )),
            html.Td((run.get("run_name") or "")[:30] or "-"),
            html.Td(run.get("dataset_name", "N/A")),
            html.Td(run.get("model_name", "N/A")),
            html.Td(run.get("icl_strategy_name", "-") or "-"),
            html.Td(run.get("status", "N/A"), className=status_class),
            html.Td(mean_ap_str),
            html.Td(run.get("n_samples", 0)),
            html.Td(created_at),
        ]))

    body = html.Tbody(rows)

    return dbc.Table(
        [header, body],
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        size="sm",
    )


def create_scores_table(
    scores_data: Sequence[dict],
    show_sample_id: bool = True,
) -> html.Div:
    """Create a table displaying sample scores.

    Args:
        scores_data: List of score dictionaries.
        show_sample_id: Whether to show sample ID column.

    Returns:
        Dash HTML component.
    """
    if not scores_data:
        return html.P("No scores to display", className="text-muted")

    headers = []
    if show_sample_id:
        headers.append(html.Th("Sample ID"))
    headers.extend([
        html.Th("Intent"),
        html.Th("Status"),
        html.Th("AP"),
        html.Th("Retries"),
    ])

    header = html.Thead(html.Tr(headers))

    rows = []
    for score in scores_data[:100]:  # Limit to 100 rows
        status_class = "text-success" if score.get("status") == "success" else "text-danger"

        ap = score.get("average_precision")
        ap_str = f"{ap:.3f}" if ap is not None else "N/A"

        cells = []
        if show_sample_id:
            cells.append(html.Td(score.get("sample_id", "N/A")))
        cells.extend([
            html.Td(str(score.get("intent", ""))[:40]),
            html.Td(score.get("status", "N/A"), className=status_class),
            html.Td(ap_str),
            html.Td(score.get("retry_count", 0)),
        ])

        rows.append(html.Tr(cells))

    body = html.Tbody(rows)

    return dbc.Table(
        [header, body],
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        size="sm",
    )


def create_error_details_table(
    errors: Sequence[dict],
    max_rows: int = 100,
) -> html.Div:
    """Create a table displaying error details.

    Args:
        errors: List of error dictionaries with keys like category, error_type,
            sample_id, message, severity, etc.
        max_rows: Maximum number of rows to display.

    Returns:
        Dash HTML component.
    """
    if not errors:
        return html.P("No errors to display", className="text-muted")

    header = html.Thead(html.Tr([
        html.Th("Sample ID"),
        html.Th("Category"),
        html.Th("Error Type"),
        html.Th("Severity"),
        html.Th("Intent"),
        html.Th("Message"),
        html.Th("Retries"),
    ]))

    # Severity badge classes
    severity_badges = {
        "low": "badge bg-success",
        "medium": "badge bg-warning text-dark",
        "high": "badge bg-danger",
        "critical": "badge bg-dark",
    }

    # Category badge classes
    category_badges = {
        "api_error": "badge bg-danger",
        "rate_limit": "badge bg-warning text-dark",
        "timeout": "badge bg-warning text-dark",
        "parsing_error": "badge bg-info",
        "validation_error": "badge bg-primary",
        "context_length": "badge bg-secondary",
        "content_filter": "badge bg-danger",
        "authentication": "badge bg-dark",
        "network_error": "badge bg-info",
        "unknown": "badge bg-secondary",
    }

    rows = []
    for error in errors[:max_rows]:
        category = error.get("category", "unknown")
        severity = error.get("severity", "unknown")
        message = error.get("message", "")
        if len(message) > 100:
            message = message[:97] + "..."

        rows.append(html.Tr([
            html.Td(error.get("sample_id", "N/A")),
            html.Td(html.Span(
                category,
                className=category_badges.get(category, "badge bg-secondary"),
            )),
            html.Td(error.get("error_type", "N/A")),
            html.Td(html.Span(
                severity,
                className=severity_badges.get(severity, "badge bg-secondary"),
            )),
            html.Td(str(error.get("intent", ""))[:30] or "-"),
            html.Td(message, title=error.get("message", "")),
            html.Td(error.get("retry_count", 0)),
        ]))

    body = html.Tbody(rows)

    table = dbc.Table(
        [header, body],
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        size="sm",
    )

    if len(errors) > max_rows:
        return html.Div([
            table,
            html.P(
                f"Showing {max_rows} of {len(errors)} errors",
                className="text-muted mt-2",
            ),
        ])

    return table


def create_error_patterns_table(
    patterns: Sequence[dict],
) -> html.Div:
    """Create a table displaying error patterns with suggested actions.

    Args:
        patterns: List of pattern dictionaries with keys like pattern, category,
            count, suggested_action.

    Returns:
        Dash HTML component.
    """
    if not patterns:
        return html.P("No error patterns detected", className="text-muted")

    header = html.Thead(html.Tr([
        html.Th("#"),
        html.Th("Error Pattern"),
        html.Th("Category"),
        html.Th("Count"),
        html.Th("Suggested Action"),
    ]))

    rows = []
    for i, pattern in enumerate(patterns[:20], 1):
        rows.append(html.Tr([
            html.Td(i),
            html.Td(html.Code(pattern.get("pattern", "N/A"))),
            html.Td(pattern.get("category", "N/A")),
            html.Td(html.Strong(pattern.get("count", 0))),
            html.Td(pattern.get("suggested_action", "N/A")),
        ]))

    body = html.Tbody(rows)

    return dbc.Table(
        [header, body],
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
    )


def create_comparison_table(
    comparison_results: dict[str, dict],
    baseline: str,
) -> html.Div:
    """Create a table displaying statistical comparison results.

    Args:
        comparison_results: Dict mapping strategy names to comparison results.
        baseline: Name of baseline strategy.

    Returns:
        Dash HTML component.
    """
    if not comparison_results:
        return html.P("No comparison results", className="text-muted")

    header = html.Thead(html.Tr([
        html.Th("Strategy"),
        html.Th("Best MAP"),
        html.Th("vs Baseline"),
    ]))

    rows = []

    # Add baseline row first
    if baseline in comparison_results:
        base_result = comparison_results[baseline]
        rows.append(html.Tr([
            html.Td(html.Strong(f"{baseline} (baseline)")),
            html.Td(f"{base_result.get('mean_ap', 0):.3f}"),
            html.Td("-"),
        ], className="table-secondary"))

    # Add other strategies sorted by mean_ap descending
    others = [
        (name, result) for name, result in comparison_results.items()
        if name != baseline
    ]
    others.sort(key=lambda x: x[1].get("mean_ap", 0), reverse=True)

    for name, result in others:
        mean_ap = result.get("mean_ap", 0)
        comparison = result.get("comparison", {})

        mean_diff = comparison.get("mean_diff", 0)
        diff_str = f"{mean_diff:+.3f}" if mean_diff else "-"
        row_class = "table-success" if mean_diff > 0.02 else ""
        row_class = "table-danger" if mean_diff < -0.02 else row_class

        rows.append(html.Tr([
            html.Td(name),
            html.Td(f"{mean_ap:.3f}"),
            html.Td(diff_str),
        ], className=row_class))

    body = html.Tbody(rows)

    return dbc.Table(
        [header, body],
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
    )


def create_performance_matrix_table(
    matrix_data: Sequence[dict],
) -> html.Div:
    """Create a performance matrix table.

    Rows are strategies, columns are datasets. Cells show mean_ap
    with a delta indicator from the previous run.

    Args:
        matrix_data: List of dicts from get_performance_matrix().

    Returns:
        Dash HTML component.
    """
    if not matrix_data:
        return html.P("No data available for performance matrix", className="text-muted")

    # Pivot data: collect unique strategies (rows) and datasets (columns)
    datasets: list[str] = []
    strategies: list[str] = []
    lookup: dict[tuple[str, str], dict] = {}

    for entry in matrix_data:
        ds = entry.get("dataset_name", "Unknown")
        strat = entry.get("strategy_label", "Unknown")
        if ds not in datasets:
            datasets.append(ds)
        if strat not in strategies:
            strategies.append(strat)
        lookup[(ds, strat)] = entry

    sorted_datasets = sorted(datasets)
    sorted_strategies = sorted(strategies, key=_strategy_sort_key)

    # Find best and 2nd-best mean_ap per dataset column among ICL strategies only
    # (exclude icl0 baseline from the ranking)
    _ICL0_LABEL = "icl0"
    best_per_ds: dict[str, tuple[float | None, float | None]] = {}  # ds -> (best, 2nd_best)
    icl0_per_ds: dict[str, float | None] = {}  # ds -> icl0 mean_ap
    for ds in sorted_datasets:
        ap_values = []
        for strat in sorted_strategies:
            entry = lookup.get((ds, strat))
            if entry and entry.get("mean_ap") is not None:
                if strat == _ICL0_LABEL:
                    icl0_per_ds[ds] = entry["mean_ap"]
                else:
                    ap_values.append(entry["mean_ap"])
        unique_sorted = sorted(set(ap_values), reverse=True)
        best = unique_sorted[0] if len(unique_sorted) >= 1 else None
        second = unique_sorted[1] if len(unique_sorted) >= 2 else None
        best_per_ds[ds] = (best, second)

    # Build header row: Strategy | dataset1 | dataset2 | ...
    header_cells = [html.Th("Strategy")]
    for ds in sorted_datasets:
        header_cells.append(html.Th(ds))
    header = html.Thead(html.Tr(header_cells))

    # Build body rows: one row per strategy
    rows = []
    for strat in sorted_strategies:
        cells = [html.Td(html.Strong(strat))]
        for ds in sorted_datasets:
            entry = lookup.get((ds, strat))
            if entry is None or entry.get("mean_ap") is None:
                cells.append(html.Td("-", className="text-muted"))
                continue

            mean_ap = entry["mean_ap"]
            delta = entry.get("delta")

            # Format the mean_ap value
            ap_text = f"{mean_ap:.3f}"

            # Build delta indicator
            if delta is not None and delta != 0:
                if delta > 0:
                    indicator = html.Span(
                        f" \u2191{delta:+.3f}",
                        style={"color": "var(--bs-success)", "fontSize": "0.85em"},
                    )
                else:
                    indicator = html.Span(
                        f" \u2193{delta:+.3f}",
                        style={"color": "var(--bs-danger)", "fontSize": "0.85em"},
                    )
            elif delta is not None and delta == 0:
                indicator = html.Span(
                    " \u2014",
                    style={"color": "var(--bs-secondary)", "fontSize": "0.85em"},
                )
            else:
                indicator = html.Span("")

            # Apply styling per column:
            # - ICL strategies: bold = best, underline = 2nd best
            # - ICL0: bold + red if it beats the best ICL strategy
            best, second = best_per_ds.get(ds, (None, None))
            is_icl0 = strat == _ICL0_LABEL
            if is_icl0:
                if best is not None and mean_ap > best:
                    cell_content = html.Td(
                        [html.Strong(ap_text, style={"color": "var(--bs-danger)"}), indicator],
                    )
                else:
                    cell_content = html.Td([ap_text, indicator])
            elif best is not None and mean_ap == best:
                cell_content = html.Td([html.Strong(ap_text), indicator])
            elif second is not None and mean_ap == second:
                cell_content = html.Td(
                    [html.Span(ap_text, style={
                        "border": "2px solid var(--bs-info)",
                        "borderRadius": "4px",
                        "padding": "1px 4px",
                    }), indicator],
                )
            else:
                cell_content = html.Td([ap_text, indicator])

            cells.append(cell_content)

        rows.append(html.Tr(cells))

    body = html.Tbody(rows)

    return dbc.Table(
        [header, body],
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
    )


def create_strategy_icl_table(
    data: Sequence[dict],
    baseline_ap: float | None = None,
) -> html.Div:
    """Create a performance comparison table: rows=ICL counts, columns=strategies.

    Each cell shows the best single run's mean_ap for that (strategy, n_icl) pair.
    The top 3 values across the entire table get a blue gradient background.
    If baseline_ap is provided, it is shown as a benchmark row spanning all columns.

    Args:
        data: List of dicts with 'strategy_name', 'n_icl', 'best_mean_ap'.
        baseline_ap: Optional ICL0 baseline MAP to show as benchmark.

    Returns:
        Dash HTML component.
    """
    if not data and baseline_ap is None:
        return html.P("No data available", className="text-muted")

    # Collect unique strategies (columns) and ICL counts (rows)
    strategies: list[str] = []
    icl_counts: list[int] = []
    lookup: dict[tuple[str, int], float] = {}

    for entry in data:
        strat = entry["strategy_name"]
        n_icl = entry["n_icl"]
        ap = entry["best_mean_ap"]
        if strat not in strategies:
            strategies.append(strat)
        if n_icl not in icl_counts:
            icl_counts.append(n_icl)
        lookup[(strat, n_icl)] = ap

    sorted_strategies = sorted(strategies)
    sorted_icl_counts = sorted(icl_counts)

    # Collect all AP values to find top 3
    all_ap_values = sorted(
        [v for v in lookup.values() if v is not None],
        reverse=True,
    )
    # Top 3 unique values
    top_3_thresholds = []
    seen = set()
    for v in all_ap_values:
        if v not in seen:
            top_3_thresholds.append(v)
            seen.add(v)
        if len(top_3_thresholds) == 3:
            break

    # Top 3 ranking colors using Bootstrap CSS vars (adapt to light/dark)
    _TOP_RANK_COLORS = [
        "var(--bs-primary)",    # Rank 1
        "var(--bs-info)",       # Rank 2
        "var(--bs-secondary)",  # Rank 3
    ]

    def _get_rank_style(ap: float) -> dict | None:
        for i, threshold in enumerate(top_3_thresholds):
            if ap == threshold:
                return {"color": _TOP_RANK_COLORS[i], "fontWeight": "bold"}
        return None

    # Build header: top-left cell shows baseline, then strategy columns
    if baseline_ap is not None:
        header_cells = [html.Th(
            [html.Strong(f"ICL0: {baseline_ap:.3f}")],
            style={"whiteSpace": "nowrap"},
        )]
    else:
        header_cells = [html.Th("")]
    for strat in sorted_strategies:
        header_cells.append(html.Th(strat))
    header = html.Thead(html.Tr(header_cells))

    # Data rows (no separate baseline row)
    rows = []
    for n_icl in sorted_icl_counts:
        cells = [html.Td(html.Strong(f"ICL{n_icl}"))]
        for strat in sorted_strategies:
            ap = lookup.get((strat, n_icl))
            if ap is None:
                cells.append(html.Td("-", className="text-muted"))
                continue

            style = _get_rank_style(ap) or {}
            cells.append(html.Td(f"{ap:.3f}", style=style))

        rows.append(html.Tr(cells))

    body = html.Tbody(rows)

    return dbc.Table(
        [header, body],
        striped=False,
        bordered=True,
        hover=True,
        responsive=True,
    )


def create_cross_model_table(models: list[str], rows: list[dict]) -> html.Div:
    """Create a cross-model comparison table (strategy × model MAP).

    Args:
        models: Ordered list of model names (primary model first).
        rows: List of dicts with keys 'strategy_label' and 'maps' ({model_name: mean_ap}).

    Returns:
        Dash HTML table component.
    """
    if not models or not rows:
        return html.Div()

    # Header: Strategy | model1 | model2 | ...
    header_cells = [html.Th("Strategy")] + [html.Th(m) for m in models]
    header = html.Thead(html.Tr(header_cells))

    table_rows = []
    for row in rows:
        maps = row["maps"]
        # Find the highest AP in this row to bold it
        present_values = [maps[m] for m in models if m in maps]
        max_ap = max(present_values) if present_values else None

        cells = [html.Td(row["strategy_label"])]
        for model in models:
            ap = maps.get(model)
            if ap is None:
                cells.append(html.Td("\u2014", style={"color": "#999"}))
            else:
                is_best = max_ap is not None and abs(ap - max_ap) < 1e-6
                style = {"fontWeight": "bold"} if is_best else {}
                cells.append(html.Td(f"{ap:.3f}", style=style))

        table_rows.append(html.Tr(cells))

    body = html.Tbody(table_rows)

    return dbc.Table(
        [header, body],
        striped=False,
        bordered=True,
        hover=True,
        responsive=True,
    )
