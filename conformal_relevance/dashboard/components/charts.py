"""Chart components for the dashboard."""

from typing import Literal, Sequence

import plotly.express as px
import plotly.graph_objects as go

Theme = Literal["light", "dark"]

_PLOTLY_TEMPLATE: dict[str, str] = {
    "light": "plotly_white",
    "dark": "plotly_dark",
}


def _apply_theme(fig: go.Figure, theme: Theme = "light") -> go.Figure:
    """Apply light or dark Plotly template to a figure."""
    fig.update_layout(template=_PLOTLY_TEMPLATE.get(theme, "plotly_white"))
    if theme == "dark":
        # Force transparent legend so plotly_dark's white font shows against
        # the dark paper_bgcolor rather than any white container artifact.
        fig.update_layout(
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
        )
    return fig


def create_ap_histogram(
    ap_values: Sequence[float],
    show_mean: bool = True,
    show_ci: bool = True,
    title: str = "Average Precision Distribution",
    theme: Theme = "light",
) -> go.Figure:
    """Create an AP histogram with optional mean and CI lines.

    Args:
        ap_values: List of AP values.
        show_mean: Show mean line.
        show_ci: Show confidence interval lines.
        title: Chart title.
        theme: Color theme ("light" or "dark").

    Returns:
        Plotly Figure.
    """
    import statistics

    # Filter out None/NaN values
    clean_values = [v for v in ap_values if v is not None and v == v]

    if not clean_values:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return _apply_theme(fig, theme)

    fig = px.histogram(
        x=clean_values,
        nbins=30,
        title=title,
        labels={"x": "Average Precision", "y": "Count"},
    )

    if show_mean and clean_values:
        mean_val = statistics.mean(clean_values)
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_val:.3f}",
        )

    if show_ci and len(clean_values) > 1:
        mean_val = statistics.mean(clean_values)
        std_val = statistics.stdev(clean_values)
        n = len(clean_values)
        margin = 1.96 * (std_val / (n ** 0.5))
        ci_lower = mean_val - margin
        ci_upper = mean_val + margin

        fig.add_vrect(
            x0=ci_lower,
            x1=ci_upper,
            fillcolor="rgba(100,149,237,0.25)",
            opacity=0.3,
            line_width=0,
            annotation_text="95% CI",
        )

    fig.update_layout(showlegend=False)
    return _apply_theme(fig, theme)


def create_model_comparison_bar(
    model_metrics: dict[str, dict],
    show_error_bars: bool = True,
    title: str = "Model Comparison",
    theme: Theme = "light",
) -> go.Figure:
    """Create a bar chart comparing model performance.

    Args:
        model_metrics: Dict mapping model names to metric dicts with 'mean_ap', 'std_ap'.
        show_error_bars: Show error bars for standard deviation.
        title: Chart title.
        theme: Color theme ("light" or "dark").

    Returns:
        Plotly Figure.
    """
    if not model_metrics:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return _apply_theme(fig, theme)

    models = list(model_metrics.keys())
    mean_aps = [model_metrics[m].get("mean_ap", 0) or 0 for m in models]
    std_aps = [model_metrics[m].get("std_ap", 0) or 0 for m in models] if show_error_bars else None

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=models,
        y=mean_aps,
        error_y=dict(type="data", array=std_aps) if std_aps else None,
        marker_color="steelblue",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Model",
        yaxis_title="Mean Average Precision",
        yaxis_range=[0, 1],
    )

    return _apply_theme(fig, theme)


def create_strategy_heatmap(
    metrics_matrix: dict[str, dict[str, float]],
    title: str = "Strategy Performance Heatmap",
    theme: Theme = "light",
) -> go.Figure:
    """Create a heatmap of strategy performance.

    Args:
        metrics_matrix: Nested dict of strategy -> model -> mean_ap.
        title: Chart title.
        theme: Color theme ("light" or "dark").

    Returns:
        Plotly Figure.
    """
    if not metrics_matrix:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return _apply_theme(fig, theme)

    strategies = list(metrics_matrix.keys())
    models = list(set(m for s in metrics_matrix.values() for m in s.keys()))

    z = []
    for strategy in strategies:
        row = []
        for model in models:
            val = metrics_matrix.get(strategy, {}).get(model, None)
            row.append(val)
        z.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=models,
        y=strategies,
        colorscale="Blues" if theme == "light" else "Viridis",
        text=[[f"{v:.3f}" if v else "N/A" for v in row] for row in z],
        texttemplate="%{text}",
        showscale=True,
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Model",
        yaxis_title="Strategy",
    )

    return _apply_theme(fig, theme)


def create_intent_breakdown_bar(
    intent_metrics: Sequence[dict],
    title: str = "Performance by Intent",
    theme: Theme = "light",
) -> go.Figure:
    """Create a bar chart showing performance breakdown by intent.

    Args:
        intent_metrics: List of dicts with 'intent', 'mean_ap', 'n_samples'.
        title: Chart title.
        theme: Color theme ("light" or "dark").

    Returns:
        Plotly Figure.
    """
    if not intent_metrics:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return _apply_theme(fig, theme)

    # Sort by mean_ap descending
    sorted_metrics = sorted(
        intent_metrics,
        key=lambda x: x.get("mean_ap", 0) or 0,
        reverse=True
    )

    # Truncate intent names for display
    intents = [m.get("intent", "Unknown")[:40] for m in sorted_metrics]
    mean_aps = [m.get("mean_ap", 0) or 0 for m in sorted_metrics]
    n_samples = [m.get("n_samples", 0) for m in sorted_metrics]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=intents,
        x=mean_aps,
        orientation="h",
        marker_color="steelblue",
        text=[f"n={n}" for n in n_samples],
        textposition="outside",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Mean Average Precision",
        yaxis_title="Intent",
        xaxis_range=[0, 1],
        height=max(400, len(intents) * 25),
    )

    return _apply_theme(fig, theme)


def create_error_category_pie(
    category_counts: dict[str, int],
    title: str = "Errors by Category",
    theme: Theme = "light",
) -> go.Figure:
    """Create a pie chart showing error distribution by category.

    Args:
        category_counts: Dict mapping category name to count.
        title: Chart title.
        theme: Color theme ("light" or "dark").

    Returns:
        Plotly Figure.
    """
    if not category_counts or sum(category_counts.values()) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No errors to display", xref="paper", yref="paper", x=0.5, y=0.5)
        return _apply_theme(fig, theme)

    # Define colors for each category
    category_colors = {
        "api_error": "#FF6B6B",
        "rate_limit": "#FFA07A",
        "timeout": "#FFD93D",
        "parsing_error": "#6BCB77",
        "validation_error": "#4D96FF",
        "context_length": "#9B59B6",
        "content_filter": "#E74C3C",
        "authentication": "#C0392B",
        "network_error": "#3498DB",
        "unknown": "#95A5A6",
    }

    categories = list(category_counts.keys())
    counts = list(category_counts.values())
    colors = [category_colors.get(c, "#95A5A6") for c in categories]

    fig = go.Figure(data=[go.Pie(
        labels=categories,
        values=counts,
        marker_colors=colors,
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    )])

    fig.update_layout(
        title=title,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )

    return _apply_theme(fig, theme)


def create_error_type_bar(
    type_counts: dict[str, int],
    title: str = "Top Error Types",
    max_types: int = 10,
    theme: Theme = "light",
) -> go.Figure:
    """Create a horizontal bar chart of error types by frequency.

    Args:
        type_counts: Dict mapping error type to count.
        title: Chart title.
        max_types: Maximum number of types to show.
        theme: Color theme ("light" or "dark").

    Returns:
        Plotly Figure.
    """
    if not type_counts:
        fig = go.Figure()
        fig.add_annotation(text="No errors to display", xref="paper", yref="paper", x=0.5, y=0.5)
        return _apply_theme(fig, theme)

    # Sort by count and take top N
    sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:max_types]

    types = [t[0] for t in sorted_types]
    counts = [t[1] for t in sorted_types]

    # Reverse for horizontal bar chart (top item at top)
    types.reverse()
    counts.reverse()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=types,
        x=counts,
        orientation="h",
        marker_color="#FF6B6B",
        text=counts,
        textposition="outside",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Count",
        yaxis_title="Error Type",
        height=max(300, len(types) * 35),
    )

    return _apply_theme(fig, theme)


def create_error_severity_bar(
    severity_counts: dict[str, int],
    title: str = "Errors by Severity",
    theme: Theme = "light",
) -> go.Figure:
    """Create a bar chart showing error distribution by severity.

    Args:
        severity_counts: Dict mapping severity level to count.
        title: Chart title.
        theme: Color theme ("light" or "dark").

    Returns:
        Plotly Figure.
    """
    if not severity_counts or sum(severity_counts.values()) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No errors to display", xref="paper", yref="paper", x=0.5, y=0.5)
        return _apply_theme(fig, theme)

    # Define severity order and colors
    severity_order = ["low", "medium", "high", "critical"]
    severity_colors = {
        "low": "#6BCB77",
        "medium": "#FFD93D",
        "high": "#FF6B6B",
        "critical": "#C0392B",
    }

    # Order severities correctly
    ordered_severities = [s for s in severity_order if s in severity_counts]
    counts = [severity_counts[s] for s in ordered_severities]
    colors = [severity_colors.get(s, "#95A5A6") for s in ordered_severities]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=ordered_severities,
        y=counts,
        marker_color=colors,
        text=counts,
        textposition="outside",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Severity",
        yaxis_title="Count",
    )

    return _apply_theme(fig, theme)


def create_error_timeline(
    timeline_data: Sequence[tuple],
    title: str = "Error Timeline",
    theme: Theme = "light",
) -> go.Figure:
    """Create a timeline chart showing errors over time.

    Args:
        timeline_data: List of (datetime, count) tuples.
        title: Chart title.
        theme: Color theme ("light" or "dark").

    Returns:
        Plotly Figure.
    """
    if not timeline_data:
        fig = go.Figure()
        fig.add_annotation(text="No timeline data", xref="paper", yref="paper", x=0.5, y=0.5)
        return _apply_theme(fig, theme)

    dates = [t[0] for t in timeline_data]
    counts = [t[1] for t in timeline_data]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=counts,
        mode="lines+markers",
        marker=dict(size=8, color="#FF6B6B"),
        line=dict(width=2, color="#FF6B6B"),
        fill="tozeroy",
        fillcolor="rgba(255, 107, 107, 0.2)",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Error Count",
    )

    return _apply_theme(fig, theme)


def create_error_rate_gauge(
    error_rate: float,
    title: str = "Error Rate",
    theme: Theme = "light",
) -> go.Figure:
    """Create a gauge chart showing error rate.

    Args:
        error_rate: Error rate as a decimal (0-1).
        title: Chart title.
        theme: Color theme ("light" or "dark").

    Returns:
        Plotly Figure.
    """
    # Determine color based on error rate
    if error_rate < 0.05:
        color = "#6BCB77"  # Green - good
    elif error_rate < 0.15:
        color = "#FFD93D"  # Yellow - warning
    else:
        color = "#FF6B6B"  # Red - bad

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=error_rate * 100,
        number={"suffix": "%", "valueformat": ".1f"},
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 5], "color": "rgba(107, 203, 119, 0.3)"},
                {"range": [5, 15], "color": "rgba(255, 217, 61, 0.3)"},
                {"range": [15, 100], "color": "rgba(255, 107, 107, 0.3)"},
            ],
            "threshold": {
                "line": {"color": "white" if theme == "dark" else "black", "width": 2},
                "thickness": 0.75,
                "value": error_rate * 100,
            },
        },
    ))

    fig.update_layout(height=250)

    return _apply_theme(fig, theme)


def create_error_by_intent_bar(
    intent_errors: dict[str, dict],
    title: str = "Errors by Intent",
    theme: Theme = "light",
) -> go.Figure:
    """Create a stacked bar chart showing errors by intent and category.

    Args:
        intent_errors: Dict mapping intent to {category: count}.
        title: Chart title.
        theme: Color theme ("light" or "dark").

    Returns:
        Plotly Figure.
    """
    if not intent_errors:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return _apply_theme(fig, theme)

    # Get all categories across all intents
    all_categories = set()
    for cats in intent_errors.values():
        all_categories.update(cats.keys())

    category_colors = {
        "api_error": "#FF6B6B",
        "rate_limit": "#FFA07A",
        "timeout": "#FFD93D",
        "parsing_error": "#6BCB77",
        "validation_error": "#4D96FF",
        "context_length": "#9B59B6",
        "content_filter": "#E74C3C",
        "authentication": "#C0392B",
        "network_error": "#3498DB",
        "unknown": "#95A5A6",
    }

    intents = list(intent_errors.keys())

    fig = go.Figure()

    for category in sorted(all_categories):
        counts = [intent_errors[intent].get(category, 0) for intent in intents]
        fig.add_trace(go.Bar(
            name=category,
            x=intents,
            y=counts,
            marker_color=category_colors.get(category, "#95A5A6"),
        ))

    fig.update_layout(
        title=title,
        barmode="stack",
        xaxis_title="Intent",
        yaxis_title="Error Count",
        legend_title="Category",
    )

    return _apply_theme(fig, theme)


