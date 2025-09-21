"""Plotly utilities for generating charts from visualization DSL."""

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
from typing import Dict, Any, Tuple


def create_plotly_chart(data_csv: str, visualization_dsl: Dict[str, Any]) -> go.Figure:
    """Create a plotly chart from CSV data and visualization DSL.

    Args:
        data_csv: CSV string containing the data
        visualization_dsl: Dictionary containing chart configuration

    Returns:
        Plotly Figure object
    """
    if not data_csv or not visualization_dsl:
        return create_empty_chart("No data available")

    if "error" in visualization_dsl:
        return create_empty_chart(f"Visualization error: {visualization_dsl['error']}")

    try:
        # Parse CSV data
        df = pd.read_csv(StringIO(data_csv))

        if df.empty:
            return create_empty_chart("No data to visualize")

        chart_type = visualization_dsl.get("chart_type", "table")
        config = visualization_dsl.get("config", {})
        layout = visualization_dsl.get("layout", {})

        # Create chart based on type
        if chart_type == "line":
            return create_line_chart(df, config, layout)
        elif chart_type == "bar":
            return create_bar_chart(df, config, layout)
        elif chart_type == "pie":
            return create_pie_chart(df, config, layout)
        elif chart_type == "scatter":
            return create_scatter_chart(df, config, layout)
        elif chart_type == "histogram":
            return create_histogram_chart(df, config, layout)
        elif chart_type == "box":
            return create_box_chart(df, config, layout)
        elif chart_type == "table":
            return create_table_chart(df, config, layout)
        else:
            return create_empty_chart(f"Unsupported chart type: {chart_type}")

    except Exception as e:
        return create_empty_chart(f"Chart generation error: {str(e)}")


def create_line_chart(df: pd.DataFrame, config: Dict[str, Any], layout: Dict[str, Any]) -> go.Figure:
    """Create a line chart."""
    x_col = config.get("x")
    y_col = config.get("y")
    color_col = config.get("color")

    if not x_col or x_col not in df.columns:
        return create_empty_chart("Missing required x column for line chart")

    # Handle multiple y columns case
    if isinstance(y_col, list):
        # Multiple metrics - need to melt the data
        if not all(col in df.columns for col in y_col):
            return create_empty_chart("Some y columns missing from data")

        # Melt the dataframe to long format for multiple series
        melted_df = df.melt(id_vars=[x_col], value_vars=y_col, var_name="metric", value_name="value")
        fig = px.line(melted_df, x=x_col, y="value", color="metric")

    else:
        # Single y column
        if not y_col or y_col not in df.columns:
            return create_empty_chart("Missing required y column for line chart")

        # Check if color column exists and is valid
        if color_col and color_col in df.columns:
            fig = px.line(df, x=x_col, y=y_col, color=color_col)
        else:
            fig = px.line(df, x=x_col, y=y_col)

    fig.update_layout(**layout)
    return fig


def create_bar_chart(df: pd.DataFrame, config: Dict[str, Any], layout: Dict[str, Any]) -> go.Figure:
    """Create a bar chart."""
    x_col = config.get("x")
    y_col = config.get("y")

    if not x_col or x_col not in df.columns:
        return create_empty_chart("Missing required x column for bar chart")

    # Handle multiple y columns case
    if isinstance(y_col, list):
        # Multiple metrics - need to melt the data
        if not all(col in df.columns for col in y_col):
            return create_empty_chart("Some y columns missing from data")

        # Melt the dataframe to long format for multiple series
        melted_df = df.melt(id_vars=[x_col], value_vars=y_col, var_name="metric", value_name="value")
        fig = px.bar(melted_df, x=x_col, y="value", color="metric")

    else:
        # Single y column
        if not y_col or y_col not in df.columns:
            return create_empty_chart("Missing required y column for bar chart")

        fig = px.bar(df, x=x_col, y=y_col)

    fig.update_layout(**layout)
    return fig


def create_pie_chart(df: pd.DataFrame, config: Dict[str, Any], layout: Dict[str, Any]) -> go.Figure:
    """Create a pie chart."""
    labels_col = config.get("labels")
    values_col = config.get("values")

    if not labels_col or not values_col or labels_col not in df.columns or values_col not in df.columns:
        return create_empty_chart("Missing required columns for pie chart")

    fig = px.pie(df, names=labels_col, values=values_col)
    fig.update_layout(**layout)
    return fig


def create_scatter_chart(df: pd.DataFrame, config: Dict[str, Any], layout: Dict[str, Any]) -> go.Figure:
    """Create a scatter plot."""
    x_col = config.get("x")
    y_col = config.get("y")

    if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
        return create_empty_chart("Missing required columns for scatter plot")

    fig = px.scatter(df, x=x_col, y=y_col)
    fig.update_layout(**layout)
    return fig


def create_histogram_chart(df: pd.DataFrame, config: Dict[str, Any], layout: Dict[str, Any]) -> go.Figure:
    """Create a histogram."""
    x_col = config.get("x")
    nbins = config.get("nbins", 20)

    if not x_col or x_col not in df.columns:
        return create_empty_chart("Missing required column for histogram")

    fig = px.histogram(df, x=x_col, nbins=nbins)
    fig.update_layout(**layout)
    return fig


def create_box_chart(df: pd.DataFrame, config: Dict[str, Any], layout: Dict[str, Any]) -> go.Figure:
    """Create a box plot."""
    y_col = config.get("y")
    x_col = config.get("x")

    if not y_col or y_col not in df.columns:
        return create_empty_chart("Missing required column for box plot")

    if x_col and x_col in df.columns:
        fig = px.box(df, x=x_col, y=y_col)
    else:
        fig = px.box(df, y=y_col)

    fig.update_layout(**layout)
    return fig


def create_table_chart(df: pd.DataFrame, config: Dict[str, Any], layout: Dict[str, Any]) -> go.Figure:
    """Create a table display."""
    columns = config.get("columns", list(df.columns))

    # Limit to first 100 rows for display
    display_df = df.head(100)

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=columns, fill_color="lightblue", align="left"),
                cells=dict(
                    values=[display_df[col] for col in columns if col in display_df.columns],
                    fill_color="white",
                    align="left",
                ),
            )
        ]
    )

    fig.update_layout(**layout)
    return fig


def create_empty_chart(message: str) -> go.Figure:
    """Create an empty chart with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        xanchor="center",
        yanchor="middle",
        showarrow=False,
        font=dict(size=16),
    )
    fig.update_layout(
        title="Chart Generation Issue",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
    )
    return fig


def visualization_dsl_to_gradio_plot(data_csv: str, visualization_dsl: Dict[str, Any]) -> Tuple[go.Figure, str]:
    """Convert visualization DSL to Gradio-compatible plotly figure.

    Args:
        data_csv: CSV string containing the data
        visualization_dsl: Dictionary containing chart configuration

    Returns:
        Tuple of (plotly figure, description string)
    """
    fig = create_plotly_chart(data_csv, visualization_dsl)

    if visualization_dsl:
        chart_type = visualization_dsl.get("chart_type", "unknown")
        layout_title = visualization_dsl.get("layout", {}).get("title", f"{chart_type.title()} Chart")
        description = f"Generated {chart_type} visualization: {layout_title}"
    else:
        description = "Data table view"

    return fig, description


def create_inline_chart_markdown(data_csv: str, visualization_dsl: Dict[str, Any]) -> str:
    """Create a simplified markdown representation of the chart for inline display.

    This creates a text-based summary with a clickable link to show the interactive chart.
    """
    if not data_csv or not visualization_dsl:
        return "📊 *No visualization data available*"

    if "error" in visualization_dsl:
        return f"⚠️ *Visualization error: {visualization_dsl['error']}*"

    try:
        import pandas as pd
        from io import StringIO

        df = pd.read_csv(StringIO(data_csv))
        chart_type = visualization_dsl.get("chart_type", "table")
        layout = visualization_dsl.get("layout", {})
        title = layout.get("title", f"{chart_type.title()} Chart")

        # Create a text summary with key data points and view instruction
        summary_lines = [
            f"📊 **{title}**",
            "",
            f"*Chart Type: {chart_type.title()}* | *Data Points: {len(df)} rows, {len(df.columns)} columns*",
            "",
            "✨ **Interactive chart will appear automatically in the chart panel →**",
            "",
        ]

        # Add a small data sample
        if len(df) > 0:
            summary_lines.append("**Sample Data:**")
            summary_lines.append("```")
            # Show first few rows in a clean format
            sample_df = df.head(3)
            summary_lines.append(sample_df.to_string(index=False))
            if len(df) > 3:
                summary_lines.append(f"... and {len(df) - 3} more rows")
            summary_lines.append("```\n")

        return "\n".join(summary_lines)

    except Exception as e:
        return f"⚠️ *Chart generation error: {str(e)}*"
