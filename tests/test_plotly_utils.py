"""Tests for plotly utilities in the UI."""

import pytest
import plotly.graph_objects as go
from sample_ui.plotly_utils import (
    create_plotly_chart,
    create_line_chart,
    create_bar_chart,
    create_pie_chart,
    create_scatter_chart,
    create_histogram_chart,
    create_box_chart,
    create_table_chart,
    create_empty_chart,
    visualization_dsl_to_gradio_plot,
)


@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing."""
    return """product,sales,region,month
Widget A,10000,North,Jan
Widget B,15000,South,Jan  
Widget C,8000,East,Jan
Widget A,12000,North,Feb
Widget B,18000,South,Feb
Widget C,9000,East,Feb"""


@pytest.fixture
def sample_line_dsl():
    """Sample DSL for line chart."""
    return {
        "chart_type": "line",
        "data_columns": ["month", "sales"],
        "config": {"x": "month", "y": "sales", "mode": "lines+markers"},
        "layout": {"title": "Sales Over Time", "xaxis_title": "Month", "yaxis_title": "Sales"},
    }


@pytest.fixture
def sample_bar_dsl():
    """Sample DSL for bar chart."""
    return {
        "chart_type": "bar",
        "data_columns": ["region", "sales"],
        "config": {"x": "region", "y": "sales"},
        "layout": {"title": "Sales by Region", "xaxis_title": "Region", "yaxis_title": "Sales"},
    }


@pytest.fixture
def sample_pie_dsl():
    """Sample DSL for pie chart."""
    return {
        "chart_type": "pie",
        "data_columns": ["product", "sales"],
        "config": {"labels": "product", "values": "sales"},
        "layout": {"title": "Sales Distribution by Product"},
    }


class TestPlotlyChartCreation:
    """Tests for individual chart creation functions."""

    def test_create_line_chart_success(self, sample_csv_data, sample_line_dsl):
        """Test successful line chart creation."""
        fig = create_plotly_chart(sample_csv_data, sample_line_dsl)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.title.text == "Sales Over Time"

    def test_create_line_chart_with_color(self, sample_csv_data):
        """Test line chart creation with color parameter for multiple series."""
        multi_series_dsl = {
            "chart_type": "line",
            "data_columns": ["month", "sales", "product"],
            "config": {"x": "month", "y": "sales", "color": "product"},
            "layout": {"title": "Sales Over Time by Product", "xaxis_title": "Month", "yaxis_title": "Sales"},
        }

        fig = create_plotly_chart(sample_csv_data, multi_series_dsl)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0  # Should have multiple traces for different products
        assert fig.layout.title.text == "Sales Over Time by Product"

    def test_create_line_chart_with_multiple_y_columns(self):
        """Test line chart creation with multiple y columns."""
        multi_metric_data = """date,revenue,profit,users
2023-01-01,50000,15000,1000
2023-02-01,55000,18000,1100
2023-03-01,60000,20000,1200"""

        multi_y_dsl = {
            "chart_type": "line",
            "data_columns": ["date", "revenue", "profit"],
            "config": {"x": "date", "y": ["revenue", "profit"]},
            "layout": {"title": "Multiple Metrics Over Time", "xaxis_title": "Date", "yaxis_title": "Value"},
        }

        fig = create_plotly_chart(multi_metric_data, multi_y_dsl)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0  # Should have multiple traces for different metrics
        assert fig.layout.title.text == "Multiple Metrics Over Time"

    def test_create_bar_chart_success(self, sample_csv_data, sample_bar_dsl):
        """Test successful bar chart creation."""
        fig = create_plotly_chart(sample_csv_data, sample_bar_dsl)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.title.text == "Sales by Region"

    def test_create_pie_chart_success(self, sample_csv_data, sample_pie_dsl):
        """Test successful pie chart creation."""
        fig = create_plotly_chart(sample_csv_data, sample_pie_dsl)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.title.text == "Sales Distribution by Product"

    def test_create_scatter_chart(self, sample_csv_data):
        """Test scatter chart creation."""
        scatter_dsl = {
            "chart_type": "scatter",
            "data_columns": ["sales", "region"],
            "config": {"x": "sales", "y": "region", "mode": "markers"},
            "layout": {"title": "Sales Scatter Plot"},
        }

        fig = create_plotly_chart(sample_csv_data, scatter_dsl)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_create_histogram_chart(self, sample_csv_data):
        """Test histogram chart creation."""
        histogram_dsl = {
            "chart_type": "histogram",
            "data_columns": ["sales"],
            "config": {"x": "sales", "nbins": 10},
            "layout": {"title": "Sales Distribution"},
        }

        fig = create_plotly_chart(sample_csv_data, histogram_dsl)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_create_box_chart(self, sample_csv_data):
        """Test box chart creation."""
        box_dsl = {
            "chart_type": "box",
            "data_columns": ["sales", "region"],
            "config": {"y": "sales", "x": "region"},
            "layout": {"title": "Sales Distribution by Region"},
        }

        fig = create_plotly_chart(sample_csv_data, box_dsl)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_create_table_chart(self, sample_csv_data):
        """Test table chart creation."""
        table_dsl = {
            "chart_type": "table",
            "data_columns": ["product", "sales", "region", "month"],
            "config": {"columns": ["product", "sales", "region", "month"]},
            "layout": {"title": "Data Table"},
        }

        fig = create_plotly_chart(sample_csv_data, table_dsl)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.data[0].type == "table"


class TestErrorHandling:
    """Tests for error handling in chart creation."""

    def test_empty_data(self):
        """Test handling of empty data."""
        fig = create_plotly_chart("", {})

        assert isinstance(fig, go.Figure)
        # Should create an empty chart with error message

    def test_invalid_csv_data(self, sample_bar_dsl):
        """Test handling of invalid CSV data."""
        invalid_csv = "invalid,csv\ndata"

        fig = create_plotly_chart(invalid_csv, sample_bar_dsl)

        assert isinstance(fig, go.Figure)
        # Should create an empty chart with error message

    def test_missing_columns(self, sample_csv_data):
        """Test handling of missing columns in DSL."""
        invalid_dsl = {
            "chart_type": "line",
            "data_columns": ["nonexistent_col"],
            "config": {"x": "nonexistent_col", "y": "another_missing_col"},
            "layout": {"title": "Invalid Chart"},
        }

        fig = create_plotly_chart(sample_csv_data, invalid_dsl)

        assert isinstance(fig, go.Figure)
        # Should create an empty chart with error message

    def test_unsupported_chart_type(self, sample_csv_data):
        """Test handling of unsupported chart types."""
        invalid_dsl = {"chart_type": "unsupported_type", "data_columns": ["sales"], "config": {}, "layout": {}}

        fig = create_plotly_chart(sample_csv_data, invalid_dsl)

        assert isinstance(fig, go.Figure)
        # Should create an empty chart with error message

    def test_visualization_dsl_error(self):
        """Test handling of DSL with error field."""
        error_dsl = {"error": "Failed to generate visualization"}

        fig = create_plotly_chart("some,data\n1,2", error_dsl)

        assert isinstance(fig, go.Figure)
        # Should create an empty chart with error message


class TestVisualizationDslToGradioPlot:
    """Tests for the main interface function."""

    def test_successful_conversion(self, sample_csv_data, sample_line_dsl):
        """Test successful DSL to Gradio plot conversion."""
        fig, description = visualization_dsl_to_gradio_plot(sample_csv_data, sample_line_dsl)

        assert isinstance(fig, go.Figure)
        assert isinstance(description, str)
        assert "line" in description.lower()
        assert "Sales Over Time" in description

    def test_empty_dsl(self, sample_csv_data):
        """Test conversion with empty DSL."""
        fig, description = visualization_dsl_to_gradio_plot(sample_csv_data, {})

        assert isinstance(fig, go.Figure)
        assert isinstance(description, str)
        assert "table" in description.lower()

    def test_no_data(self, sample_line_dsl):
        """Test conversion with no data."""
        fig, description = visualization_dsl_to_gradio_plot("", sample_line_dsl)

        assert isinstance(fig, go.Figure)
        assert isinstance(description, str)


class TestCreateEmptyChart:
    """Tests for empty chart creation."""

    def test_create_empty_chart(self):
        """Test empty chart creation with message."""
        message = "Test error message"
        fig = create_empty_chart(message)

        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Chart Generation Issue"
        # Check if annotation contains the message
        assert len(fig.layout.annotations) > 0
        assert fig.layout.annotations[0].text == message


@pytest.fixture
def sample_time_series_data():
    """Sample time series data for testing."""
    return """date,revenue,users
2023-01-01,50000,1000
2023-02-01,55000,1100
2023-03-01,60000,1200
2023-04-01,52000,1050
2023-05-01,58000,1150"""


class TestIntegrationScenarios:
    """Integration tests for complete visualization scenarios."""

    def test_sales_dashboard_scenario(self, sample_csv_data):
        """Test a complete sales dashboard scenario."""
        # Test multiple chart types with the same data
        chart_configs = [
            {"chart_type": "bar", "config": {"x": "product", "y": "sales"}, "layout": {"title": "Sales by Product"}},
            {
                "chart_type": "pie",
                "config": {"labels": "region", "values": "sales"},
                "layout": {"title": "Sales by Region"},
            },
        ]

        for config in chart_configs:
            config["data_columns"] = list(config["config"].values())
            fig = create_plotly_chart(sample_csv_data, config)
            assert isinstance(fig, go.Figure)
            assert len(fig.data) > 0

    def test_time_series_scenario(self, sample_time_series_data):
        """Test time series visualization scenario."""
        line_dsl = {
            "chart_type": "line",
            "data_columns": ["date", "revenue"],
            "config": {"x": "date", "y": "revenue", "mode": "lines+markers"},
            "layout": {"title": "Revenue Trend Over Time", "xaxis_title": "Date", "yaxis_title": "Revenue"},
        }

        fig, description = visualization_dsl_to_gradio_plot(sample_time_series_data, line_dsl)

        assert isinstance(fig, go.Figure)
        assert "line" in description.lower()
        assert "Revenue Trend Over Time" in description

    def test_multiple_metrics_scenario(self, sample_time_series_data):
        """Test scenario with multiple metrics."""
        scatter_dsl = {
            "chart_type": "scatter",
            "data_columns": ["revenue", "users"],
            "config": {"x": "revenue", "y": "users", "mode": "markers"},
            "layout": {"title": "Revenue vs Users Correlation", "xaxis_title": "Revenue", "yaxis_title": "Users"},
        }

        fig = create_plotly_chart(sample_time_series_data, scatter_dsl)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.title.text == "Revenue vs Users Correlation"
