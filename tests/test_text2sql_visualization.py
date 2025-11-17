"""Tests for text2sql visualization functionality."""

import pytest

from openchatbi.text2sql.visualization import ChartType, VisualizationConfig, VisualizationDSL, VisualizationService


class TestVisualizationService:
    """Tests for the VisualizationService class."""

    def test_generate_visualization_dsl_basic(self):
        """Test basic DSL generation with schema info."""
        schema_info = {
            "columns": ["name", "age", "salary", "department"],
            "row_count": 4,
            "numeric_columns": ["age", "salary"],
            "categorical_columns": ["name", "department"],
            "datetime_columns": [],
            "unique_counts": {"name": 4, "department": 2},
        }

        service = VisualizationService()
        question = "Compare salary by department"
        dsl = service.generate_visualization_dsl(question, schema_info)

        assert dsl.chart_type == "bar"
        # Should use first categorical column which is "name"
        assert "name" in dsl.data_columns
        assert "age" in dsl.data_columns and "salary" in dsl.data_columns  # Both numeric columns should be included

    def test_get_chart_type_by_rule_with_datetime(self):
        """Test chart type recommendation with datetime columns."""
        schema_info = {
            "columns": ["date", "sales", "region"],
            "numeric_columns": ["sales"],
            "categorical_columns": ["region"],
            "datetime_columns": ["date"],
            "row_count": 3,
        }

        service = VisualizationService()
        question = "Show sales trend over time"
        chart_type = service._get_chart_type_by_rule(question, schema_info)

        assert chart_type == ChartType.LINE

    def test_generate_visualization_dsl_error_handling(self):
        """Test DSL generation with error in schema info."""
        schema_info = {"error": "Failed to analyze data schema"}

        service = VisualizationService()
        question = "Show data"
        dsl = service.generate_visualization_dsl(question, schema_info)

        assert dsl.chart_type == "table"
        assert "error" in dsl.config

    def test_get_chart_type_by_rule_line_chart(self):
        """Test recommendation for line chart based on question keywords."""
        question = "Show me the sales trend over time"
        schema = {
            "numeric_columns": ["sales"],
            "categorical_columns": ["region"],
            "datetime_columns": ["date"],
            "row_count": 10,
        }

        service = VisualizationService()
        chart_type = service._get_chart_type_by_rule(question, schema)

        assert chart_type == ChartType.LINE

    def test_get_chart_type_by_rule_pie_chart(self):
        """Test recommendation for pie chart based on question keywords."""
        question = "What is the percentage breakdown by department?"
        schema = {
            "numeric_columns": ["count"],
            "categorical_columns": ["department"],
            "datetime_columns": [],
            "row_count": 5,
            "unique_counts": {"department": 4},
        }

        service = VisualizationService()
        chart_type = service._get_chart_type_by_rule(question, schema)

        assert chart_type == ChartType.PIE

    def test_get_chart_type_by_rule_bar_chart(self):
        """Test recommendation for bar chart based on question keywords."""
        question = "Compare sales by region"
        schema = {
            "numeric_columns": ["sales"],
            "categorical_columns": ["region"],
            "datetime_columns": [],
            "row_count": 10,
            "unique_counts": {"region": 4},
        }

        service = VisualizationService()
        chart_type = service._get_chart_type_by_rule(question, schema)

        assert chart_type == ChartType.BAR

    def test_get_chart_type_by_rule_scatter_plot(self):
        """Test recommendation for scatter plot based on data characteristics."""
        question = "Show relationship between age and salary"
        schema = {
            "numeric_columns": ["age", "salary"],
            "categorical_columns": ["name"],
            "datetime_columns": [],
            "row_count": 10,
        }

        service = VisualizationService()
        chart_type = service._get_chart_type_by_rule(question, schema)

        assert chart_type == ChartType.SCATTER

    def test_get_chart_type_by_rule_histogram(self):
        """Test recommendation for histogram based on keywords."""
        question = "What is the distribution of ages?"
        schema = {"numeric_columns": ["age"], "categorical_columns": [], "datetime_columns": [], "row_count": 100}

        service = VisualizationService()
        chart_type = service._get_chart_type_by_rule(question, schema)

        assert chart_type == ChartType.HISTOGRAM

    def test_get_chart_type_by_rule_data_based_priority(self):
        """Test that data characteristics take priority over row count."""
        question = "Show all records"
        schema = {
            "numeric_columns": ["value"],
            "categorical_columns": ["category"],
            "datetime_columns": [],
            "row_count": 15,
            "unique_counts": {"category": 5},  # Small number of categories
        }

        service = VisualizationService()
        chart_type = service._get_chart_type_by_rule(question, schema)

        # Should choose PIE because of categorical + numeric columns, not TABLE due to row count
        assert chart_type == ChartType.PIE

    def test_generate_visualization_dsl_line_chart(self):
        """Test DSL generation for line chart."""
        question = "Show sales trend over time"
        schema_info = {
            "columns": ["date", "sales"],
            "numeric_columns": ["sales"],
            "categorical_columns": [],
            "datetime_columns": ["date"],
            "row_count": 3,
        }

        service = VisualizationService()
        dsl = service.generate_visualization_dsl(question, schema_info)

        assert dsl.chart_type == "line"
        assert "date" in dsl.data_columns
        assert "sales" in dsl.data_columns
        assert dsl.config["x"] == "date"
        assert dsl.config["y"] == "sales"

    def test_generate_visualization_dsl_bar_chart(self):
        """Test DSL generation for bar chart."""
        question = "Compare sales by region"
        schema_info = {
            "columns": ["region", "sales"],
            "numeric_columns": ["sales"],
            "categorical_columns": ["region"],
            "datetime_columns": [],
            "row_count": 4,
            "unique_counts": {"region": 4},
        }

        service = VisualizationService()
        dsl = service.generate_visualization_dsl(question, schema_info)

        assert dsl.chart_type == "bar"
        assert "region" in dsl.data_columns
        assert "sales" in dsl.data_columns
        assert dsl.config["x"] == "region"
        assert dsl.config["y"] == "sales"

    def test_generate_visualization_dsl_pie_chart(self):
        """Test DSL generation for pie chart."""
        question = "Show percentage breakdown by department"
        schema_info = {
            "columns": ["department", "count"],
            "numeric_columns": ["count"],
            "categorical_columns": ["department"],
            "datetime_columns": [],
            "row_count": 4,
            "unique_counts": {"department": 4},
        }

        service = VisualizationService()
        dsl = service.generate_visualization_dsl(question, schema_info, ChartType.PIE)

        assert dsl.chart_type == "pie"
        assert dsl.config["labels"] == "department"
        assert dsl.config["values"] == "count"

    def test_generate_visualization_dsl_empty_data(self):
        """Test DSL generation with empty data."""
        question = "Show data"
        schema_info = {
            "columns": [],
            "numeric_columns": [],
            "categorical_columns": [],
            "datetime_columns": [],
            "row_count": 0,
        }

        service = VisualizationService()
        dsl = service.generate_visualization_dsl(question, schema_info)

        assert dsl.chart_type == "table"
        assert "columns" in dsl.config

    def test_visualization_config_dataclass(self):
        """Test VisualizationConfig dataclass."""
        config = VisualizationConfig(
            chart_type=ChartType.BAR, x_column="category", y_column="value", title="Test Chart"
        )

        assert config.chart_type == ChartType.BAR
        assert config.x_column == "category"
        assert config.y_column == "value"
        assert config.title == "Test Chart"
        assert config.show_legend is True  # default value

    def test_visualization_dsl_to_dict(self):
        """Test VisualizationDSL to_dict method."""
        dsl = VisualizationDSL(
            chart_type="bar",
            data_columns=["x", "y"],
            config={"x": "category", "y": "value"},
            layout={"title": "Test Chart"},
        )

        result = dsl.to_dict()

        assert result["chart_type"] == "bar"
        assert result["data_columns"] == ["x", "y"]
        assert result["config"]["x"] == "category"
        assert result["layout"]["title"] == "Test Chart"


class TestChartType:
    """Tests for ChartType enum."""

    def test_chart_type_values(self):
        """Test ChartType enum values."""
        assert ChartType.LINE.value == "line"
        assert ChartType.BAR.value == "bar"
        assert ChartType.PIE.value == "pie"
        assert ChartType.SCATTER.value == "scatter"
        assert ChartType.HISTOGRAM.value == "histogram"
        assert ChartType.BOX.value == "box"
        assert ChartType.HEATMAP.value == "heatmap"
        assert ChartType.TABLE.value == "table"


@pytest.fixture
def sample_csv_data():
    """Fixture providing sample CSV data for testing."""
    return """product,sales,region,quarter
Widget A,10000,North,Q1
Widget B,15000,South,Q1
Widget C,8000,East,Q1
Widget A,12000,North,Q2
Widget B,18000,South,Q2
Widget C,9000,East,Q2"""


@pytest.fixture
def sample_time_series_data():
    """Fixture providing sample time series data for testing."""
    return """date,revenue,users
2023-01-01,50000,1000
2023-02-01,55000,1100
2023-03-01,60000,1200
2023-04-01,52000,1050
2023-05-01,58000,1150"""


class TestVisualizationIntegration:
    """Integration tests for visualization functionality."""

    def test_complete_workflow_line_chart(self, sample_time_series_data):
        """Test complete workflow for generating line chart."""
        question = "Show revenue trend over time"

        # Mock schema info for time series data
        schema_info = {
            "columns": ["date", "revenue", "users"],
            "numeric_columns": ["revenue", "users"],
            "categorical_columns": [],
            "datetime_columns": ["date"],
            "row_count": 5,
        }

        service = VisualizationService()
        # Recommend chart type
        chart_type = service._get_chart_type_by_rule(question, schema_info)

        # Generate DSL
        dsl = service.generate_visualization_dsl(question, schema_info, chart_type)

        assert chart_type == ChartType.LINE
        assert dsl.chart_type == "line"
        assert "date" in dsl.data_columns
        assert "revenue" in dsl.data_columns

    def test_complete_workflow_bar_chart(self, sample_csv_data):
        """Test complete workflow for generating bar chart."""
        question = "Compare sales by product"

        # Mock schema info for sample CSV data
        schema_info = {
            "columns": ["product", "sales", "region", "quarter"],
            "numeric_columns": ["sales"],
            "categorical_columns": ["product", "region", "quarter"],
            "datetime_columns": [],
            "row_count": 6,
            "unique_counts": {"product": 3, "region": 3, "quarter": 2},
        }

        service = VisualizationService()
        # Generate DSL directly (will analyze schema internally)
        dsl = service.generate_visualization_dsl(question, schema_info)

        assert dsl.chart_type in ["bar", "line"]  # Could be either based on heuristics
        assert len(dsl.data_columns) >= 2
        assert dsl.layout.get("title") is not None
