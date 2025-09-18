"""Visualization generation for SQL query results using Plotly."""

from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


class ChartType(Enum):
    """Supported chart types for data visualization."""

    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"
    HEATMAP = "heatmap"
    TABLE = "table"


@dataclass
class VisualizationConfig:
    """Configuration for generating visualization DSL."""

    chart_type: ChartType
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    color_column: Optional[str] = None
    size_column: Optional[str] = None
    title: Optional[str] = None
    x_title: Optional[str] = None
    y_title: Optional[str] = None
    show_legend: bool = True
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass
class VisualizationDSL:
    """Plotly-friendly DSL for data visualization."""

    chart_type: str
    data_columns: List[str]
    config: Dict[str, Any]
    layout: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "chart_type": self.chart_type,
            "data_columns": self.data_columns,
            "config": self.config,
            "layout": self.layout,
        }


class VisualizationAnalyzer:
    """Analyzes query results and user questions to determine appropriate visualization."""

    @staticmethod
    def analyze_data_schema(data_csv: str) -> Dict[str, Any]:
        """Analyze CSV data to understand column types and characteristics."""
        import pandas as pd
        from io import StringIO

        try:
            df = pd.read_csv(StringIO(data_csv))
            schema_info = {
                "columns": list(df.columns),
                "column_types": {},
                "row_count": len(df),
                "numeric_columns": [],
                "categorical_columns": [],
                "datetime_columns": [],
            }

            for col in df.columns:
                dtype = str(df[col].dtype)
                schema_info["column_types"][col] = dtype

                # Classify column types
                if df[col].dtype in ["int64", "float64", "int32", "float32"]:
                    schema_info["numeric_columns"].append(col)
                elif df[col].dtype == "object":
                    # Check if it could be datetime
                    try:
                        pd.to_datetime(df[col].head(10))
                        schema_info["datetime_columns"].append(col)
                    except:
                        schema_info["categorical_columns"].append(col)

            # Calculate unique value counts for categorical columns
            schema_info["unique_counts"] = {}
            for col in schema_info["categorical_columns"]:
                schema_info["unique_counts"][col] = df[col].nunique()

            return schema_info
        except Exception as e:
            return {"error": f"Failed to analyze data schema: {str(e)}"}

    @staticmethod
    def recommend_chart_type(question: str, schema_info: Dict[str, Any]) -> ChartType:
        """Recommend chart type based on user question and data schema."""
        question_lower = question.lower()

        # Get data characteristics
        numeric_cols = schema_info.get("numeric_columns", [])
        categorical_cols = schema_info.get("categorical_columns", [])
        datetime_cols = schema_info.get("datetime_columns", [])
        row_count = schema_info.get("row_count", 0)

        # Question-based heuristics
        if any(keyword in question_lower for keyword in ["trend", "over time", "timeline", "time series"]):
            return ChartType.LINE
        elif any(keyword in question_lower for keyword in ["distribution", "frequency", "histogram"]):
            return ChartType.HISTOGRAM
        elif any(keyword in question_lower for keyword in ["correlation", "relationship", "scatter"]):
            return ChartType.SCATTER
        elif any(keyword in question_lower for keyword in ["proportion", "percentage", "share", "pie"]):
            return ChartType.PIE
        elif any(keyword in question_lower for keyword in ["compare", "comparison", "vs", "versus", "bar"]):
            return ChartType.BAR
        elif any(keyword in question_lower for keyword in ["summary", "range", "quartile", "box"]):
            return ChartType.BOX

        # Data-based heuristics
        if len(datetime_cols) > 0 and len(numeric_cols) > 0:
            return ChartType.LINE
        elif len(categorical_cols) == 1 and len(numeric_cols) == 1:
            unique_count = schema_info.get("unique_counts", {}).get(categorical_cols[0], 0)
            if unique_count <= 10:
                return ChartType.PIE if unique_count <= 6 else ChartType.BAR
            else:
                return ChartType.BAR
        elif len(numeric_cols) == 2:
            return ChartType.SCATTER
        elif len(numeric_cols) == 1 and len(categorical_cols) == 0:
            return ChartType.HISTOGRAM
        elif row_count <= 20:  # Changed from 50 to 20
            return ChartType.TABLE
        else:
            return ChartType.BAR

    @staticmethod
    def generate_visualization_dsl(
        question: str, data_csv: str, chart_type: Optional[ChartType] = None
    ) -> VisualizationDSL:
        """Generate visualization DSL based on question and data."""
        schema_info = VisualizationAnalyzer.analyze_data_schema(data_csv)

        if "error" in schema_info:
            # Return table view for error cases
            return VisualizationDSL(
                chart_type="table",
                data_columns=["error"],
                config={"error": schema_info["error"]},
                layout={"title": "Data Analysis Error"},
            )

        # Determine chart type
        if chart_type is None:
            chart_type = VisualizationAnalyzer.recommend_chart_type(question, schema_info)

        columns = schema_info["columns"]
        numeric_cols = schema_info["numeric_columns"]
        categorical_cols = schema_info["categorical_columns"]
        datetime_cols = schema_info["datetime_columns"]

        # Generate DSL based on chart type
        if chart_type == ChartType.LINE:
            x_col = datetime_cols[0] if datetime_cols else (categorical_cols[0] if categorical_cols else columns[0])
            y_col = numeric_cols[0] if numeric_cols else columns[-1]
            return VisualizationDSL(
                chart_type="line",
                data_columns=[x_col, y_col],
                config={"x": x_col, "y": y_col, "mode": "lines+markers"},
                layout={"title": f"Line Chart: {y_col} over {x_col}", "xaxis_title": x_col, "yaxis_title": y_col},
            )

        elif chart_type == ChartType.BAR:
            x_col = categorical_cols[0] if categorical_cols else columns[0]
            y_col = numeric_cols[0] if numeric_cols else columns[-1]
            return VisualizationDSL(
                chart_type="bar",
                data_columns=[x_col, y_col],
                config={"x": x_col, "y": y_col},
                layout={"title": f"Bar Chart: {y_col} by {x_col}", "xaxis_title": x_col, "yaxis_title": y_col},
            )

        elif chart_type == ChartType.PIE:
            label_col = categorical_cols[0] if categorical_cols else columns[0]
            value_col = numeric_cols[0] if numeric_cols else columns[-1]
            return VisualizationDSL(
                chart_type="pie",
                data_columns=[label_col, value_col],
                config={"labels": label_col, "values": value_col},
                layout={"title": f"Pie Chart: {value_col} by {label_col}"},
            )

        elif chart_type == ChartType.SCATTER:
            x_col = numeric_cols[0] if len(numeric_cols) > 0 else columns[0]
            y_col = numeric_cols[1] if len(numeric_cols) > 1 else columns[-1]
            return VisualizationDSL(
                chart_type="scatter",
                data_columns=[x_col, y_col],
                config={"x": x_col, "y": y_col, "mode": "markers"},
                layout={"title": f"Scatter Plot: {y_col} vs {x_col}", "xaxis_title": x_col, "yaxis_title": y_col},
            )

        elif chart_type == ChartType.HISTOGRAM:
            col = numeric_cols[0] if numeric_cols else columns[0]
            return VisualizationDSL(
                chart_type="histogram",
                data_columns=[col],
                config={"x": col, "nbins": 20},
                layout={"title": f"Histogram: Distribution of {col}", "xaxis_title": col, "yaxis_title": "Frequency"},
            )

        elif chart_type == ChartType.BOX:
            y_col = numeric_cols[0] if numeric_cols else columns[0]
            x_col = categorical_cols[0] if categorical_cols else None
            config = {"y": y_col}
            if x_col:
                config["x"] = x_col
            return VisualizationDSL(
                chart_type="box",
                data_columns=[col for col in [x_col, y_col] if col],
                config=config,
                layout={
                    "title": f"Box Plot: {y_col}" + (f" by {x_col}" if x_col else ""),
                    "xaxis_title": x_col if x_col else "",
                    "yaxis_title": y_col,
                },
            )

        else:  # TABLE or fallback
            return VisualizationDSL(
                chart_type="table", data_columns=columns, config={"columns": columns}, layout={"title": "Data Table"}
            )
