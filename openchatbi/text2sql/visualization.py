"""Visualization generation for SQL query results using Plotly."""

from dataclasses import dataclass
from enum import Enum
from io import StringIO
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from openchatbi.prompts.system_prompt import VISUALIZATION_PROMPT_TEMPLATE


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


class VisualizationService:
    """Service class to handle visualization generation with configurable analysis method."""

    # Chart type mapping for LLM responses
    CHART_TYPE_MAPPING = {
        "line": ChartType.LINE,
        "bar": ChartType.BAR,
        "pie": ChartType.PIE,
        "scatter": ChartType.SCATTER,
        "histogram": ChartType.HISTOGRAM,
        "box": ChartType.BOX,
        "heatmap": ChartType.HEATMAP,
        "table": ChartType.TABLE,
    }

    def __init__(self, llm: Optional[BaseChatModel] = None):
        """Initialize visualization service.

        Args:
            llm: BaseChatModel LLM instance, will skip using LLM if None
        """
        self.llm = llm

    def _get_chart_type_by_rule(self, question: str, schema_info: Dict[str, Any]) -> ChartType:
        """Recommend chart type based on user question and data schema using rules."""
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

    def generate_visualization_dsl(
        self, question: str, schema_info: Dict[str, Any], chart_type: Optional[ChartType] = None
    ) -> VisualizationDSL:
        """Generate visualization DSL based on question and schema info."""
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
            chart_type = self._get_chart_type_by_rule(question, schema_info)

        columns = schema_info["columns"]
        numeric_cols = schema_info["numeric_columns"]
        categorical_cols = schema_info["categorical_columns"]
        datetime_cols = schema_info["datetime_columns"]

        # Generate DSL based on chart type
        if chart_type == ChartType.LINE:
            x_col = datetime_cols[0] if datetime_cols else (categorical_cols[0] if categorical_cols else columns[0])
            # For line charts, include all numeric columns for multiple metrics
            y_cols = numeric_cols if numeric_cols else [columns[-1]]
            data_columns = [x_col] + y_cols

            # Support multiple y-axis columns
            config = {"x": x_col, "mode": "lines+markers"}
            if len(y_cols) == 1:
                config["y"] = y_cols[0]
                title = f"Line Chart: {y_cols[0]} over {x_col}"
            else:
                config["y"] = y_cols  # Multiple metrics
                title = f"Line Chart: {', '.join(y_cols)} over {x_col}"

            return VisualizationDSL(
                chart_type="line",
                data_columns=data_columns,
                config=config,
                layout={"title": title, "xaxis_title": x_col, "yaxis_title": "Value"},
            )

        elif chart_type == ChartType.BAR:
            x_col = categorical_cols[0] if categorical_cols else columns[0]
            # For bar charts, include all numeric columns for multiple metrics
            y_cols = numeric_cols if numeric_cols else [columns[-1]]
            data_columns = [x_col] + y_cols

            config = {"x": x_col}
            if len(y_cols) == 1:
                config["y"] = y_cols[0]
                title = f"Bar Chart: {y_cols[0]} by {x_col}"
            else:
                config["y"] = y_cols  # Multiple metrics
                title = f"Bar Chart: {', '.join(y_cols)} by {x_col}"

            return VisualizationDSL(
                chart_type="bar",
                data_columns=data_columns,
                config=config,
                layout={"title": title, "xaxis_title": x_col, "yaxis_title": "Value"},
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

    def _llm_recommend_chart_type(self, question: str, schema_info: Dict[str, Any], data_sample: str) -> ChartType:
        """Use LLM to recommend chart type based on question and data analysis.

        Args:
            question: User's question or intent
            schema_info: Data schema information
            data_sample: Sample of the data

        Returns:
            ChartType: Recommended chart type
        """
        try:
            prompt = (
                VISUALIZATION_PROMPT_TEMPLATE.replace("[question]", question)
                .replace("[columns]", str(schema_info.get("columns", [])))
                .replace("[numeric_columns]", str(schema_info.get("numeric_columns", [])))
                .replace("[categorical_columns]", str(schema_info.get("categorical_columns", [])))
                .replace("[datetime_columns]", str(schema_info.get("datetime_columns", [])))
                .replace("[row_count]", str(schema_info.get("row_count", 0)))
                .replace("[data_sample]", data_sample)
            )

            # Call LLM with the formatted prompt
            response = self.llm.invoke([HumanMessage(content=prompt)])
            chart_type_str = response.content.strip().lower()
            return self.CHART_TYPE_MAPPING.get(chart_type_str, ChartType.TABLE)

        except Exception:
            # Fallback to rule-based recommendation on other LLM errors
            return self._get_chart_type_by_rule(question, schema_info)

    def generate_visualization(
        self, question: str, schema_info: Dict[str, Any], csv_data: str, chart_type: Optional[ChartType] = None
    ) -> Optional[VisualizationDSL]:
        """Generate visualization using the configured analysis method.

        Args:
            question: User's question or intent
            schema_info: Pre-analyzed schema information
            csv_data: CSV data string for LLM analysis if needed
            chart_type: Optional specific chart type to use

        Returns:
            VisualizationDSL or None: Generated visualization configuration, or None if skipped
        """
        # Use existing DSL generation if chart type is already specified
        if chart_type is not None:
            return self.generate_visualization_dsl(question, schema_info, chart_type)

        # Determine chart type based on configured method
        if self.llm:
            if "error" in schema_info:
                return VisualizationDSL(
                    chart_type="table",
                    data_columns=["error"],
                    config={"error": schema_info["error"]},
                    layout={"title": "Data Analysis Error"},
                )

            # Prepare data sample for LLM analysis
            try:
                df = pd.read_csv(StringIO(csv_data))
                data_sample = df.head(3).to_string() if len(df) > 0 else "No data available"
            except Exception:
                data_sample = "Unable to parse data"

            chart_type = self._llm_recommend_chart_type(question, schema_info, data_sample)

        # Generate DSL using determined or recommended chart type
        return self.generate_visualization_dsl(question, schema_info, chart_type)
