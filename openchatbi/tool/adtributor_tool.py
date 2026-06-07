import logging
from typing import Any

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from openchatbi.analysis.adtributor import adtributor
from openchatbi.analysis.models import AdtributorOutput

logger = logging.getLogger(__name__)


class DrilldownRow(BaseModel):
    """A single row of data for the adtributor drilldown analysis."""

    dimension_name: str = Field(description="Name of the dimension (e.g., 'device', 'province')")
    element_value: str | int | float = Field(description="Value of the dimension element (e.g., 'ios', 1)")

    # 绝对指标
    predict: float | None = Field(
        description="Baseline/expected value for absolute metrics. Pass null if using derived metrics."
    )
    real: float | None = Field(
        description="Actual observed value for absolute metrics. Pass null if using derived metrics."
    )

    # 派生指标 (比率)
    predict_numerator: float | None = Field(
        description="Baseline numerator for derived/ratio metrics. Pass null if using absolute metrics."
    )
    predict_denominator: float | None = Field(
        description="Baseline denominator for derived metrics. Pass null if using absolute metrics."
    )
    real_numerator: float | None = Field(
        description="Actual numerator for derived metrics. Pass null if using absolute metrics."
    )
    real_denominator: float | None = Field(
        description="Actual denominator for derived metrics. Pass null if using absolute metrics."
    )

    # 预留给 additional check 的字段
    proportion: float | None = Field(description="Optional proportion of real value. Pass null if not available.")
    base_proportion: float | None = Field(
        description="Optional proportion of predict value. Pass null if not available."
    )


class AdtributorToolInput(BaseModel):
    """Input schema for the adtributor drilldown tool."""

    reasoning: str = Field(description="Reason for using adtributor tool and what insights you expect to gain.")
    data: list[DrilldownRow] = Field(description="Melted table data representing the anomaly data.")
    derived: bool = Field(description="Whether the metric is derived (ratio).")
    issue_type: str = Field(default="drop", description="Type of anomaly: 'drop' or 'rise'. Default is 'drop'.")
    tep: float = Field(default=0.7, description="Threshold for cumulative explanatory power")
    teep: float = Field(default=0.02, description="Threshold for individual explanatory power")
    k: int = Field(default=1, description="Number of top candidate dimensions to return")


@tool("adtributor_drilldown", args_schema=AdtributorToolInput, return_direct=False)
def adtributor_drilldown(
    reasoning: str,
    data: list[DrilldownRow],
    derived: bool,
    issue_type: str = "drop",
    tep: float = 0.7,
    teep: float = 0.02,
    k: int = 1,
) -> dict[str, Any]:
    """
    Performs multi-dimensional root cause analysis on anomaly data using the Microsoft Adtributor algorithm.
    """
    # 1. Convert to DataFrame
    try:
        df = pd.DataFrame([row.model_dump(exclude_none=True) for row in data])
    except Exception as e:
        return {"error": f"Failed to parse input data: {str(e)}"}

    if df.empty:
        return {"error": "Input data is empty."}

    required_cols = ["dimension_name", "element_value"]
    if derived:
        required_cols.extend(["predict_numerator", "predict_denominator", "real_numerator", "real_denominator"])
    else:
        required_cols.extend(["predict", "real"])

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        return {"error": f"Missing required columns in data based on derived={derived}: {missing_cols}"}

    # 2. Transform melted table into dict[str, DataFrame]
    df_dict = {}
    dimensions = df["dimension_name"].unique()
    for dim in dimensions:
        dim_df = df[df["dimension_name"] == dim].copy()
        # Rename 'element_value' to actual dimension name to match algorithm expectation
        dim_df = dim_df.rename(columns={"element_value": dim})
        # Keep relevant columns
        cols_to_keep = [dim] + [c for c in required_cols if c not in ("dimension_name", "element_value")]
        if "proportion" in df.columns:
            cols_to_keep.append("proportion")
        if "base_proportion" in df.columns:
            cols_to_keep.append("base_proportion")

        df_dict[dim] = dim_df[cols_to_keep]

    # 3. Call algorithm
    try:
        output: AdtributorOutput = adtributor(
            derived=derived, df_dict=df_dict, issue_type=issue_type, tep=tep, teep=teep, k=k
        )
    except Exception as e:
        logger.exception("Error running adtributor algorithm")
        return {"error": f"Algorithm execution failed: {str(e)}"}

    # 4. Generate Narrative
    result = {"status": output.status, "root_causes": output.root_causes, "dimension_details": {}}

    if output.status == "no_anomaly_direction":
        result["summary_narrative"] = (
            f"The algorithm did not detect any valid attributes matching the '{issue_type}' direction."
        )
    elif output.status == "no_root_cause":
        result["summary_narrative"] = (
            "The anomaly is systemic and evenly distributed. No specific root cause elements exceeded the required threshold."
        )
    else:
        result["summary_narrative"] = "Root cause analysis successfully identified contributing elements."

    for dim, details in output.dimension_details.items():
        if details.elements:
            elements_str = ", ".join(map(str, details.elements))
            ep_pct = details.explanatory_power * 100 if details.explanatory_power else 0
            narrative = (
                f"The elements [{elements_str}] in this dimension contributed to {ep_pct:.2f}% of the overall anomaly."
            )
        else:
            narrative = f"This dimension was skipped or did not have specific elements causing the anomaly. Reason: {details.reason}"

        result["dimension_details"][dim] = {
            "contribution": details.explanatory_power,
            "elements": details.elements,
            "narrative": narrative,
            "raw_metrics": {
                "total_surprise": details.total_surprise,
                "surprise": details.surprise,
                "reason": details.reason,
            },
        }

    return result
