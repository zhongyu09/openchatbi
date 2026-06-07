from typing import Any

from pydantic import BaseModel, Field


class DimensionResult(BaseModel):
    explanatory_power: float | None = Field(default=None, description="Cumulative explanatory power")
    total_surprise: float = Field(description="Sum of surprise for all elements in this dimension")
    elements: list[Any] | None = Field(default=None, description="Root cause elements that passed thresholds")
    surprise: float | None = Field(default=None, description="Sum of surprise for the root cause elements")
    reason: str = Field(description="Reason why the dimension was selected or skipped")


class AdtributorOutput(BaseModel):
    root_causes: dict[str, list[Any]] = Field(description="Mapping of dimension name to list of root cause elements")
    ranked_dimensions: list[str] = Field(
        description="All analyzed dimensions sorted by total surprise in descending order"
    )
    dimension_details: dict[str, DimensionResult] = Field(description="Detailed diagnostic info for each dimension")
    status: str = Field(description="Analysis status: 'success', 'no_root_cause', or 'no_anomaly_direction'")
    reason_flag: str = Field(default="", description="Internal flag for skipping reasons")
