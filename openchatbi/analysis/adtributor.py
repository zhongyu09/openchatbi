import logging
from typing import Any

import numpy as np
import pandas as pd

from openchatbi.analysis.models import AdtributorOutput, DimensionResult

logger = logging.getLogger(__name__)

NO_SPECIFIC_INV = "NO_SPECIFIC_INV"
ATTRS_NOT_FOUND = "ATTRS_NOT_FOUND"


def additional_check(dimension: str, df: pd.DataFrame, attr_list: list[Any]) -> tuple[bool, str]:
    """
    1. if rc real value proportion ~= 100%, skip this dimension
    2. if all attrs evenly drop, skip this check for now
    """
    checks = {"proportion": 0.98, "base_proportion": 0.98}
    reason = []
    for check, threshold in checks.items():
        if check not in df.columns:
            continue
        target_proportion = df.loc[df[dimension].isin(attr_list)][check].sum()
        if target_proportion >= threshold:
            return False, f"skip: root cause items {attr_list} {check} is {target_proportion:.2%} ~ 100%"
        reason.append(f"{check} is {target_proportion:.2%} < {threshold:.2%}")

    return True, ". ".join(reason)


def add_surprise(df: pd.DataFrame, derived: bool, merged_divide: int = 1) -> pd.DataFrame:
    """Computes the surprise for all elements in the dataframe."""

    def compute_surprise(col_real: str, col_predict: str) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            f = df[col_predict].sum() / merged_divide
            a = df[col_real].sum() / merged_divide

            p = df[col_predict] / f
            q = df[col_real] / a

            # Using JS divergence like formulation
            p_term = np.nan_to_num(p * np.log(2 * p / (p + q)))
            q_term = np.nan_to_num(q * np.log(2 * q / (p + q)))
            surprise = 0.5 * (p_term + q_term)
        return surprise

    if derived:
        df["surprise"] = compute_surprise("real_numerator", "predict_numerator") + compute_surprise(
            "real_denominator", "predict_denominator"
        )
    else:
        df["surprise"] = compute_surprise("real", "predict")
    return df


def add_explanatory_power(df: pd.DataFrame, derived: bool, issue_type: str = "drop") -> pd.DataFrame:
    """Computes the explanatory power for all elements in the dataframe."""
    if derived:
        f_a = df["predict_numerator"].sum()
        f_b = df["predict_denominator"].sum()

        n = (df["real_numerator"] - df["predict_numerator"]) * f_b - (
            df["real_denominator"] - df["predict_denominator"]
        ) * f_a
        d = f_b * (f_b + df["real_denominator"] - df["predict_denominator"])
        df["ep"] = n / d

        # Clean up invalid values
        df["ep"] = df["ep"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Normalize to sum up to 1
        total_ep = df["ep"].sum()
        if abs(total_ep) > 1e-9:
            df["ep"] = df["ep"] / total_ep
        else:
            df["ep"] = 0.0
    else:
        f = df["predict"].sum()
        a = df["real"].sum()
        if abs(a - f) > 1e-9:
            df["ep"] = (df["real"] - df["predict"]) / (a - f)
        else:
            df["ep"] = 0.0
    return df


def merge_dimensions(df: pd.DataFrame, derived: bool) -> pd.DataFrame:
    if derived:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["predict"] = np.nan_to_num(
                df["predict_numerator"] / df["predict_denominator"], nan=0.0, posinf=0.0, neginf=0.0
            )
            df["real"] = np.nan_to_num(df["real_numerator"] / df["real_denominator"], nan=0.0, posinf=0.0, neginf=0.0)
    df = df.reset_index(drop=True)
    return df


def adtributor(
    derived: bool,
    df_dict: dict[str, pd.DataFrame],
    dimension_weights: dict[str, float] | None = None,
    tep: float = 0.7,
    teep: float = 0.02,
    k: int = 1,
    issue_type: str = "drop",
) -> AdtributorOutput:
    """
    Analyzes the input data and identifies candidate dimensions for drill-down analysis.
    """
    if dimension_weights is None:
        dimension_weights = {}

    candidates = []
    reason_flag = ""
    status = "success"

    for d, dim_df in df_dict.items():
        if dim_df.empty:
            continue

        elements = merge_dimensions(dim_df, derived)

        if issue_type == "drop":
            elements = elements.loc[elements["predict"] > elements["real"] + 0.001].copy()
        elif issue_type == "rise":
            elements = elements.loc[elements["predict"] + 0.001 < elements["real"]].copy()

        if elements.empty:
            logger.info(f"Skip {d} drill-down: no attributes match the issue_type {issue_type}")
            candidates.append(
                {
                    "dimension": d,
                    "total_surprise": 0.0,
                    "reason": f"Skip: no attributes match the issue_type {issue_type}",
                    "skipped_direction": True,
                }
            )
            continue

        elements = add_explanatory_power(elements, derived, issue_type)
        elements = add_surprise(elements, derived, merged_divide=len([d]))

        dim_elems = elements.set_index(d)
        dim_elems = dim_elems.sort_values("surprise", ascending=False)

        cumulative_ep = dim_elems.loc[dim_elems["ep"] > teep, "ep"].cumsum()

        dimension_weight = dimension_weights.get(d, 1.0)
        total_surprise = float(dim_elems["surprise"].sum() * dimension_weight)

        candidate = {
            "explanatory_power": float(cumulative_ep.max()) if not cumulative_ep.empty else 0.0,
            "total_surprise": total_surprise,
            "dimension": d,
        }

        reason = ""
        if not cumulative_ep.empty and np.any(cumulative_ep > tep):
            idx = (cumulative_ep > tep).idxmax()
            attr_list = cumulative_ep.loc[:idx].index.values.tolist()
            further_check_passed, check_reason = additional_check(d, elements, attr_list)

            if further_check_passed:
                candidate["elements"] = attr_list
                candidate["surprise"] = float(dim_elems.loc[attr_list, "surprise"].sum() * dimension_weight)
                candidate["explanatory_power"] = float(cumulative_ep.loc[idx])
                reason = check_reason if check_reason else "Passed all thresholds."
            else:
                reason = check_reason
                reason_flag = ATTRS_NOT_FOUND
        else:
            reason = f"skip: cumulative_ep({candidate['explanatory_power']:.4f}) < tep({tep})"
            reason_flag = ATTRS_NOT_FOUND

        candidate["reason"] = reason
        logger.info(f"Dimension: {d}, {reason}")
        candidates.append(candidate)

    if not candidates or all(c.get("skipped_direction", False) for c in candidates):
        return AdtributorOutput(
            root_causes={},
            ranked_dimensions=[],
            dimension_details={},
            status="no_anomaly_direction",
            reason_flag=NO_SPECIFIC_INV,
        )

    # Rank dimensions
    ranked_candidates = sorted(
        [c for c in candidates if not c.get("skipped_direction", False)],
        key=lambda x: x.get("total_surprise", 0),
        reverse=True,
    )
    ranked_dimensions = [c["dimension"] for c in ranked_candidates]

    # Get root causes (top k)
    rc_candidates = sorted([c for c in candidates if "elements" in c], key=lambda t: t["surprise"], reverse=True)[:k]

    root_causes = {c["dimension"]: c["elements"] for c in rc_candidates}

    if not root_causes:
        status = "no_root_cause"

    dimension_details = {}
    for c in candidates:
        dim = c["dimension"]
        dimension_details[dim] = DimensionResult(
            explanatory_power=c.get("explanatory_power"),
            total_surprise=c.get("total_surprise", 0.0),
            elements=c.get("elements"),
            surprise=c.get("surprise"),
            reason=c.get("reason", ""),
        )

    return AdtributorOutput(
        root_causes=root_causes,
        ranked_dimensions=ranked_dimensions,
        dimension_details=dimension_details,
        status=status,
        reason_flag=reason_flag,
    )
