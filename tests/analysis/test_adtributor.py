import pandas as pd

from openchatbi.analysis.adtributor import adtributor
from openchatbi.tool.adtributor_tool import adtributor_drilldown


def test_adtributor_absolute():
    # Setup test data similar to demo
    df_site = pd.DataFrame(
        {
            "site_section": [1, 2, 3],
            "predict": [0.5, 0.1, 0.9],
            "real": [0.7, 0.9, 1.0],
            "proportion": [0.2, 0.5, 0.3],
            "base_proportion": [0.2, 0.5, 0.3],
        }
    )
    df_dict = {"site_section": df_site}

    # Execute
    output = adtributor(derived=False, df_dict=df_dict, issue_type="drop", tep=0.7)

    # Note: For drop, we look at where predict > real.
    # In the demo data:
    # row 1: predict 0.5, real 0.7 -> predict < real (rise)
    # row 2: predict 0.1, real 0.9 -> predict < real (rise)
    # row 3: predict 0.9, real 1.0 -> predict < real (rise)
    # None of them drop!
    assert output.status == "no_anomaly_direction"

    # Let's test a real drop
    df_site_drop = pd.DataFrame(
        {
            "site_section": [1, 2, 3],
            "predict": [10.0, 5.0, 2.0],
            "real": [8.0, 1.0, 2.0],  # 1 and 2 dropped, 2 dropped heavily
            "proportion": [0.5, 0.3, 0.2],
            "base_proportion": [0.5, 0.3, 0.2],
        }
    )
    output = adtributor(derived=False, df_dict={"site_section": df_site_drop}, issue_type="drop", tep=0.7)

    assert output.status == "success"
    assert "site_section" in output.root_causes
    # 2 dropped from 5 to 1 (large drop compared to size), 1 dropped from 10 to 8
    # 2 should be the primary root cause
    assert 2 in output.root_causes["site_section"]


BASE_FIELDS = {
    "predict": None,
    "real": None,
    "predict_numerator": None,
    "predict_denominator": None,
    "real_numerator": None,
    "real_denominator": None,
    "proportion": None,
    "base_proportion": None,
}


def test_adtributor_tool_interface():
    # Test melted table format
    melted_data = [
        {**BASE_FIELDS, "dimension_name": "device", "element_value": "ios", "predict": 1000, "real": 500},
        {**BASE_FIELDS, "dimension_name": "device", "element_value": "android", "predict": 2000, "real": 1900},
        {**BASE_FIELDS, "dimension_name": "province", "element_value": "guangdong", "predict": 500, "real": 200},
        {**BASE_FIELDS, "dimension_name": "province", "element_value": "beijing", "predict": 500, "real": 450},
    ]

    result = adtributor_drilldown.invoke(
        {"reasoning": "Test drilldown", "data": melted_data, "derived": False, "issue_type": "drop", "k": 2}
    )

    assert "error" not in result
    assert result["status"] == "success"
    assert "device" in result["dimension_details"]
    assert "province" in result["dimension_details"]

    # iOS dropped by 500 (50%), android by 100 (5%). iOS should be a root cause.
    assert "ios" in result["root_causes"].get("device", [])
    assert "guangdong" in result["root_causes"].get("province", [])

    # Narrative check
    narrative = result["dimension_details"]["device"]["narrative"]
    assert "ios" in narrative
    assert "contributed to" in narrative


def test_adtributor_derived_metric_tool():
    # Test derived metric via tool interface
    melted_data = [
        {
            **BASE_FIELDS,
            "dimension_name": "ad_type",
            "element_value": "banner",
            "predict_numerator": 50,  # clicks
            "predict_denominator": 1000,  # impressions -> CTR 5%
            "real_numerator": 10,
            "real_denominator": 1000,  # CTR 1% -> heavy drop
        },
        {
            **BASE_FIELDS,
            "dimension_name": "ad_type",
            "element_value": "video",
            "predict_numerator": 20,
            "predict_denominator": 100,  # CTR 20%
            "real_numerator": 19,
            "real_denominator": 100,  # CTR 19% -> minor drop
        },
    ]

    result = adtributor_drilldown.invoke(
        {"reasoning": "Test derived drilldown", "data": melted_data, "derived": True, "issue_type": "drop"}
    )

    assert "error" not in result
    assert result["status"] == "success"
    assert "banner" in result["root_causes"].get("ad_type", [])


def test_adtributor_complex_case():
    import random

    random.seed(42)

    dimensions = ["region", "device", "browser", "channel"]
    melted_data = []

    # Generate 12 attributes per dimension (at least 10)
    for dim in dimensions:
        for i in range(12):
            attr_name = f"{dim}_{i}"
            # Base values
            predict_val = random.randint(500, 1500)

            # Inject anomaly: Region_3 and Device_7 have heavy drops
            if attr_name == "region_3":
                real_val = predict_val * 0.1  # 90% drop
            elif attr_name == "device_7":
                real_val = predict_val * 0.15  # 85% drop
            else:
                # Normal fluctuation (+- 5%)
                real_val = predict_val * random.uniform(0.95, 1.05)

            melted_data.append(
                {
                    **BASE_FIELDS,
                    "dimension_name": dim,
                    "element_value": attr_name,
                    "predict": predict_val,
                    "real": real_val,
                }
            )

    result = adtributor_drilldown.invoke(
        {
            "reasoning": "Test complex drilldown with 4 dims and 10+ attrs",
            "data": melted_data,
            "derived": False,
            "issue_type": "drop",
            "k": 2,
        }
    )

    assert "error" not in result
    assert result["status"] == "success"

    # Verify the ones with heavy drops are detected as root causes
    assert "region_3" in result["root_causes"].get("region", [])
    assert "device_7" in result["root_causes"].get("device", [])

    # Verify narrative exists and contains the root cause
    assert "region_3" in result["dimension_details"]["region"]["narrative"]
    assert "device_7" in result["dimension_details"]["device"]["narrative"]
