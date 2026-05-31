Tools and Utilities
===================

Overview
--------

LangGraph tools for human interaction, code execution, and knowledge search.

Python Code Execution
----------------------

.. automodule:: openchatbi.tool.run_python_code
    :members:
    :undoc-members:
    :show-inheritance:

Human Interaction
-----------------

.. automodule:: openchatbi.tool.ask_human
    :members:
    :undoc-members:
    :show-inheritance:

Memory Management
-----------------

.. automodule:: openchatbi.tool.memory
    :members:
    :undoc-members:
    :show-inheritance:

Knowledge Search
----------------

.. automodule:: openchatbi.tool.search_knowledge
    :members:
    :undoc-members:
    :show-inheritance:

Anomaly Detection
-----------------

Time series anomaly detection tool. Compares recent points against a
forecasting baseline and produces a multi-factor anomaly score. Shares the
time series forecasting service health check with ``timeseries_forecast`` —
when the forecasting service is unavailable, both tools are excluded.

.. automodule:: openchatbi.tool.anomaly_detection
    :members:
    :undoc-members:
    :show-inheritance:

Anomaly Drill-Down (Adtributor)
-------------------------------

Multi-dimensional root cause analysis based on the Microsoft Adtributor
algorithm. Accepts a 1D melted table (list of rows with
``dimension_name`` / ``element_value`` and baseline/actual values) and
returns the dimensions and elements that best explain an anomaly, with a
business-friendly narrative.

.. automodule:: openchatbi.tool.adtributor_tool
    :members:
    :undoc-members:
    :show-inheritance:

Data Analysis Agent
===================

The narrative documentation for the data analysis package — covering the
**Data Analysis Agent** and the two analysis algorithms (**anomaly detection**
and **Adtributor drill-down**) — lives in the package README, included below:

.. include:: ../../openchatbi/analysis/README.md
    :parser: myst_parser.sphinx_

API Reference
-------------

.. automodule:: openchatbi.analysis.agent
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: openchatbi.analysis.anomaly_detection
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: openchatbi.analysis.adtributor
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: openchatbi.analysis.models
    :members:
    :undoc-members:
    :show-inheritance: