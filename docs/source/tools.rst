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

Overview
--------

The data analysis agent is a specialized sub-agent (built on the `deepagents
<https://github.com/langchain-ai/deepagents>`_ framework) that orchestrates the
analysis tools above into multi-step workflows. The main agent delegates complex
analysis requests to it through the ``data_analysis`` tool, which keeps the
main agent's context focused while the sub-agent handles planning and tool
orchestration.

Supported scenarios:

- **Single metric trend forecasting** — ``text2sql`` → ``timeseries_forecast``
- **Single metric anomaly detection** — ``text2sql`` → ``anomaly_detection``
- **Single metric anomaly drill-down** — ``text2sql`` (1D melted table) → ``adtributor_drilldown``
- **Multi-metric correlation** — ``text2sql`` → ``run_python_code``
- **Business combination analysis** — ``text2sql`` → ``run_python_code``

Tool dependencies and health checks:

- ``text2sql`` reuses the main SQL generation subgraph.
- ``timeseries_forecast`` and ``anomaly_detection`` require the forecasting
  service to be healthy; both are excluded together when it is not.
- ``adtributor_drilldown`` and ``run_python_code`` have no external service
  dependency.

LLM selection:

- The agent uses the optional ``analysis_llm`` configuration when present,
  otherwise it falls back to ``default_llm`` (see
  :func:`openchatbi.llm.llm.get_analysis_llm`).

Sub-agent isolation:

- The sub-agent is invoked with a derived child ``thread_id``
  (``"{parent_thread_id}:data_analysis"``) so it does not clobber the main
  agent's checkpoint thread, while keeping interrupt/resume deterministic.

.. automodule:: openchatbi.analysis.agent
    :members:
    :undoc-members:
    :show-inheritance: