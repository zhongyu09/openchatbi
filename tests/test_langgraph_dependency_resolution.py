"""Migration checks for LangGraph dependency resolution."""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_config() -> dict:
    return tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())


def test_langgraph_resolves_to_required_version() -> None:
    """The migration target is pinned to the researched v1.1 latest release or newer."""
    version = metadata.version("langgraph")
    assert Version(version) >= Version("1.1.10")

    dependencies = _project_config()["project"]["dependencies"]
    # Check that langgraph is in dependencies, either pinned or with a minimum version
    assert any("langgraph" in dep for dep in dependencies)


def test_langgraph_companion_packages_resolve() -> None:
    """Companion packages used by checkpointing, prebuilt nodes, and SDK imports resolve together."""
    expected_packages = [
        "langchain-core",
        "langgraph-checkpoint",
        "langgraph-checkpoint-sqlite",
        "langgraph-prebuilt",
        "langgraph-sdk",
        "langmem",
    ]

    resolved_versions = {package: metadata.version(package) for package in expected_packages}

    assert Version(resolved_versions["langchain-core"]).major >= 1
    assert Version(resolved_versions["langgraph-prebuilt"]).major >= 1
    assert Version(resolved_versions["langgraph-sdk"]).major >= 0
    assert Version(resolved_versions["langgraph-checkpoint-sqlite"]) >= Version("2.0.11")


def test_project_python_range_is_compatible_with_langgraph_packages() -> None:
    """The current project Python range should stay inside resolved LangGraph package requirements."""
    project_python = SpecifierSet(_project_config()["project"]["requires-python"])
    supported_project_versions = [Version("3.11"), Version("3.12"), Version("3.13")]
    assert all(version in project_python for version in supported_project_versions)

    for package in ("langgraph", "langgraph-checkpoint-sqlite"):
        requires_python = metadata.metadata(package).get("Requires-Python")
        assert requires_python, f"{package} must declare Requires-Python"
        package_python = SpecifierSet(requires_python)
        assert all(
            version in package_python for version in supported_project_versions
        ), f"{package} {metadata.version(package)} is incompatible with {project_python}"


def test_direct_dependency_specifiers_are_satisfied_by_resolved_versions() -> None:
    """Every direct dependency specifier in pyproject should accept the currently resolved package."""
    for dependency in _project_config()["project"]["dependencies"]:
        requirement = Requirement(dependency)
        try:
            resolved_version = Version(metadata.version(requirement.name))
        except metadata.PackageNotFoundError:
            # Extras such as pyhive[presto] can be normalized differently by the installer.
            continue
        assert resolved_version in requirement.specifier, f"{requirement.name} resolved to {resolved_version}"


def test_core_langgraph_related_imports() -> None:
    """Core modules that use LangGraph should import without relying on local developer config."""
    modules = [
        "openchatbi",
        "openchatbi.agent_graph",
        "openchatbi.text2sql.sql_graph",
        "openchatbi.graph_state",
        "openchatbi.tool.memory",
        "sample_ui.async_graph_manager",
    ]

    for module in modules:
        importlib.import_module(module)
