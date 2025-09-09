#!/usr/bin/env python3
"""Test runner script for OpenChatBI."""

import argparse
import subprocess
import sys


def run_command(cmd, description):
    """Run a command and return the result."""
    print(f"\\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print("STDOUT:")
        print(result.stdout)

    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    if result.returncode != 0:
        print(f"❌ {description} failed with return code {result.returncode}")
        return False
    else:
        print(f"✅ {description} passed")
        return True


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run OpenChatBI tests")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--lint", action="store_true", help="Run linting checks")
    parser.add_argument("--type-check", action="store_true", help="Run type checking")
    parser.add_argument("--all", action="store_true", help="Run all checks (tests, lint, type-check)")
    parser.add_argument("--file", help="Run specific test file")

    args = parser.parse_args()

    # Determine test command
    base_cmd = ["uv", "run", "pytest"]

    if args.verbose:
        base_cmd.append("-v")

    if args.coverage:
        base_cmd.extend(["--cov=openchatbi", "--cov-report=html", "--cov-report=term-missing"])

    if args.unit:
        base_cmd.extend(["-m", "unit"])
    elif args.integration:
        base_cmd.extend(["-m", "integration"])
    elif args.fast:
        base_cmd.extend(["-m", "not slow"])

    if args.file:
        base_cmd.append(f"tests/{args.file}")

    success = True

    # Run tests
    if not args.lint and not args.type_check:
        success &= run_command(base_cmd, "Unit Tests")

    # Run linting if requested
    if args.lint or args.all:
        lint_commands = [
            (["uv", "run", "black", "--check", "."], "Black formatting check"),
            (["uv", "run", "isort", "--check-only", "."], "Import sorting check"),
            (["uv", "run", "ruff", "check", "."], "Ruff linting"),
            (["uv", "run", "bandit", "-r", "openchatbi/"], "Security scanning"),
        ]

        for cmd, desc in lint_commands:
            success &= run_command(cmd, desc)

    # Run type checking if requested
    if args.type_check or args.all:
        success &= run_command(["uv", "run", "mypy", "openchatbi/"], "Type checking")

    # Run all tests if --all is specified
    if args.all:
        test_commands = [
            (["uv", "run", "pytest", "-m", "unit", "-v"], "Unit Tests"),
            (["uv", "run", "pytest", "-m", "integration", "-v"], "Integration Tests"),
            (["uv", "run", "pytest", "--cov=openchatbi", "--cov-report=html"], "Coverage Report"),
        ]

        for cmd, desc in test_commands:
            success &= run_command(cmd, desc)

    # Print summary
    print(f"\\n{'=' * 60}")
    if success:
        print("🎉 All checks passed!")
        sys.exit(0)
    else:
        print("❌ Some checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
