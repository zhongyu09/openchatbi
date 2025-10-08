"""Test runner script for context management tests."""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(test_type="all", verbose=False, coverage=False):
    """Run context management tests.

    Args:
        test_type: Type of tests to run ('all', 'unit', 'integration', 'performance')
        verbose: Enable verbose output
        coverage: Enable coverage reporting
    """
    # Base pytest command
    cmd = ["python", "-m", "pytest"]

    # Test directory
    test_dir = Path(__file__).parent

    # Add specific test files based on type
    if test_type == "all":
        cmd.append(str(test_dir))
    elif test_type == "unit":
        cmd.extend([str(test_dir / "test_context_manager.py"), str(test_dir / "test_context_config.py")])
    elif test_type == "integration":
        cmd.append(str(test_dir / "test_agent_graph_integration.py"))
    elif test_type == "performance":
        cmd.extend([str(test_dir / "test_performance_edge_cases.py"), "-m", "performance"])
    else:
        print(f"Unknown test type: {test_type}")
        return False

    # Add verbose flag
    if verbose:
        cmd.append("-v")

    # Add coverage
    if coverage:
        cmd.extend(
            [
                "--cov=openchatbi.context_manager",
                "--cov=openchatbi.context_config",
                "--cov-report=html",
                "--cov-report=term-missing",
            ]
        )

    # Add other useful flags
    cmd.extend(
        [
            "--tb=short",  # Shorter traceback format
            "-x",  # Stop on first failure
            "--strict-markers",  # Strict marker checking
        ]
    )

    print(f"Running command: {' '.join(cmd)}")
    print("-" * 50)

    # Run the tests
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return False


def main():
    """Main function for test runner."""
    parser = argparse.ArgumentParser(description="Run context management tests")

    parser.add_argument(
        "--type",
        "-t",
        choices=["all", "unit", "integration", "performance"],
        default="all",
        help="Type of tests to run (default: all)",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    parser.add_argument("--coverage", "-c", action="store_true", help="Enable coverage reporting")

    parser.add_argument(
        "--quick", "-q", action="store_true", help="Run only quick tests (exclude slow/performance tests)"
    )

    args = parser.parse_args()

    if args.quick:
        # Exclude slow tests
        cmd_extra = ["-m", "not slow and not performance"]
        print("Running quick tests only (excluding slow and performance tests)")
    else:
        cmd_extra = []

    success = run_tests(test_type=args.type, verbose=args.verbose, coverage=args.coverage)

    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
