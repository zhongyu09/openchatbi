#!/usr/bin/env python3
"""test_forecasting.py: Test script for Timer forecasting service."""

import time
from datetime import datetime, timedelta

import numpy as np
import requests
from requests.exceptions import RequestException


class TimeseriesForecastingTester:
    """Test class for Timer forecasting service."""

    def __init__(self, base_url="http://localhost:8765"):
        """Initialize the tester."""
        self.base_url = base_url
        self.predictions_endpoint = f"{base_url}/predict"
        self.health_endpoint = f"{base_url}/health"

    def generate_sample_data(self, length=100, frequency="H"):
        """Generate sample time series data for testing."""
        # Generate synthetic time series with trend and seasonality
        t = np.arange(length)

        # Add trend
        trend = 0.1 * t

        # Add seasonality (daily pattern for hourly data)
        if frequency == "H":
            seasonality = 5 * np.sin(2 * np.pi * t / 24)
        else:
            seasonality = 3 * np.sin(2 * np.pi * t / 7)  # Weekly pattern for daily data

        # Add noise
        noise = np.random.normal(0, 1, length)

        # Combine components
        values = 100 + trend + seasonality + noise

        return values.tolist()

    def test_basic_forecasting(self):
        """Test basic time series forecasting."""
        print("\n=== Testing Basic Forecasting ===")

        # Generate sample data
        sample_data = self.generate_sample_data(length=168, frequency="H")  # 1 week of hourly data

        # Prepare request payload
        payload = {
            "input": sample_data,
            "forecast_window": 24,  # Forecast next 24 hours
            "frequency": "H",
            "input_len": 168,  # Use last week of data
        }

        # Send request
        try:
            response = requests.post(
                self.predictions_endpoint, json=payload, headers={"Content-Type": "application/json"}, timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                print("✓ Basic forecasting successful")
                print(f"  - Input length: {len(sample_data)}")
                print(f"  - Forecast Window: {payload['forecast_window']}")
                print(f"  - Predictions length: {len(result.get('predictions', []))}")
                print(f"  - Sample predictions: {result.get('predictions', [])[:5]}")
                return True
            else:
                print(f"✗ Request failed with status: {response.status_code}")
                print(f"  Response: {response.text}")
                return False

        except requests.exceptions.RequestException as e:
            print(f"✗ Request failed: {str(e)}")
            return False

    def test_structured_data(self):
        """Test forecasting with structured data (timestamps + values)."""
        print("\n=== Testing Structured Data Forecasting ===")

        # Generate structured data with timestamps
        start_time = datetime.now() - timedelta(days=7)
        structured_data = []

        for i in range(168):  # 1 week of hourly data
            timestamp = start_time + timedelta(hours=i)
            value = 100 + 0.1 * i + 5 * np.sin(2 * np.pi * i / 24) + np.random.normal(0, 1)

            structured_data.append({"timestamp": timestamp.isoformat(), "value": value})

        # Prepare request payload
        payload = {
            "input": structured_data,
            "forecast_window": 48,  # Forecast next 48 hours
            "frequency": "H",
            "target_column": "value",
        }

        # Send request
        try:
            response = requests.post(
                self.predictions_endpoint, json=payload, headers={"Content-Type": "application/json"}, timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                print("✓ Structured data forecasting successful")
                print(f"  - Input records: {len(structured_data)}")
                print(f"  - Forecast Window: {payload['forecast_window']}")
                print(f"  - Predictions length: {len(result.get('predictions', []))}")
                return True
            else:
                print(f"✗ Request failed with status: {response.status_code}")
                print(f"  Response: {response.text}")
                return False

        except requests.exceptions.RequestException as e:
            print(f"✗ Request failed: {str(e)}")
            return False

    def test_different_windows(self):
        """Test forecasting with different forecast windows."""
        print("\n=== Testing Different Forecast Horizons ===")

        sample_data = self.generate_sample_data(length=100)
        windows = [1, 12, 24, 48, 72]

        for window in windows:
            payload = {"input": sample_data, "forecast_window": window, "frequency": "H"}

            try:
                response = requests.post(
                    self.predictions_endpoint, json=payload, headers={"Content-Type": "application/json"}, timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    predictions_len = len(result.get("predictions", []))
                    print(f"✓ Window {window}: {predictions_len} predictions")
                else:
                    print(f"✗ Window {window}: Failed with status {response.status_code}")
                    return False

            except requests.exceptions.RequestException as e:
                print(f"✗ Window {window}: Request failed - {str(e)}")
                return False
        return True

    def test_error_handling(self):
        """Test error handling with invalid inputs."""
        print("\n=== Testing Error Handling ===")

        # Test empty input
        try:
            response = requests.post(
                self.predictions_endpoint, json={"input": []}, headers={"Content-Type": "application/json"}, timeout=10
            )
            print(f"Empty input: Status {response.status_code}")
            if response.status_code != 400:
                print("✗ Empty input: Expected 400 status code")
                return False
        except RequestException:
            print("Empty input: exception occurred not expected")
            return False

        # Test invalid JSON
        try:
            response = requests.post(
                self.predictions_endpoint, data="invalid json", headers={"Content-Type": "application/json"}, timeout=10
            )
            print(f"Invalid JSON: Status {response.status_code}")
            if response.status_code != 422:
                print("✗ Empty input: Expected 422 status code")
                return False
        except RequestException:
            print("Empty input: exception occurred not expected")
            return False
        return True

    def test_health_check(self):
        """Test service health check."""
        print("\n=== Testing Service Health ===")

        try:
            # Test health endpoint
            response = requests.get(self.health_endpoint, timeout=5)

            if response.status_code == 200:
                result = response.json()
                print("✓ Service health check passed")
                print(f"  - Model initialized: {result.get('model_initialized', 'Unknown')}")
                print(f"  - Uptime: {result.get('uptime_seconds', 'Unknown')} seconds")
                return True
            else:
                print(f"✗ Health check failed: {response.status_code}")
                return False

        except RequestException as e:
            print(f"Health check failed: {str(e)}")
            return False

    def run_all_tests(self):
        """Run all tests."""
        print("=" * 50)
        print("TIMER FORECASTING SERVICE TESTS")
        print("=" * 50)

        # Wait for service to be ready
        print("Waiting for service to be ready...")
        for _i in range(30):  # Wait up to 30 seconds
            try:
                response = requests.get(self.health_endpoint, timeout=2)
                if response.status_code == 200:
                    result = response.json()
                    if result.get("model_initialized", False):
                        print("✓ Service is ready")
                        break
            except RequestException:
                pass
            time.sleep(1)
        else:
            print("✗ Service not ready after 30 seconds")
            return False

        # Run tests
        tests = [
            self.test_health_check,
            self.test_basic_forecasting,
            self.test_structured_data,
            self.test_different_windows,
            self.test_error_handling,
        ]

        passed = 0
        total = len(tests)

        for test in tests:
            try:
                if test():
                    passed += 1
            except Exception as e:
                print(f"✗ Test {test.__name__} failed with exception: {str(e)}")

        print("\n" + "=" * 50)
        print(f"TESTS COMPLETED: {passed}/{total} passed")
        print("=" * 50)

        return passed == total


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Timer forecasting service")
    parser.add_argument(
        "--url", default="http://localhost:8765", help="Base URL of the service (default: http://localhost:8080)"
    )

    args = parser.parse_args()

    tester = TimeseriesForecastingTester(args.url)
    success = tester.run_all_tests()

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
