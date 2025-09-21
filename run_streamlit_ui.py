#!/usr/bin/env python3
"""
Launch script for the Streamlit-based OpenChatBI interface.

Usage:
    python run_streamlit_ui.py

This will start the Streamlit server on http://localhost:8501
"""

import subprocess
import sys
import os


def main():
    """Launch the Streamlit UI"""
    # Change to the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)

    print("🚀 Starting OpenChatBI Streamlit Interface...")
    print("📍 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)

    try:
        # Run streamlit with the new UI file
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "sample_ui/streamlit_ui.py",
                "--server.port=8501",
                "--server.address=localhost",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        print("\n👋 Stopping Streamlit server...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting Streamlit: {e}")
        print("\n💡 Make sure Streamlit is installed:")
        print("   pip install streamlit")
    except FileNotFoundError:
        print("❌ Python or Streamlit not found")
        print("\n💡 Make sure Python and Streamlit are installed:")
        print("   pip install streamlit")


if __name__ == "__main__":
    main()
