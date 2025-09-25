import docker
import tempfile
import os
import subprocess
import shutil
from pathlib import Path
from typing import Tuple
from docker.errors import ContainerError

from openchatbi.code.executor_base import ExecutorBase


def check_docker_status() -> Tuple[bool, str]:
    """
    Check Docker installation and status without initializing DockerExecutor.

    Returns:
        Tuple[bool, str]: (is_available, status_message)
    """
    try:
        # Check if Docker CLI is installed
        if not shutil.which("docker"):
            return False, "Docker is not installed. Please install Docker."

        # Check if Docker daemon is running
        result = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            return True, "Docker is installed and running"
        else:
            if "Cannot connect to the Docker daemon" in result.stderr:
                return False, "Docker is installed but not running. Please start the Docker daemon."
            else:
                return False, f"Docker is not available: {result.stderr.strip()}"

    except subprocess.TimeoutExpired:
        return False, "Docker command timed out. Docker may not be running properly."
    except FileNotFoundError:
        return False, "Docker command not found. Please install Docker."
    except Exception as e:
        return False, f"Error checking Docker status: {str(e)}"


class DockerExecutor(ExecutorBase):
    """Docker-based Python code executor for isolated execution."""

    def __init__(self, variable: dict = None):
        super().__init__(variable)
        self.image_name = "python-executor"
        self.dockerfile_path = Path(__file__).parent.parent.parent / "Dockerfile.python-executor"

        # Check Docker installation and status
        self._check_docker_availability()

        try:
            self.client = docker.from_env()
            # Build Docker image if it doesn't exist
            self._ensure_image_exists()
        except Exception as e:
            self._handle_docker_error(e)

    @staticmethod
    def _check_docker_availability():
        """Check if Docker is installed and available."""
        # Check if Docker CLI is installed
        if not shutil.which("docker"):
            raise RuntimeError("Docker is not installed. Please install Docker and ensure it's in your system PATH.")

        # Check if Docker daemon is running
        try:
            result = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                if "Cannot connect to the Docker daemon" in result.stderr:
                    raise RuntimeError(
                        "Docker is installed but not running. Please start the Docker daemon and try again."
                    )
                else:
                    raise RuntimeError(
                        f"Docker is not available. Please check Docker installation and status. "
                        f"Error: {result.stderr.strip()}"
                    )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Docker command timed out. Please check if Docker is running properly.")
        except FileNotFoundError:
            raise RuntimeError("Docker command not found. Please install Docker and ensure it's in your system PATH.")

    @staticmethod
    def _handle_docker_error(error: Exception):
        """Handle Docker-related errors with specific error messages."""
        error_str = str(error).lower()

        if "connection aborted" in error_str and "no such file or directory" in error_str:
            raise RuntimeError("Docker is not running. Please start the Docker daemon and try again.")
        elif "permission denied" in error_str:
            raise RuntimeError(
                "Permission denied accessing Docker. Please ensure your user has Docker permissions "
                "or try running with appropriate privileges."
            )
        elif "docker daemon" in error_str or "connection refused" in error_str:
            raise RuntimeError("Cannot connect to Docker daemon. Please start the Docker daemon and try again.")
        else:
            raise RuntimeError(
                f"Failed to initialize Docker client. Please ensure Docker is installed and running. "
                f"Error: {str(error)}"
            )

    def _ensure_image_exists(self):
        """Build Docker image if it doesn't exist."""
        try:
            self.client.images.get(self.image_name)
        except docker.errors.ImageNotFound:
            print(f"Building Docker image '{self.image_name}'...")
            self.client.images.build(
                path=str(self.dockerfile_path.parent),
                dockerfile=self.dockerfile_path.name,
                tag=self.image_name,
                rm=True,
            )
            print(f"Docker image '{self.image_name}' built successfully.")

    def run_code(self, code: str) -> Tuple[bool, str]:
        """Execute Python code in a Docker container."""
        try:
            # Create a temporary file with the code
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                # Add variable definitions to the code
                variable_code = ""
                for key, value in self._variable.items():
                    if isinstance(value, str):
                        variable_code += f'{key} = "{value}"\n'
                    else:
                        variable_code += f"{key} = {repr(value)}\n"

                full_code = variable_code + "\n" + code
                f.write(full_code)
                temp_file_path = f.name

            try:
                # Run the code in a Docker container
                container = self.client.containers.run(
                    self.image_name,
                    command=["python3", f"/app/{os.path.basename(temp_file_path)}"],
                    volumes={temp_file_path: {"bind": f"/app/{os.path.basename(temp_file_path)}", "mode": "ro"}},
                    remove=True,
                    detach=False,
                    stdout=True,
                    stderr=True,
                    network_mode="none",  # Disable network access for security
                )

                # Get the output
                output = container.decode("utf-8")
                return True, output

            except ContainerError as e:
                # Container exited with non-zero code
                error_output = e.stderr if e.stderr else str(e)
                return False, f"Container execution failed: {error_output}"

        except Exception as e:
            return False, f"Docker execution error: {str(e)}"

        finally:
            # Clean up temporary file
            if "temp_file_path" in locals() and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except (OSError, PermissionError) as e:
                    # Log but don't fail the operation for cleanup issues
                    print(f"Warning: Failed to clean up temporary file {temp_file_path}: {e}")

    def __del__(self):
        """Clean up Docker client on deletion."""
        try:
            if hasattr(self, "client") and self.client is not None:
                self.client.close()
        except Exception:
            # Ignore cleanup errors during object destruction
            pass
