"""Tool for saving reports to files."""

import datetime
from pathlib import Path

from langchain.tools import tool
from pydantic import BaseModel, Field

from openchatbi import config
from openchatbi.utils import log


class SaveReportInput(BaseModel):
    content: str = Field(description="The content of the report to save")
    title: str = Field(description="The title of the report (will be used in filename)")
    file_format: str = Field(description="The file format/extension (e.g., 'md', 'csv', 'txt', 'json')")


@tool("save_report", args_schema=SaveReportInput, return_direct=False, infer_schema=True)
def save_report(content: str, title: str, file_format: str = "md") -> str:
    """Save a report to a file with timestamp and title in filename.

    Args:
        content: The content of the report to save
        title: The title of the report (will be used in filename)
        file_format: The file format/extension (e.g., 'md', 'csv', 'txt', 'json')

    Returns:
        str: Success message with download link or error message
    """
    try:
        # Get report directory from config
        report_dir = config.get().report_directory

        # Create directory if it doesn't exist
        Path(report_dir).mkdir(parents=True, exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Clean title for filename (remove invalid characters)
        clean_title = "".join(c for c in title if c.isalnum() or c in (" ", "-")).rstrip()
        clean_title = clean_title.replace(" ", "_")

        # Ensure file format doesn't have leading dot
        file_format = file_format.lstrip(".")

        # Create filename
        filename = f"{timestamp}_{clean_title}.{file_format}"
        file_path = Path(report_dir) / filename

        # Write content to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        log(f"Report saved: {file_path}")

        # Return success message with download link
        download_url = f"/api/download/report/{filename}"
        return f"Report saved successfully! Download link: {download_url}"

    except Exception as e:
        error_msg = f"Failed to save report: {str(e)}"
        log(error_msg)
        return error_msg
