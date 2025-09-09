"""Tool for asking human clarification when information is ambiguous."""

from pydantic import BaseModel, Field


class AskHuman(BaseModel):
    """Ask user for clarification when data is missing or ambiguous.

    Use this tool ONLY when you are STRONGLY certain that information is
    ambiguous or missing. First try to solve the question with available
    user input before calling this tool.
    """

    question: str = Field(description="Question to ask the user for clarification")
    options: list[str] = Field(description="Options for user to choose (max 3). Empty if not a choice question.")
