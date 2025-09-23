"""Utility functions for OpenChatBI."""

import json
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.messages import AIMessageChunk
from langchain_core.vectorstores import VectorStore
from regex import regex
from fastapi import HTTPException
from fastapi.responses import FileResponse


def log(args) -> None:
    """Log messages to stderr for debugging."""
    print(args, file=sys.stderr, flush=True)


def get_text_from_content(content: str | list[str | dict]) -> str:
    """Extract text from various content formats.

    Args:
        content: String, list of strings, or list of dicts with 'text' key.

    Returns:
        str: Extracted text content.
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        if isinstance(content[0], str):
            return "".join(content)
        elif isinstance(content[0], dict):
            return "".join([item.get("text", "") for item in content])
    return ""


def get_text_from_message_chunk(chunk: AIMessageChunk) -> str:
    """Extract content from an AIMessageChunk.

    Args:
        chunk (AIMessageChunk): The message chunk to extract text from.

    Returns:
        str: Extracted text content or empty string.
    """
    if not hasattr(chunk, "content") or not chunk.content:
        return ""
    return get_text_from_content(chunk.content)


def extract_json_from_answer(answer: str) -> dict:
    """Extract the first JSON object from a string answer.

    Args:
        answer (str): String that may contain JSON objects.

    Returns:
        dict: Parsed JSON object or empty dict if none found.
    """
    pattern = regex.compile(r"\{(?:[^{}]+|(?R))*\}")
    matches = pattern.findall(answer)
    json_result = matches[0] if matches else "{}"
    return json.loads(json_result)


def get_report_download_response(filename: str) -> FileResponse:
    """Get FileResponse for downloading a report file.

    Args:
        filename: The filename of the report to download

    Returns:
        FileResponse: Response object for file download

    Raises:
        HTTPException: Various HTTP errors for invalid requests
    """
    try:
        # Import config here to avoid circular imports
        from openchatbi import config

        # Get report directory from config
        report_dir = config.get().report_directory
        file_path = Path(report_dir) / filename

        # Check if file exists and is within the report directory
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Report file not found")

        if not file_path.is_file():
            raise HTTPException(status_code=400, detail="Invalid file path")

        # Ensure the file is within the report directory (security check)
        try:
            file_path.resolve().relative_to(Path(report_dir).resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied") from None

        # Determine media type based on file extension
        media_type_map = {
            ".md": "text/markdown",
            ".csv": "text/csv",
            ".txt": "text/plain",
            ".json": "application/json",
            ".html": "text/html",
            ".xml": "application/xml",
        }

        file_extension = file_path.suffix.lower()
        media_type = media_type_map.get(file_extension, "application/octet-stream")

        return FileResponse(path=str(file_path), media_type=media_type, filename=filename)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download report: {str(e)}") from e


def _create_chroma_from_texts(
    texts: list[str],
    embedding,
    collection_name: str,
    metadatas,
    collection_metadata: dict,
    chroma_dir: str,
):
    """Helper function to create Chroma client from texts."""
    return Chroma.from_texts(
        texts,
        embedding,
        metadatas=metadatas,
        collection_name=collection_name,
        collection_metadata=collection_metadata,
        persist_directory=chroma_dir,
    )


def create_vector_db(
    texts: list[str],
    embedding=None,
    collection_name: str = "langchain",
    metadatas=None,
    collection_metadata: dict = None,
) -> VectorStore:
    """Create or reuse a Chroma vector database.

    Args:
        texts (List[str]): Text documents to index.
        embedding: Embedding function to use.
        collection_name (str): Name of the collection.
        metadatas: Metadata for each document.
        collection_metadata (dict): Collection-level metadata.

    Returns:
        Chroma: Vector database instance.
    """
    chroma_dir = "./.chroma_db"
    client = Chroma(
        collection_name,
        persist_directory=chroma_dir,
        embedding_function=embedding,
        collection_metadata=collection_metadata,
    )
    try:
        # Try to get documents to check if collection exists and has content
        existing_docs = client.get()
        if not existing_docs["documents"]:
            print(f"Init new client from text for {collection_name}...")
            client = _create_chroma_from_texts(
                texts, embedding, collection_name, metadatas, collection_metadata, chroma_dir
            )
        else:
            print(f"Re-use collection for {collection_name}")
    except Exception:
        # If collection doesn't exist or any error, create new one
        print(f"Init new client from text for {collection_name}...")
        client = _create_chroma_from_texts(
            texts, embedding, collection_name, metadatas, collection_metadata, chroma_dir
        )
    return client
