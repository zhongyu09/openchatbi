"""Utility functions for OpenChatBI."""

import json
import sys
import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.responses import FileResponse
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, RemoveMessage, ToolMessage
from langchain_core.vectorstores import VectorStore
from rank_bm25 import BM25Okapi
from regex import regex

from openchatbi.graph_state import AgentState
from openchatbi.text_segmenter import _segmenter


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
    if not isinstance(chunk, AIMessageChunk) or not hasattr(chunk, "content") or not chunk.content:
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
    # fallback to Simple vector store using BM25 if no embedding model configured
    if not embedding:
        return SimpleStore(texts, metadatas)

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


def recover_incomplete_tool_calls(state: AgentState) -> list:
    """Recover from incomplete tool calls by creating message operations to insert ToolMessages correctly.

    When the graph execution is interrupted (e.g., by kill or app restart)
    during tool execution, the state can end up with AIMessage containing
    tool_calls but no corresponding ToolMessage responses. This function
    detects such cases and creates the necessary message operations to insert
    failure ToolMessages in the correct position (right after the AIMessage).

    Args:
        state (AgentState): The current graph state containing messages.

    Returns:
        list: Message operations to insert recovery ToolMessages, or empty list if no recovery needed.
    """
    messages = state.get("messages", [])
    if not messages:
        return []

    # Find the last AIMessage with tool_calls
    last_ai_message = None
    last_ai_index = -1

    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage) and messages[i].tool_calls:
            last_ai_message = messages[i]
            last_ai_index = i
            break

    if not last_ai_message:
        return []

    # Check if there are any ToolMessages after this AIMessage
    tool_call_ids = {call["id"] for call in last_ai_message.tool_calls}
    handled_tool_call_ids = set()

    # Look for ToolMessages that respond to these tool calls
    for msg in messages[last_ai_index + 1 :]:
        if isinstance(msg, ToolMessage) and msg.tool_call_id in tool_call_ids:
            handled_tool_call_ids.add(msg.tool_call_id)

    # Find unhandled tool calls
    unhandled_tool_call_ids = tool_call_ids - handled_tool_call_ids

    if not unhandled_tool_call_ids:
        return []  # All tool calls have responses

    # Create failure ToolMessages for unhandled tool calls
    recovery_messages = []
    for tool_call in last_ai_message.tool_calls:
        if tool_call["id"] in unhandled_tool_call_ids:
            failure_msg = ToolMessage(
                content=f"Tool `{tool_call['name']}` execution was interrupted due to system restart or process termination. Please retry the operation.",
                tool_call_id=tool_call["id"],
            )
            recovery_messages.append(failure_msg)

    # Build operations to insert recovery messages in correct position
    operations = []
    messages_after_ai = messages[last_ai_index + 1 :]

    # Collect IDs that will be removed
    removed_ids = set()

    # If there are messages after the AIMessage, we need to remove them first
    if messages_after_ai:
        for msg in messages_after_ai:
            operations.append(RemoveMessage(id=msg.id))
            removed_ids.add(msg.id)

    # Add recovery messages (they will be inserted right after the AIMessage)
    operations.extend(recovery_messages)

    # Re-add the messages that were after the AIMessage (if any)
    # CRITICAL: Must regenerate Message ids if matches a RemoveMessage to prevent RemoveMessage from being cancelled
    if messages_after_ai:
        for msg in messages_after_ai:
            # Only regenerate ID if this message's ID was removed
            if msg.id in removed_ids:
                # Create a copy with new ID to prevent the RemoveMessage from being discarded
                new_msg = msg.model_copy(update={"id": str(uuid.uuid4())})
                operations.append(new_msg)
            else:
                # Keep original message as-is if ID wasn't removed
                operations.append(msg)

    log(f"Recovered {len(recovery_messages)} incomplete tool calls")
    return operations


class SimpleStore(VectorStore):
    """Simple vector store using BM25 for text retrieval without embeddings."""

    def __init__(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ):
        """Initialize SimpleStore with texts.

        Args:
            texts: List of text documents to store.
            metadatas: Optional list of metadata dicts for each document.
            ids: Optional list of IDs for each document.
        """
        self.texts = texts
        self.metadatas = metadatas or [{} for _ in texts]
        self.ids = ids or [str(uuid.uuid4()) for _ in texts]

        # Create Document objects
        self.documents = [
            Document(id=doc_id, page_content=text, metadata=meta)
            for doc_id, text, meta in zip(self.ids, self.texts, self.metadatas)
        ]

        # Tokenize texts and create BM25 index
        self.tokenized_corpus = [self._tokenize(text) for text in texts]
        # BM25Okapi doesn't support empty corpus, so set to None if empty
        self.bm25 = BM25Okapi(self.tokenized_corpus) if texts else None

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25 indexing using TextSegmenter.

        Args:
            text: Text to tokenize.

        Returns:
            List of tokens.
        """
        return _segmenter.cut(text)

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        """Search for documents similar to the query using BM25.

        Args:
            query: Query text.
            k: Number of documents to return.
            **kwargs: Additional arguments (unused).

        Returns:
            List of most similar Document objects.
        """
        if not self.texts:
            return []

        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_k = min(k, len(scores))
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        # Return corresponding documents
        return [self.documents[i] for i in top_k_indices]

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs: Any) -> list[tuple[Document, float]]:
        """Search for documents similar to the query with BM25 scores.

        Args:
            query: Query text.
            k: Number of documents to return.
            **kwargs: Additional arguments (unused).

        Returns:
            List of (Document, score) tuples.
        """
        if not self.texts:
            return []

        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k items
        top_k = min(k, len(scores))
        top_k_items = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        # Return (Document, score) tuples
        return [(self.documents[i], score) for i, score in top_k_items]

    def _select_relevance_score_fn(self):
        """Return relevance score function for BM25.

        BM25 scores are already relevance scores, so return identity function.
        """
        return lambda score: score

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add texts to the store.

        Args:
            texts: Texts to add.
            metadatas: Optional metadata for each text.
            ids: Optional IDs for each text.
            **kwargs: Additional arguments (unused).

        Returns:
            List of IDs of added texts.
        """

        if metadatas is None:
            metadatas = [{} for _ in texts]

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        # Add to existing data
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)

        # Create new Document objects
        new_documents = [
            Document(id=doc_id, page_content=text, metadata=meta) for doc_id, text, meta in zip(ids, texts, metadatas)
        ]
        self.documents.extend(new_documents)

        # Update BM25 index
        new_tokenized = [self._tokenize(text) for text in texts]
        self.tokenized_corpus.extend(new_tokenized)
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        return ids

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Delete documents by IDs.

        Args:
            ids: List of document IDs to delete.
            **kwargs: Additional arguments (unused).

        Returns:
            True if deletion successful, False otherwise.
        """
        if ids is None:
            return False

        # Find indices to delete
        indices_to_delete = [i for i, doc_id in enumerate(self.ids) if doc_id in ids]

        if not indices_to_delete:
            return False

        # Remove items in reverse order to maintain indices
        for idx in sorted(indices_to_delete, reverse=True):
            del self.texts[idx]
            del self.metadatas[idx]
            del self.ids[idx]
            del self.documents[idx]
            del self.tokenized_corpus[idx]

        # Rebuild BM25 index
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

        return True

    def get_by_ids(self, ids: list[str], /) -> list[Document]:
        """Get documents by their IDs.

        Args:
            ids: List of document IDs to retrieve.

        Returns:
            List of Document objects.
        """
        id_to_doc = {doc.id: doc for doc in self.documents}
        return [id_to_doc[doc_id] for doc_id in ids if doc_id in id_to_doc]

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Any = None,  # Unused but required by interface
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> "SimpleStore":
        """Create SimpleStore from texts.

        Args:
            texts: List of texts.
            embedding: Unused (SimpleStore doesn't use embeddings).
            metadatas: Optional metadata for each text.
            ids: Optional IDs for each text.
            **kwargs: Additional arguments (unused).

        Returns:
            SimpleStore instance.
        """
        return cls(texts, metadatas, ids)

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of `Document` objects to return.
            fetch_k: Number of `Document` objects to fetch to pass to MMR algorithm.
            lambda_mult: Number between `0` and `1` that determines the degree
                of diversity among the results with `0` corresponding
                to maximum diversity and `1` to minimum diversity.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of `Document` objects selected by maximal marginal relevance.
        """
        if not self.texts:
            return []

        # Get initial candidates using BM25 similarity search
        candidates = self.similarity_search_with_score(query, k=fetch_k, **kwargs)

        if not candidates:
            return []

        if len(candidates) <= k:
            return [doc for doc, _ in candidates]

        # Normalize BM25 scores to [0, 1] for proper MMR calculation
        scores = [score for _, score in candidates]
        min_score = min(scores) if scores else 0
        max_score = max(scores) if scores else 1
        score_range = max_score - min_score if max_score > min_score else 1

        normalized_candidates = [(doc, (score - min_score) / score_range) for doc, score in candidates]

        # MMR implementation following standard algorithm
        selected = []
        remaining = list(range(len(normalized_candidates)))

        # Select documents iteratively using MMR formula
        while len(selected) < k and remaining:
            best_mmr_score = float("-inf")
            best_idx = -1
            best_remaining_idx = -1

            for i, doc_idx in enumerate(remaining):
                candidate_doc, relevance_score = normalized_candidates[doc_idx]

                # Calculate maximum similarity to already selected documents
                max_similarity = 0.0
                if selected:
                    max_similarity = max(
                        self._calculate_similarity(candidate_doc, normalized_candidates[sel_idx][0])
                        for sel_idx in selected
                    )

                # Standard MMR formula: λ * Sim(q, d) - (1-λ) * max(Sim(d, s)) for s in selected
                mmr_score = lambda_mult * relevance_score - (1 - lambda_mult) * max_similarity

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = doc_idx
                    best_remaining_idx = i

            if best_idx != -1:
                selected.append(best_idx)
                remaining.pop(best_remaining_idx)

        return [normalized_candidates[idx][0] for idx in selected]

    def _calculate_similarity(self, doc1: Document, doc2: Document) -> float:
        """Calculate similarity between two documents using Jaccard similarity.

        Args:
            doc1: First document.
            doc2: Second document.

        Returns:
            Similarity score between 0 and 1 (higher means more similar).
        """
        tokens1 = set(self._tokenize(doc1.page_content))
        tokens2 = set(self._tokenize(doc2.page_content))

        # Calculate Jaccard similarity
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0
