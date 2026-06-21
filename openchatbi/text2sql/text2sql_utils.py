"""Utility functions for text2sql retrieval systems."""

from __future__ import annotations

import threading
from datetime import UTC, datetime

from openchatbi.llm.llm import get_embedding_model
from openchatbi.utils import create_vector_db, log


def _init_sql_example_retriever(catalog, vector_db_path: str | None = None):
    """Initialize SQL example retriever from catalog.

    Args:
        catalog: Catalog store containing SQL examples.
        vector_db_path: Path to the vector database file.

    Returns:
        tuple: (retriever, sql_example_dict, vector_db)
    """
    sql_examples = catalog.get_sql_examples()
    sql_example_dict = {q: (sql, table) for q, sql, table in sql_examples}

    texts = list(sql_example_dict.keys())
    vector_db = create_vector_db(
        texts,
        get_embedding_model(),
        collection_name="text2sql",
        collection_metadata={"hnsw:space": "cosine"},
        chroma_db_path=vector_db_path,
    )
    retriever = vector_db.as_retriever(
        search_type="mmr", search_kwargs={"distance_metric": "cosine", "fetch_k": 30, "k": 10}
    )
    return retriever, sql_example_dict, vector_db


class LearnedSQLStore:
    """Runtime-mutable learned SQL knowledge base.

    Wraps the text2sql vector store so that approved (``source='golden'``) and
    auto-captured (``source='auto'``) examples can be written at runtime and
    retrieved alongside the static catalog examples. Writes are guarded by a
    lock because ``SimpleStore.add_texts`` rebuilds the BM25 index O(N) and is
    not threadsafe; callers are responsible for the durable YAML half of the
    dual-write contract (the in-memory ``add_texts`` here is the volatile half).
    """

    def __init__(self, vector_db, example_dict: dict, lock: threading.Lock | None = None):
        self.vector_db = vector_db
        self.example_dict = example_dict
        self.lock = lock or threading.Lock()

    def add(
        self,
        question: str,
        sql: str,
        tables: list[str],
        *,
        source: str,
        importance: float = 1.0,
        namespace: str = "global",
    ) -> None:
        """Add a learned example to the runtime store (volatile half of dual-write).

        Args:
            question: Natural-language question (the indexed text).
            sql: SQL answer.
            tables: Tables used by the SQL.
            source: Provenance; 'golden' (human-approved) or 'auto' (S2-gated capture).
            importance: Base importance weight used by composite scoring.
            namespace: Tenant/scope tag; 'global' must hold only schema-level patterns.
        """
        now = datetime.now(UTC).isoformat()
        metadata = {
            "sql": sql,
            "tables": tables,
            "source": source,
            "importance": importance,
            "use_count": 0,
            "last_used": now,
            "namespace": namespace,
        }
        with self.lock:
            # Volatile half: BM25 rebuild / Chroma add is O(N) and non-threadsafe.
            self.vector_db.add_texts([question], metadatas=[metadata])
            # Mirror into the dict so _get_relevant_sql_examples_prompt keeps working.
            self.example_dict[question] = (sql, tables)

    def add_golden_sql(self, question: str, sql: str, tables: list[str]) -> None:
        """Alias: add a human-approved golden example with high importance."""
        self.add(question, sql, tables, source="golden", importance=2.0)

    def retrieve(
        self,
        question: str,
        k: int = 10,
        score_fn=None,
    ) -> list[tuple[str, str, list[str]]]:
        """Retrieve top-k learned examples for a question.

        Args:
            question: Query text.
            k: Number of examples to return.
            score_fn: Optional re-ranker ``(metadata, base_rank) -> float`` (e.g.
                composite_score from memory_scoring); higher is better. When None,
                the underlying MMR order is preserved.

        Returns:
            List of (question, sql, tables) tuples.
        """
        docs = self.vector_db.max_marginal_relevance_search(question, k=max(k, 1), fetch_k=30)
        ranked = list(enumerate(docs))
        if score_fn is not None:
            ranked.sort(key=lambda pair: score_fn(pair[1].metadata, pair[0]), reverse=True)
        top_docs = [doc for _, doc in ranked[:k]]
        # Composite scoring weights frequency/recency, so retrieval must feed
        # them back: bump use_count/last_used on the returned patterns.
        self._touch(top_docs)
        results: list[tuple[str, str, list[str]]] = []
        for doc in top_docs:
            q = doc.page_content
            sql = doc.metadata.get("sql")
            if sql is None and q in self.example_dict:
                sql, tables = self.example_dict[q]
            else:
                tables = doc.metadata.get("tables", [])
            results.append((q, sql, tables))
        return results

    def _touch(self, docs: list) -> None:
        """Bump use_count/last_used on retrieved docs (best-effort, never raises).

        In-place metadata mutation covers SimpleStore (search returns the shared
        Document objects); a Chroma-style ``_collection.update`` persists the bump
        for stores that return copies.
        """
        now = datetime.now(UTC).isoformat()
        with self.lock:
            for doc in docs:
                try:
                    doc.metadata["use_count"] = int(doc.metadata.get("use_count", 0) or 0) + 1
                    doc.metadata["last_used"] = now
                    collection = getattr(self.vector_db, "_collection", None)
                    doc_id = getattr(doc, "id", None)
                    if collection is not None and doc_id:
                        collection.update(ids=[doc_id], metadatas=[doc.metadata])
                except Exception as e:
                    log(f"use_count touch failed (ignored): {e}")


def _init_table_selection_example_dict(catalog, vector_db_path: str | None = None):
    """Initialize table selection example retriever from catalog.

    Args:
        catalog: Catalog store containing table selection examples.
        vector_db_path: Path to the vector database file.

    Returns:
        tuple: (retriever, table_selection_example_dict)
    """
    sql_examples = catalog.get_table_selection_examples()
    table_selection_example_dict = dict(sql_examples)

    texts = list(table_selection_example_dict.keys())
    if not texts:
        texts = [""]  # Empty text as fallback

    vector_db = create_vector_db(
        texts,
        get_embedding_model(),
        collection_name="table_selection_example",
        collection_metadata={"hnsw:space": "cosine"},
        chroma_db_path=vector_db_path,
    )
    retriever = vector_db.as_retriever(
        search_type="mmr", search_kwargs={"distance_metric": "cosine", "fetch_k": 30, "k": 10}
    )
    return retriever, table_selection_example_dict
