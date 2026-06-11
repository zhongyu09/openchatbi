"""Tests for the S3 runtime-mutable learned SQL store (LearnedSQLStore)."""

import threading

import pytest

from openchatbi.text2sql.text2sql_utils import LearnedSQLStore, _init_sql_example_retriever
from openchatbi.utils import SimpleStore


def test_init_sql_example_retriever_returns_three_tuple(mock_catalog_store, monkeypatch):
    # Force the no-embedding (SimpleStore/BM25) path.
    monkeypatch.setattr(
        "openchatbi.text2sql.text2sql_utils.get_embedding_model", lambda: None
    )
    result = _init_sql_example_retriever(mock_catalog_store, vector_db_path=None)
    assert isinstance(result, tuple)
    assert len(result) == 3
    retriever, example_dict, vector_db = result
    assert isinstance(example_dict, dict)
    assert isinstance(vector_db, SimpleStore)
    assert hasattr(retriever, "invoke")


def _make_store():
    vector_db = SimpleStore(
        ["How many users are there?"],
        metadatas=[{"sql": "SELECT COUNT(*) FROM users;", "tables": ["users"], "source": "static"}],
    )
    example_dict = {"How many users are there?": ("SELECT COUNT(*) FROM users;", ["users"])}
    return LearnedSQLStore(vector_db, example_dict, threading.Lock())


def test_add_then_retrieve_round_trip_simplestore():
    store = _make_store()
    store.add_golden_sql(
        "What is the average age of users?",
        "SELECT AVG(age) FROM users;",
        ["users"],
    )
    # dict mirror updated in place
    assert store.example_dict["What is the average age of users?"] == (
        "SELECT AVG(age) FROM users;",
        ["users"],
    )
    results = store.retrieve("average age users", k=5)
    questions = [q for q, _, _ in results]
    assert "What is the average age of users?" in questions
    sql = dict((q, s) for q, s, _ in results)["What is the average age of users?"]
    assert sql == "SELECT AVG(age) FROM users;"


def test_add_stamps_namespace_and_source_metadata():
    store = _make_store()
    store.add("foo bar baz", "SELECT 1", ["t"], source="auto", importance=0.5, namespace="tenant_a")
    meta = next(d.metadata for d in store.vector_db.documents if d.page_content == "foo bar baz")
    assert meta["source"] == "auto"
    assert meta["namespace"] == "tenant_a"
    assert meta["importance"] == 0.5
    assert meta["use_count"] == 0
    assert "last_used" in meta


def test_concurrent_add_is_threadsafe():
    store = _make_store()

    def worker(i):
        store.add(f"question number {i}", f"SELECT {i}", ["t"], source="auto")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 1 seed + 20 concurrent adds, no lost writes and BM25 index consistent.
    assert len(store.vector_db.texts) == 21
    assert len(store.vector_db.documents) == 21
    assert len(store.vector_db.tokenized_corpus) == 21
    results = store.retrieve("question number 7", k=21)
    assert "question number 7" in [q for q, _, _ in results]


def test_retrieve_score_fn_reranks():
    store = _make_store()
    store.add("alpha query about users", "SELECT a", ["users"], source="golden", importance=2.0)
    store.add("beta query about users", "SELECT b", ["users"], source="auto", importance=0.1)

    # score_fn prefers higher importance regardless of MMR order.
    def score_fn(meta, base_rank):
        return meta.get("importance", 0.0)

    results = store.retrieve("users query", k=3, score_fn=score_fn)
    assert results[0][0] == "alpha query about users"


def test_retrieve_bumps_use_count_and_last_used():
    """Retrieval must feed back into composite scoring: use_count++ and last_used refresh."""
    store = _make_store()
    seed_meta = store.vector_db.documents[0].metadata
    assert "use_count" not in seed_meta  # static seed has no provenance yet

    results = store.retrieve("How many users are there?", k=1)
    assert results
    assert seed_meta["use_count"] == 1
    assert seed_meta["last_used"]

    store.retrieve("How many users are there?", k=1)
    assert seed_meta["use_count"] == 2


def test_retrieve_touch_only_affects_returned_top_k():
    """Patterns outside the returned top-k are not counted as used."""
    store = _make_store()
    store.add("alpha query about users", "SELECT a", ["users"], source="golden", importance=2.0)

    def score_fn(meta, base_rank):
        return meta.get("importance", 0.0)

    results = store.retrieve("users", k=1, score_fn=score_fn)
    assert len(results) == 1
    touched = [d for d in store.vector_db.documents if int(d.metadata.get("use_count", 0) or 0) > 0]
    assert len(touched) == 1
    assert touched[0].page_content == results[0][0]


def test_get_learned_sql_store_accessor_exists():
    """Convention #4: get_learned_sql_store() accessor must exist in data.py."""
    from openchatbi.text2sql.data import get_learned_sql_store

    # Should be callable and return None when no catalog is configured
    result = get_learned_sql_store()
    assert result is None or isinstance(result, LearnedSQLStore)
