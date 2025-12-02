"""Unit tests for SimpleStore."""

import pytest

from openchatbi.utils import SimpleStore


class TestSimpleStore:
    """Test suite for SimpleStore class."""

    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing."""
        return [
            "Python is a programming language",
            "Machine learning is a subset of AI",
            "Deep learning uses neural networks",
            "Natural language processing works with text",
        ]

    @pytest.fixture
    def sample_metadatas(self):
        """Sample metadata for testing."""
        return [
            {"category": "programming"},
            {"category": "ai"},
            {"category": "ai"},
            {"category": "nlp"},
        ]

    @pytest.fixture
    def simple_store(self, sample_texts):
        """Create a SimpleStore instance for testing."""
        return SimpleStore(sample_texts)

    def test_initialization_basic(self, sample_texts):
        """Test basic initialization."""
        store = SimpleStore(sample_texts)

        assert len(store.texts) == len(sample_texts)
        assert store.texts == sample_texts
        assert len(store.documents) == len(sample_texts)
        assert store.bm25 is not None

    def test_initialization_with_metadata_and_ids(self, sample_texts, sample_metadatas):
        """Test initialization with metadata and custom IDs."""
        ids = ["id1", "id2", "id3", "id4"]
        store = SimpleStore(sample_texts, sample_metadatas, ids)

        assert store.texts == sample_texts
        assert store.metadatas == sample_metadatas
        assert store.ids == ids
        # Check documents are created correctly
        for doc, text, meta, doc_id in zip(store.documents, sample_texts, sample_metadatas, ids):
            assert doc.page_content == text
            assert doc.metadata == meta
            assert doc.id == doc_id

    def test_similarity_search(self, simple_store):
        """Test similarity search functionality."""
        query = "programming"
        results = simple_store.similarity_search(query, k=2)

        assert len(results) == 2
        assert "Python" in results[0].page_content

        # Test k parameter
        results = simple_store.similarity_search(query, k=10)
        assert len(results) == 4  # Should return all documents

    def test_similarity_search_with_score(self, simple_store):
        """Test similarity search with scores."""
        query = "programming"
        results = simple_store.similarity_search_with_score(query, k=2)

        assert len(results) == 2
        for doc, score in results:
            assert hasattr(doc, "page_content")
            assert isinstance(score, (int, float))
            assert score >= 0

        # Scores should be in descending order
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_store(self):
        """Test empty store operations."""
        store = SimpleStore([])

        assert store.bm25 is None
        assert store.similarity_search("test", k=5) == []
        assert store.similarity_search_with_score("test", k=5) == []

    def test_add_texts(self, simple_store):
        """Test adding texts with and without metadata."""
        initial_count = len(simple_store.texts)
        new_texts = ["Data science is important", "Statistics is fundamental"]
        new_metadatas = [{"type": "test"}, {"type": "example"}]

        # Add with metadata and custom IDs
        custom_ids = ["custom_1", "custom_2"]
        returned_ids = simple_store.add_texts(new_texts, metadatas=new_metadatas, ids=custom_ids)

        assert returned_ids == custom_ids
        assert len(simple_store.texts) == initial_count + len(new_texts)
        assert all(text in simple_store.texts for text in new_texts)

        # Check metadata was added correctly
        added_docs = [doc for doc in simple_store.documents if doc.id in custom_ids]
        assert len(added_docs) == 2
        assert added_docs[0].metadata == {"type": "test"}

        # Verify BM25 index is updated
        results = simple_store.similarity_search("data science", k=1)
        assert "data" in results[0].page_content.lower() or "science" in results[0].page_content.lower()

    def test_delete(self):
        """Test deleting documents."""
        texts = ["Text A", "Text B", "Text C", "Text D"]
        ids = ["id1", "id2", "id3", "id4"]
        store = SimpleStore(texts, ids=ids)

        # Delete specific IDs
        result = store.delete(["id2", "id3"])
        assert result is True
        assert len(store.texts) == 2
        assert store.texts == ["Text A", "Text D"]
        assert store.ids == ["id1", "id4"]

        # Delete non-existent IDs
        result = store.delete(["nonexistent"])
        assert result is False

        # Delete with None
        result = store.delete(None)
        assert result is False

        # Delete all remaining documents
        result = store.delete(["id1", "id4"])
        assert result is True
        assert len(store.texts) == 0
        assert store.bm25 is None

    def test_get_by_ids(self, sample_texts):
        """Test retrieving documents by IDs."""
        ids = ["id1", "id2", "id3", "id4"]
        store = SimpleStore(sample_texts, ids=ids)

        # Get existing IDs
        docs = store.get_by_ids(["id1", "id3"])
        assert len(docs) == 2
        assert docs[0].id == "id1"
        assert docs[0].page_content == sample_texts[0]

        # Get non-existent IDs
        docs = store.get_by_ids(["nonexistent"])
        assert len(docs) == 0

        # Mixed existent and non-existent
        docs = store.get_by_ids(["id1", "nonexistent", "id3"])
        assert len(docs) == 2

    def test_from_texts(self, sample_texts, sample_metadatas):
        """Test creating store using from_texts class method."""
        ids = ["id1", "id2", "id3", "id4"]
        store = SimpleStore.from_texts(sample_texts, embedding=None, metadatas=sample_metadatas, ids=ids)

        assert isinstance(store, SimpleStore)
        assert store.texts == sample_texts
        assert store.metadatas == sample_metadatas
        assert store.ids == ids

    def test_as_retriever(self, simple_store):
        """Test creating a retriever from the store."""
        retriever = simple_store.as_retriever(search_kwargs={"k": 2})

        results = retriever.invoke("programming")
        assert len(results) <= 2
        assert all(hasattr(doc, "page_content") for doc in results)

    def test_chinese_and_mixed_language(self):
        """Test search with Chinese and mixed language texts."""
        from openchatbi.text_segmenter import _jieba_available

        mixed_texts = [
            "Python programming language",
            "机器学习很重要",
            "Deep learning neural networks",
            "数据科学分析",
        ]
        store = SimpleStore(mixed_texts)

        # Search in English
        en_results = store.similarity_search("programming", k=1)
        assert "Python" in en_results[0].page_content

        # Search in Chinese - result depends on jieba availability
        cn_results = store.similarity_search("机器学习", k=2)
        assert len(cn_results) > 0

        # If jieba is available, expect better Chinese matching
        if _jieba_available:
            assert "机器学习" in cn_results[0].page_content
        else:
            # Without jieba, just verify results are returned
            # (Chinese text may not be perfectly tokenized)
            assert any("机器学习" in doc.page_content for doc in cn_results) or any(
                "数据科学" in doc.page_content for doc in cn_results
            )
