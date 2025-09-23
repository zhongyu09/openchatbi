"""Schema and column retrieval functionality for finding relevant database structures."""

import os
import re

import Levenshtein
import jieba

from openchatbi.catalog.entry import catalog_store
from openchatbi.catalog.retrival_helper import build_column_tables_mapping, build_columns_retriever
from openchatbi.utils import log

# Skip build during documentation build
if not os.environ.get("SPHINX_BUILD"):
    bm25, vector_db, columns, col_dict = build_columns_retriever(catalog_store)
    column_tables_mapping = build_column_tables_mapping(catalog_store)
else:
    bm25, vector_db, columns, col_dict = None, None, [], {}
    column_tables_mapping = {}


def column_retrieval(query, db, k=10, threshold=0.5, filter=None):
    """Retrieves relevant columns based on a similarity search.

    Args:
        query (str): The query string to search for.
        db: The vector database to search in.
        k (int, optional): The number of top results to return. Defaults to 10.
        threshold (float, optional): The similarity threshold for filtering results. Defaults to 0.5.
        filter (dict, optional): A filter to apply to the search. Defaults to None.

    Returns:
        list: List of relevant column names.
    """
    log(f"Get the top relevant columns for query: {query}")
    similar_column_key_scores = db.similarity_search_with_score(query, k=k, filter=filter)
    # log(f"similar_column_key_scores: {similar_column_key_scores}")
    column_names = [key.metadata["column_name"] for (key, score) in similar_column_key_scores if score < threshold]
    log(f"Filtered relevant columns: {column_names}")
    return column_names


def merge_list(list1, list2):
    return list(set(list1 + list2))


def edit_distance_score(key1, key2):
    """Calculate normalized edit distance score between two strings.

    Returns:
        float: Score between 0 (identical) and 1 (completely different).
    """
    dist = Levenshtein.distance(key1, key2)
    max_len = max(len(key1), len(key2))
    return dist / max_len if max_len > 0 else 1


def edit_distance_search(keywords_list, top_k=10, threshold=0.5):
    """Searches for columns using edit distance similarity.

    Args:
        keywords_list (list): List of keywords to search for.
        top_k (int, optional): The number of top results to return per keyword. Defaults to 10.
        threshold (float, optional): The maximum edit distance score to consider. Defaults to 0.5.

    Returns:
        list: List of relevant column names.
    """
    keys = set([re.sub(r"(_id|_name| id| name)$", "", key.lower()) for key in keywords_list])
    column_similarity_score = set()
    for key in keys:
        key_column_similarity_score = {}
        for column_name, row in col_dict.items():
            column_name_score = edit_distance_score(
                key, re.sub(r"(_id|_name| id| name)$", "", row.get("column_name", ""))
            )
            display_score = edit_distance_score(
                key, re.sub(r"(_id|_name| id| name)$", "", row.get("display_name", "").lower())
            )
            if column_name_score < threshold or display_score < threshold:
                key_column_similarity_score[column_name] = min(column_name_score, display_score)
        key_top_column = [
            key for key, _ in sorted(key_column_similarity_score.items(), key=lambda x: x[1], reverse=True)[:top_k]
        ]
        column_similarity_score.update(key_top_column)
    return list(column_similarity_score)


def bm25_search(query_list, top_k=5, score_threshold=0.5):
    """Performs a BM25 search on columns based on the query.

    Args:
        query_list (list): List of query terms.
        top_k (int, optional): The number of top results to return. Defaults to 5.
        score_threshold (float, optional): The minimum BM25 score to consider. Defaults to 0.5.

    Returns:
        list: List of relevant column names.
    """
    query_tokens = [token for token in jieba.cut_for_search(" ".join(query_list)) if token not in ("_", " ")]
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in ranked[:top_k]:
        if score_threshold and score < score_threshold:
            continue
        results.append(columns[idx]["column_name"])
    return results


def get_relevant_columns(keywords_list, dimensions, metrics):
    """Get the most relevant columns for given keywords, dimensions, and metrics.

    Uses multiple retrieval methods (BM25, edit distance, vector similarity)
    to find the best matching columns.

    Args:
        keywords_list (list): General keywords to search for.
        dimensions (list): Dimension-specific keywords.
        metrics (list): Metric-specific keywords.

    Returns:
        list: Relevant column names.
    """
    # 1. BM25 search for general keywords
    total_results = bm25_search(keywords_list, top_k=len(keywords_list) * 4)

    # 2. Edit distance search for exact matches
    keyword_len = len(keywords_list + dimensions + metrics)
    ed_results = edit_distance_search(keywords_list + dimensions + metrics, top_k=keyword_len, threshold=0.3)
    total_results = merge_list(total_results, ed_results)

    # 3. Vector similarity search for dimensions
    if dimensions:
        d_results = column_retrieval(" ".join(dimensions), vector_db, k=10, filter={"category": "dimension"})
        total_results = merge_list(total_results, d_results)

    # 4. Vector similarity search for metrics
    if metrics:
        m_results = column_retrieval(" ".join(metrics), vector_db, k=10, threshold=0.55, filter={"category": "metric"})
        total_results = merge_list(total_results, m_results)

    log(f"Relevant columns: {total_results}")
    return total_results
