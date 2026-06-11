"""Scoring helpers shared by S3 SQL-pattern retrieval and langmem long-term rerank."""

import math
from datetime import UTC, datetime
from typing import Any


def _parse_iso(ts: str) -> datetime | None:
    """Parse an ISO-8601 timestamp, returning None on any failure."""
    try:
        dt = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def decay_factor(last_used_iso: str, half_life_days: float, now: datetime | None = None) -> float:
    """Exponential recency decay: exp(-ln2 * age_days / half_life_days).

    Returns 1.0 for unparseable timestamps or non-positive half_life (no decay).
    """
    if half_life_days <= 0:
        return 1.0
    last_used = _parse_iso(last_used_iso)
    if last_used is None:
        return 1.0
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    age_days = max(0.0, (now - last_used).total_seconds() / 86400.0)
    return math.exp(-math.log(2) * age_days / half_life_days)


def composite_score(
    similarity: float,
    importance: float,
    last_used_iso: str,
    use_count: int,
    cfg: Any,
) -> float:
    """Blend similarity x importance x recency-decay, lightly boosted by use_count.

    Args:
        similarity: Retrieval similarity in [0, 1].
        importance: Provenance weight (golden > auto).
        last_used_iso: ISO timestamp of last access.
        use_count: Number of times this memory has been used.
        cfg: A MemoryConfig (reads importance_decay_half_life_days).

    Returns:
        float: A non-negative composite ranking score.
    """
    half_life = getattr(cfg, "importance_decay_half_life_days", 30.0)
    decay = decay_factor(last_used_iso, half_life)
    usage_boost = 1.0 + math.log1p(max(0, int(use_count or 0))) * 0.1
    return float(similarity) * float(importance) * decay * usage_boost


def bump_on_access(meta: dict) -> dict:
    """Return a copy of `meta` with use_count+=1 and last_used=now (UTC ISO)."""
    out = dict(meta)
    out["use_count"] = int(out.get("use_count", 0) or 0) + 1
    out["last_used"] = datetime.now(UTC).isoformat()
    return out


def prune_stale(store: Any, namespace: str, cfg: Any) -> int:
    """Remove items whose composite recency-decay drops below cfg.min_retrieval_score.

    Iterates `store.search((namespace,))` items, computing a recency-only decay
    score (similarity treated as 1.0) and deleting those below the floor. Returns
    the number of pruned items. Best-effort: store errors are swallowed.

    Args:
        store: A langgraph BaseStore-like object with search()/delete().
        namespace: The top-level namespace segment to prune.
        cfg: A MemoryConfig (reads min_retrieval_score, importance_decay_half_life_days).

    Returns:
        int: Count of pruned items.
    """
    pruned = 0
    floor = getattr(cfg, "min_retrieval_score", 0.2)
    half_life = getattr(cfg, "importance_decay_half_life_days", 30.0)
    try:
        items = store.search((namespace,))
    except Exception:
        return 0
    for item in items:
        value = getattr(item, "value", None) or {}
        last_used = value.get("last_used", "")
        importance = float(value.get("importance", 1.0) or 1.0)
        score = importance * decay_factor(last_used, half_life)
        if score < floor:
            try:
                store.delete(getattr(item, "namespace", (namespace,)), getattr(item, "key", ""))
                pruned += 1
            except Exception:
                continue
    return pruned
