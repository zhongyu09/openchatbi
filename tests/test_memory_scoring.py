"""Tests for memory scoring (decay + composite ranking)."""

from datetime import datetime, timedelta, timezone

from openchatbi.memory_config import MemoryConfig
from openchatbi.memory_scoring import (
    bump_on_access,
    composite_score,
    decay_factor,
)


def _iso(days_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def test_decay_factor_at_zero_age_is_one():
    now = datetime.now(timezone.utc)
    assert decay_factor(now.isoformat(), half_life_days=30.0, now=now) == 1.0


def test_decay_factor_at_half_life_is_half():
    now = datetime.now(timezone.utc)
    last_used = (now - timedelta(days=30)).isoformat()
    assert abs(decay_factor(last_used, half_life_days=30.0, now=now) - 0.5) < 1e-6


def test_decay_factor_bad_timestamp_falls_back_to_one():
    assert decay_factor("not-a-date", half_life_days=30.0) == 1.0


def test_composite_score_blends_similarity_importance_decay_usecount():
    cfg = MemoryConfig(importance_decay_half_life_days=30.0)
    now = datetime.now(timezone.utc)
    fresh = composite_score(0.8, 1.0, now.isoformat(), 5, cfg)
    stale = composite_score(0.8, 1.0, (now - timedelta(days=90)).isoformat(), 5, cfg)
    # Fresher memory must outrank a stale one with identical similarity/importance.
    assert fresh > stale


def test_composite_score_higher_importance_wins_at_equal_similarity():
    cfg = MemoryConfig()
    iso = datetime.now(timezone.utc).isoformat()
    assert composite_score(0.6, 2.0, iso, 1, cfg) > composite_score(0.6, 1.0, iso, 1, cfg)


def test_bump_on_access_increments_use_count_and_stamps_last_used():
    meta = {"use_count": 2}
    out = bump_on_access(meta)
    assert out["use_count"] == 3
    assert "last_used" in out
    # original dict not mutated in place
    assert meta["use_count"] == 2
