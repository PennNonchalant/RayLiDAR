"""Shared metric statistics (range, miss, cells, percentiles)."""

from __future__ import annotations

import math
from typing import Iterable

TAUS_M = (0.05, 0.20, 0.50)
PRIMARY_TAU_M = 0.20
BUCKET_NAMES = ("0_20", "20_40", "40_50", "50_60")
PRIMARY_BUCKETS = 3


class EvalError(RuntimeError):
    """An evaluation input or artifact invariant failed."""


def bucket_of(t_gt: float) -> int:
    if not math.isfinite(t_gt) or t_gt < 0.0 or t_gt > 60.0:
        raise EvalError(f"range {t_gt} is outside the frozen [0, 60] m domain")
    if t_gt < 20.0:
        return 0
    if t_gt < 40.0:
        return 1
    if t_gt <= 50.0:
        return 2
    return 3


def percentile_nearest_rank(sorted_values: list[float], percentile: float) -> float:
    n = len(sorted_values)
    rank = int(math.ceil(percentile * n))
    rank = max(rank, 1)
    rank = min(rank, n)
    return sorted_values[rank - 1]


def range_from_errors(errors: Iterable[float]) -> dict:
    errors = list(errors)
    n_hits = len(errors)
    if not errors:
        return {"mae": None, "rmse": None, "medae": None, "bias": None, "p90": None, "p95": None, "n_hits": 0}
    absolute = [abs(error) for error in errors]
    count = float(len(errors))
    mae = sum(absolute) / count
    rmse = math.sqrt(sum(error * error for error in errors) / count)
    bias = sum(errors) / count
    absolute_sorted = sorted(absolute)
    middle = len(absolute_sorted) // 2
    if len(absolute_sorted) % 2 == 0:
        medae = 0.5 * (absolute_sorted[middle - 1] + absolute_sorted[middle])
    else:
        medae = absolute_sorted[middle]
    return {
        "mae": mae,
        "rmse": rmse,
        "medae": medae,
        "bias": bias,
        "p90": percentile_nearest_rank(absolute_sorted, 0.90),
        "p95": percentile_nearest_rank(absolute_sorted, 0.95),
        "n_hits": n_hits,
    }


def miss_from_counts(n_total: int, n_hits: int) -> dict:
    n_misses = max(n_total - n_hits, 0)
    if n_total:
        miss_rate = n_misses / float(n_total)
        ray_recall = n_hits / float(n_total)
    else:
        miss_rate = None
        ray_recall = None
    return {
        "miss_rate": miss_rate,
        "ray_recall": ray_recall,
        "n_total": n_total,
        "n_hits": n_hits,
        "n_misses": n_misses,
    }


def make_cell(n_raw: int, n_valid: int, n_hits: int, errors: Iterable[float]) -> dict:
    return {
        "n_raw": n_raw,
        "n_valid": n_valid,
        "n_hits": n_hits,
        "n_misses": max(n_valid - n_hits, 0),
        "coverage": (n_valid / float(n_raw)) if n_raw else None,
        "range": range_from_errors(errors),
        "miss": miss_from_counts(n_valid, n_hits),
    }


def mean_and_sample_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)
