"""Serialization of the Phase-3 metrics schema (JSON + CSV)."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from .stats import BUCKET_NAMES, EvalError, PRIMARY_TAU_M, TAUS_M


def _row_flat(row: dict) -> dict:
    flat = {
        "schema_version": 1,
        "frame": row["frame"],
        "camera": row["camera"],
        "bucket": row["bucket"],
        "n_raw": row["cell"]["n_raw"],
        "n_valid": row["cell"]["n_valid"],
        "n_hits": row["cell"]["n_hits"],
        "n_misses": row["cell"]["n_misses"],
        "coverage": row["cell"]["coverage"],
        "range": row["cell"]["range"],
        "miss": row["cell"]["miss"],
        "prf": row["prf"],
        "cd": row["cd"],
    }
    return flat


def build_metrics_json(report: dict, inputs: dict, mode: str, generated_at_utc: str) -> dict:
    rows = [_row_flat(row) for row in report["rows"]]
    return {
        "schema_version": 1,
        "format": {
            "distance": "true_euclidean",
            "frame": "W",
            "null_policy": "json_null_empty_csv",
            "units": "m",
        },
        "generated_at_utc": generated_at_utc,
        "mode": mode,
        "inputs": inputs,
        "protocol": {
            "schema_version": 1,
            "protocol_version": 1,
            "taus_m": list(TAUS_M),
            "primary_tau_m": PRIMARY_TAU_M,
            "bucket_names": list(BUCKET_NAMES),
            "primary_buckets": [[0.0, 20.0], [20.0, 40.0], [40.0, 50.0]],
            "secondary_bucket": [50.0, 60.0],
            "r_min_m": 1.5,
            "t_min_m": 1.0e-4,
            "t_max_m": 60.0,
            "percentile_rule": "nearest_rank",
            "macro_std": "sample_n_minus_1",
            "null_policy": "json_null_empty_csv",
            "cd_convention": "symmetric_half_average",
            "prf_primary": "nearest_neighbour_within_tau",
            "prf_secondary": "one_to_one_greedy",
            "greedy_candidate_cap": 4096,
        },
        "rows": rows,
        "aggregates": {
            "micro_primary": report["micro_primary"],
            "micro_secondary": report["micro_secondary"],
            "per_camera": report["per_camera"],
            "macro": report["macro"],
        },
        "caveats": [],
    }


def write_metrics_json(path: str | Path, document: dict) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def _value(value) -> str:
    return "" if value is None else repr(value) if isinstance(value, float) else str(value)


def write_metrics_csv(path: str | Path, document: dict) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "schema_version",
        "frame",
        "camera",
        "bucket",
        "n_raw",
        "n_valid",
        "n_hits",
        "n_misses",
        "coverage",
        "range_mae_m",
        "range_rmse_m",
        "range_medae_m",
        "range_bias_m",
        "range_p90_m",
        "range_p95_m",
        "miss_rate",
        "ray_recall",
    ]
    for tau in TAUS_M:
        for kind in ("nn", "greedy"):
            for metric in ("precision", "recall", "f_score"):
                fields.append(f"{metric}_{kind}_tau_{tau:.2f}")
    fields += ["cd_l1_m", "cd_l2_m2", "cd_accuracy_l1_m", "cd_completeness_l1_m", "cd_accuracy_l2_m2", "cd_completeness_l2_m2"]

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(fields)
    for row in document["rows"]:
        line = [
            row["schema_version"],
            row["frame"],
            row["camera"],
            row["bucket"],
            row["n_raw"],
            row["n_valid"],
            row["n_hits"],
            row["n_misses"],
            _value(row["coverage"]),
            _value(row["range"]["mae"]),
            _value(row["range"]["rmse"]),
            _value(row["range"]["medae"]),
            _value(row["range"]["bias"]),
            _value(row["range"]["p90"]),
            _value(row["range"]["p95"]),
            _value(row["miss"]["miss_rate"]),
            _value(row["miss"]["ray_recall"]),
        ]
        for tau in TAUS_M:
            key = f"{tau:.2f}"
            for kind in ("nearest", "greedy"):
                entry = row["prf"][key][kind]
                for metric in ("precision", "recall", "f_score"):
                    line.append(_value(entry[metric]))
        line += [
            _value(row["cd"]["cd_l1"]),
            _value(row["cd"]["cd_l2"]),
            _value(row["cd"]["accuracy_l1"]),
            _value(row["cd"]["completeness_l1"]),
            _value(row["cd"]["accuracy_l2"]),
            _value(row["cd"]["completeness_l2"]),
        ]
        writer.writerow(line)
    target.write_text(buffer.getvalue(), encoding="utf-8")
    return target


def validate_metrics_json(document: dict) -> None:
    if document.get("schema_version") != 1:
        raise EvalError("metrics.json schema_version must be 1")
    if not isinstance(document.get("rows"), list):
        raise EvalError("metrics.json is missing rows[]")
    aggregates = document.get("aggregates")
    if not isinstance(aggregates, dict):
        raise EvalError("metrics.json is missing aggregates")
    for key in ("micro_primary", "micro_secondary", "per_camera", "macro"):
        if key not in aggregates:
            raise EvalError(f"metrics.json aggregates is missing {key}")
    for row in document["rows"]:
        for key in ("frame", "camera", "bucket", "n_raw", "n_valid", "coverage"):
            if key not in row:
                raise EvalError(f"metrics.json row is missing {key}")


def count_csv_rows(path: str | Path) -> int:
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise EvalError(f"{path}: metrics CSV is empty")
    return len(lines) - 1


def validate_metrics_csv(path: str | Path, document: dict) -> None:
    rows = count_csv_rows(path)
    if rows != len(document["rows"]):
        raise EvalError(f"metrics CSV rows {rows} != metrics.json rows {len(document['rows'])}")
