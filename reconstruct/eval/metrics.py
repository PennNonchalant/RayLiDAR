"""Nearest-neighbour, Chamfer, and precision/recall/F-score metrics."""

from __future__ import annotations

import numpy as np

from .stats import EvalError

MAX_GREEDY_CANDIDATES = 4096


class Cloud:
    """A cached ``cKDTree`` over an ``(N, 3)`` point cloud."""

    def __init__(self, points: np.ndarray) -> None:
        points = np.asarray(points, dtype=np.float64)
        if points.size and not np.isfinite(points).all():
            raise EvalError("point cloud contains a non-finite point")
        self.points = points
        self._tree = None

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def tree(self):
        if self._tree is None:
            if len(self) == 0:
                raise EvalError("cannot index an empty point cloud")
            from scipy.spatial import cKDTree

            self._tree = cKDTree(self.points)
        return self._tree

    def nearest(self, queries: np.ndarray) -> np.ndarray:
        queries = np.asarray(queries, dtype=np.float64)
        if len(self) == 0:
            raise EvalError("cannot query an empty point cloud")
        distances, _ = self.tree().query(queries, k=1, workers=-1)
        return np.atleast_1d(distances)

    def radius_many(self, queries: np.ndarray, radius: float) -> list[list[int]]:
        if len(self) == 0:
            return [[] for _ in range(len(queries))]
        return self.tree().query_ball_point(np.asarray(queries, dtype=np.float64), radius, workers=-1)


def predicted_points(samples) -> np.ndarray:
    rows = [sample.origin_w + sample.dir_w * sample.t_pred for sample in samples if np.isfinite(sample.t_pred)]
    if not rows:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def gt_points(samples) -> np.ndarray:
    rows = [sample.origin_w + sample.dir_w * sample.t_gt for sample in samples]
    if not rows:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def compute_chamfer(predicted: np.ndarray, gt: np.ndarray, *, pred_cloud: Cloud | None = None, gt_cloud: Cloud | None = None) -> dict:
    n_pred = int(len(predicted))
    n_gt = int(len(gt))
    result = {
        "cd_l1": None,
        "cd_l2": None,
        "accuracy_l1": None,
        "completeness_l1": None,
        "accuracy_l2": None,
        "completeness_l2": None,
        "n_pred": n_pred,
        "n_gt": n_gt,
    }
    if n_pred == 0 or n_gt == 0:
        return result
    predicted_cloud = pred_cloud or Cloud(predicted)
    gt_cloud = gt_cloud or Cloud(gt)
    accuracy = gt_cloud.nearest(predicted)
    completeness = predicted_cloud.nearest(gt)
    accuracy_l1 = float(accuracy.mean())
    accuracy_l2 = float((accuracy * accuracy).mean())
    completeness_l1 = float(completeness.mean())
    completeness_l2 = float((completeness * completeness).mean())
    result.update(
        {
            "accuracy_l1": accuracy_l1,
            "accuracy_l2": accuracy_l2,
            "completeness_l1": completeness_l1,
            "completeness_l2": completeness_l2,
            "cd_l1": 0.5 * (accuracy_l1 + completeness_l1),
            "cd_l2": 0.5 * (accuracy_l2 + completeness_l2),
        }
    )
    return result


def _f_score(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0


def compute_prf(
    predicted: np.ndarray,
    gt: np.ndarray,
    tau: float,
    one_to_one: bool,
    *,
    pred_cloud: Cloud | None = None,
    gt_cloud: Cloud | None = None,
) -> dict:
    if not np.isfinite(tau) or tau < 0.0:
        raise EvalError("tau must be finite and non-negative")
    n_pred = int(len(predicted))
    n_gt = int(len(gt))
    result = {
        "precision": None,
        "recall": None,
        "f_score": None,
        "tp_pred": 0,
        "tp_gt": 0,
        "n_pred": n_pred,
        "n_gt": n_gt,
        "greedy_truncated": False,
    }
    if one_to_one:
        matched = 0
        truncated = False
        if n_pred and n_gt:
            gt_index = gt_cloud or Cloud(gt)
            neighbourhoods = gt_index.radius_many(predicted, tau)
            candidates: list[tuple[float, int, int]] = []
            for pred_index, indices in enumerate(neighbourhoods):
                if len(indices) > MAX_GREEDY_CANDIDATES:
                    truncated = True
                    indices = indices[:MAX_GREEDY_CANDIDATES]
                if not indices:
                    continue
                query = predicted[pred_index]
                for gt_index_value in indices:
                    distance = float(np.linalg.norm(gt[gt_index_value] - query))
                    candidates.append((distance, pred_index, int(gt_index_value)))
            candidates.sort(key=lambda item: (item[0], item[1], item[2]))
            pred_used = np.zeros(n_pred, dtype=bool)
            gt_used = np.zeros(n_gt, dtype=bool)
            for _distance, pred_index, gt_index_value in candidates:
                if pred_used[pred_index] or gt_used[gt_index_value]:
                    continue
                pred_used[pred_index] = True
                gt_used[gt_index_value] = True
                matched += 1
        precision = (matched / float(n_pred)) if n_pred else None
        recall = (matched / float(n_gt)) if n_gt else None
        result.update(
            {
                "precision": precision,
                "recall": recall,
                "f_score": _f_score(precision, recall),
                "tp_pred": matched,
                "tp_gt": matched,
                "greedy_truncated": truncated,
            }
        )
        return result

    tp_pred = 0
    tp_gt = 0
    if n_pred and n_gt:
        distance_pred = (gt_cloud or Cloud(gt)).nearest(predicted)
        tp_pred = int((distance_pred <= tau).sum())
        distance_gt = (pred_cloud or Cloud(predicted)).nearest(gt)
        tp_gt = int((distance_gt <= tau).sum())
    precision = (tp_pred / float(n_pred)) if n_pred else None
    recall = (tp_gt / float(n_gt)) if n_gt else None
    result.update(
        {
            "precision": precision,
            "recall": recall,
            "f_score": _f_score(precision, recall),
            "tp_pred": int(tp_pred),
            "tp_gt": int(tp_gt),
        }
    )
    return result
