"""Readers for ``gt_rays.bin`` (RLGR) and ``hits.bin`` (RLHT)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .stats import EvalError

GT_RAY_MAGIC = b"RLGR"
GT_RAY_VERSION = 1
GT_RAY_RECORD_SIZE = 96
HIT_MAGIC = b"RLHT"
HIT_VERSION = 1
HIT_RECORD_SIZE = 24
PRIM_ID_UNAVAILABLE = 0xFFFFFFFF


@dataclass
class GtRay:
    ray_id: int
    frame: int
    camera: int
    origin_w: np.ndarray
    dir_w: np.ndarray
    t_gt: float
    t_min: float
    t_max: float


@dataclass
class Hit:
    ray_id: int
    hit: bool
    t_pred: float
    prim_id: int


def read_gt_rays(path: str | Path) -> tuple[list[GtRay], str]:
    import hashlib

    target = Path(path)
    raw = target.read_bytes()
    if len(raw) < 32:
        raise EvalError(f"gt_rays.bin is shorter than its header: {target}")
    if raw[0:4] != GT_RAY_MAGIC:
        raise EvalError(f"gt_rays.bin has a wrong magic: {target}")
    version = int(np.frombuffer(raw[4:6], dtype="<u2")[0])
    record_size = int(np.frombuffer(raw[6:8], dtype="<u2")[0])
    count = int(np.frombuffer(raw[8:12], dtype="<u4")[0])
    if version != GT_RAY_VERSION or record_size != GT_RAY_RECORD_SIZE:
        raise EvalError("gt_rays.bin has an unsupported version or record size")
    if len(raw) != 32 + count * GT_RAY_RECORD_SIZE:
        raise EvalError("gt_rays.bin length does not match its header count")
    rays: list[GtRay] = []
    seen: set[int] = set()
    for index in range(count):
        base = 32 + index * GT_RAY_RECORD_SIZE
        record = raw[base : base + GT_RAY_RECORD_SIZE]
        ray_id = int(np.frombuffer(record[0:8], dtype="<u8")[0])
        if ray_id in seen:
            raise EvalError(f"gt_rays.bin contains a duplicate ray_id {ray_id}")
        seen.add(ray_id)
        ray = GtRay(
            ray_id=ray_id,
            origin_w=np.frombuffer(record[8:32], dtype="<f8").astype(np.float64),
            dir_w=np.frombuffer(record[32:56], dtype="<f8").astype(np.float64),
            t_gt=float(np.frombuffer(record[56:64], dtype="<f8")[0]),
            t_min=float(np.frombuffer(record[64:72], dtype="<f8")[0]),
            t_max=float(np.frombuffer(record[72:80], dtype="<f8")[0]),
            frame=int(np.frombuffer(record[80:84], dtype="<u4")[0]),
            camera=int(np.frombuffer(record[84:86], dtype="<u2")[0]),
        )
        if not np.isfinite(ray.origin_w).all() or not np.isfinite(ray.dir_w).all():
            raise EvalError("gt_rays.bin contains a non-finite ray")
        if abs(np.linalg.norm(ray.dir_w) - 1.0) > 1.0e-6:
            raise EvalError("gt_rays.bin contains a non-unit dir_W")
        rays.append(ray)
    return rays, hashlib.sha256(raw).hexdigest()


def read_hits(path: str | Path) -> tuple[list[Hit], str]:
    import hashlib

    target = Path(path)
    raw = target.read_bytes()
    if len(raw) < 32:
        raise EvalError(f"hits.bin is shorter than its header: {target}")
    if raw[0:4] != HIT_MAGIC:
        raise EvalError(f"hits.bin has a wrong magic: {target}")
    version = int(np.frombuffer(raw[4:6], dtype="<u2")[0])
    record_size = int(np.frombuffer(raw[6:8], dtype="<u2")[0])
    count = int(np.frombuffer(raw[8:12], dtype="<u4")[0])
    if version != HIT_VERSION or record_size != HIT_RECORD_SIZE:
        raise EvalError("hits.bin has an unsupported version or record size")
    if len(raw) != 32 + count * HIT_RECORD_SIZE:
        raise EvalError("hits.bin length does not match its header count")
    hits: list[Hit] = []
    for index in range(count):
        base = 32 + index * HIT_RECORD_SIZE
        record = raw[base : base + HIT_RECORD_SIZE]
        hit_byte = int(record[16])
        if hit_byte not in (0, 1):
            raise EvalError("hits.bin contains an invalid hit value")
        hits.append(
            Hit(
                ray_id=int(np.frombuffer(record[0:8], dtype="<u8")[0]),
                t_pred=float(np.frombuffer(record[8:16], dtype="<f8")[0]),
                hit=hit_byte == 1,
                prim_id=int(np.frombuffer(record[20:24], dtype="<u4")[0]),
            )
        )
    return hits, hashlib.sha256(raw).hexdigest()


def validate_hits_against_rays(rays: list[GtRay], hits: list[Hit]) -> None:
    if len(rays) != len(hits):
        raise EvalError("hits.bin count does not match gt_rays.bin count")
    for ray, hit in zip(rays, hits):
        if ray.ray_id != hit.ray_id:
            raise EvalError("hits.bin ray_id does not match the ordered gt_rays.bin ray_id")
        if hit.hit:
            if not np.isfinite(hit.t_pred) or hit.t_pred <= 0.0:
                raise EvalError("hits.bin contains an invalid hit t_pred")
            if hit.t_pred < ray.t_min or hit.t_pred > ray.t_max:
                raise EvalError("hits.bin hit t_pred is outside [t_min, t_max]")
        else:
            if np.isfinite(hit.t_pred) or hit.prim_id != PRIM_ID_UNAVAILABLE:
                raise EvalError("hits.bin miss is not the +inf / 0xFFFFFFFF sentinel")
