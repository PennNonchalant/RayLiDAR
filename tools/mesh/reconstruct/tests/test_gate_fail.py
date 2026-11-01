"""G3 fails loudly, writes a complete diagnostic, and has no correction path.

The gate is run on deterministic synthetic artifacts that are deliberately bad
(two disconnected components, a bbox far smaller than the GT extent, and rays
that never hit).  ``evaluate`` must write ``g3_report.json`` and then exit 6.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from synthetic import FRAMES, make_bundle
from test_g3 import write_gt_frame, write_ray_frame, write_mesh
from tools.mesh.reconstruct import g3

REPO_ROOT = Path(__file__).resolve().parents[4]
SMOKE = list(range(4615, 4625))
FORBIDDEN_TOKENS = ("Umeyama", "umeyama", "ICP", "icp", "apply_transform", "scale(")


def _gt_points(count: int = 17_000) -> np.ndarray:
    index = np.arange(count, dtype=np.float64)
    return np.stack(
        [
            25.0 * np.sin(index * 0.001),
            20.0 * np.cos(index * 0.0017),
            10.0 * np.sin(index * 0.0031),
        ],
        axis=1,
    )


def _failing_mesh(path: Path) -> np.ndarray:
    """Two disconnected tetrahedra: 2 components, largest fraction 0.5."""
    base = np.asarray(
        [
            [4600.0, 0.0, -2.0],
            [4601.0, 0.0, -2.0],
            [4600.0, 1.0, -2.0],
            [4600.0, 0.0, -1.0],
        ]
    )
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
    vertices = np.vstack([base, base + np.asarray([20.0, 0.0, 0.0])])
    write_mesh(path, vertices, np.vstack([faces, faces + 4]))
    return vertices


def _failing_fixture(tmp_path: Path) -> tuple[Path, dict, dict, Path]:
    bundle = make_bundle(tmp_path / "bundle")
    artifacts = tmp_path / "artifacts"
    points = _gt_points()
    for frame in FRAMES:
        write_gt_frame(
            artifacts / "02" / "gt" / f"frame_{frame:010d}.bin", frame, points
        )

    for frame in SMOKE:
        directions = np.tile(np.asarray([0.0, 0.0, 1.0]), (150, 1))
        write_ray_frame(
            artifacts / "02" / "rays" / f"frame_{frame:010d}.bin",
            frame,
            directions,
            np.full(150, 3.0),
        )

    mesh_path = tmp_path / "failing.ply"
    vertices = _failing_mesh(mesh_path)
    artifact = {
        "path": str(mesh_path),
        "fmt": "ply",
        "vertex_count": int(vertices.shape[0]),
        "triangle_count": 8,
        "bbox_min_m": [float(value) for value in vertices.min(axis=0)],
        "bbox_max_m": [float(value) for value in vertices.max(axis=0)],
        "sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "frame": "W",
        "units": "m",
    }
    cfg = {
        "schema_version": 1,
        "route": "colmap_openmvs",
        "bundle_root": str(bundle),
        "out_root": str(artifacts / "04"),
        "windows": {"smoke": SMOKE, "full": list(FRAMES)},
        "params": {"cleanup": {"min_component_triangles": 1}},
    }
    return bundle, artifact, cfg, artifacts


def test_evaluate_writes_a_report_and_exits_6(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle, artifact, cfg, _ = _failing_fixture(tmp_path)
    workdir = tmp_path / "work"
    workdir.mkdir()

    with pytest.raises(SystemExit) as info:
        g3.evaluate(
            artifact,
            str(bundle),
            "smoke",
            SMOKE,
            cfg,
            str(workdir),
            {},
        )

    assert info.value.code == g3.GATE_EXIT_CODE == 6
    report = json.loads((workdir / "g3_report.json").read_text(encoding="utf-8"))
    assert report["pass"] is False
    assert report["alignment_applied"] is False
    assert report["evaluation_frames_used_for_tuning"] is False
    assert "components_ok" in report["failures"]
    assert report["metrics"]["components"]["largest_component_fraction"] == pytest.approx(
        0.5
    )
    assert report["gt_frames"] == list(FRAMES)
    assert report["smoke_range_frames"] == SMOKE
    assert report["mesh"]["sha256"] == "a" * 64
    stderr = capsys.readouterr().err
    assert "G3 FAIL components_ok" in stderr


def test_g3_source_contains_no_correction_helper() -> None:
    source = (Path(g3.__file__)).read_text(encoding="utf-8")

    for token in FORBIDDEN_TOKENS:
        assert token not in source, f"forbidden correction helper: {token}"


def test_pipeline_gate_stage_exits_6_on_a_failing_run(tmp_path: Path) -> None:
    bundle, artifact, cfg, _ = _failing_fixture(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    workdir = tmp_path / "runs" / "gate-fail" / "smoke"
    (workdir / "stages").mkdir(parents=True)
    (workdir / "stages" / "ingest.json").write_text(
        json.dumps({"artifact": artifact}, indent=2) + "\n", encoding="utf-8"
    )

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.mesh.reconstruct.pipeline",
            "--config",
            str(config_path),
            "--stage",
            "gate",
            "--window",
            "smoke",
            "--workdir",
            str(workdir),
        ],
        cwd=str(REPO_ROOT),
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert completed.returncode == 6, completed.stderr
    assert "G3 FAIL" in completed.stderr
    report_path = workdir / "g3_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["pass"] is False
    assert report["alignment_applied"] is False
