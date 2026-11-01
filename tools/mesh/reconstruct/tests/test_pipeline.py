from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tools.mesh.reconstruct import pipeline
from tools.mesh.reconstruct.config import config_sha256, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CANONICAL = CONFIG_DIR / "canonical_colmap_openmvs.yaml"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def test_stages_table_has_exactly_six_stages() -> None:
    assert set(pipeline.STAGES) == {
        "model",
        "dense",
        "build",
        "ingest",
        "gate",
        "report",
    }
    assert set(pipeline.STAGE_ORDER) == set(pipeline.STAGES)


def test_build_stage_resolves_to_the_route_builder() -> None:
    from tools.mesh.reconstruct import routes

    assert pipeline.load_stage("build") is routes.build


def test_ingest_stage_resolves_to_the_canonical_adapter() -> None:
    from tools.mesh.reconstruct import ingest

    assert pipeline.load_stage("ingest") is ingest.ingest


def test_run_stage_requires_config_sha256(monkeypatch: pytest.MonkeyPatch) -> None:
    def stage(workdir):  # noqa: ANN001
        return {}

    monkeypatch.setattr(pipeline, "load_stage", lambda name: stage)

    with pytest.raises(pipeline.StageContractError, match="config_sha256"):
        pipeline.run_stage("model", {"workdir": "/tmp"})


def test_run_stage_reports_missing_context_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def stage(colmap_result):  # noqa: ANN001
        return {}

    monkeypatch.setattr(pipeline, "load_stage", lambda name: stage)

    with pytest.raises(pipeline.StageContractError, match="colmap_result"):
        pipeline.run_stage(
            "model", {"config_sha256": "a" * 64, "workdir": tmp_path}
        )


def test_run_stage_persists_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def stage(workdir, config_sha256):  # noqa: ANN001
        return {"colmap_result": {"model": "ok"}, "config_sha256": config_sha256}

    monkeypatch.setattr(pipeline, "load_stage", lambda name: stage)
    ctx = {"config_sha256": "b" * 64, "workdir": tmp_path}

    result = pipeline.run_stage("model", ctx)

    assert result["colmap_result"] == {"model": "ok"}
    assert ctx["colmap_result"] == {"model": "ok"}
    persisted = tmp_path / "stages" / "model.json"
    assert persisted.is_file()
    assert json.loads(persisted.read_text())["colmap_result"] == {"model": "ok"}


def test_config_check_prints_hex_digest(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = pipeline.main(
        ["--config", str(CANONICAL), "--stage", "config-check", "--window", "smoke"]
    )

    assert exit_code == 0
    printed = capsys.readouterr().out.strip()
    assert HEX64.match(printed)
    assert printed == config_sha256(load_config(CANONICAL))


def test_unknown_route_exits_non_zero(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = pipeline.main(
        [
            "--config",
            str(CANONICAL),
            "--route",
            "poisson",
            "--stage",
            "config-check",
        ]
    )

    assert exit_code != 0
    assert "RouteUnavailable" in capsys.readouterr().err


def test_unknown_stage_exits_non_zero(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = pipeline.main(
        ["--config", str(CANONICAL), "--stage", "does-not-exist"]
    )

    assert exit_code != 0
    assert "unknown stage" in capsys.readouterr().err


def test_each_route_config_hashes_distinctly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    digests = []
    for name in ("canonical_colmap_openmvs", "open3d_alpha", "openmvs_direct"):
        exit_code = pipeline.main(
            [
                "--config",
                str(CONFIG_DIR / f"{name}.yaml"),
                "--stage",
                "config-check",
                "--window",
                "smoke",
            ]
        )

        assert exit_code == 0
        printed = capsys.readouterr().out.strip()
        assert HEX64.match(printed), name
        digests.append(printed)

    assert len(set(digests)) == 3


def test_route_override_changes_config_hash(
    capsys: pytest.CaptureFixture[str],
) -> None:
    base = pipeline.main(
        ["--config", str(CANONICAL), "--stage", "config-check"]
    )
    assert base == 0
    base_digest = capsys.readouterr().out.strip()

    overridden = pipeline.main(
        [
            "--config",
            str(CANONICAL),
            "--route",
            "openmvs_direct",
            "--stage",
            "config-check",
        ]
    )
    assert overridden == 0
    assert capsys.readouterr().out.strip() != base_digest
