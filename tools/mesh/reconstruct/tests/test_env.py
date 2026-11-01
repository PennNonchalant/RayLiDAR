from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools.mesh.reconstruct import env


def _make_bundle(root: Path) -> Path:
    images = root / "raw" / "images" / "image_00"
    images.mkdir(parents=True)
    for frame in (4615, 4616, 4617):
        (images / f"{frame:010d}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return root


def _write_fake_colmap(path: Path, banner: str, patch_match_ok: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    exit_code = 0 if patch_match_ok else 1
    path.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "patch_match_stereo" ]; then echo "patch match"; '
        f"exit {exit_code}; fi\n"
        f'echo "{banner}"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | 0o111)
    return path


def test_probe_returns_required_schema(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path / "bundle")
    missing = tmp_path / "missing"

    report = env.probe_environment(
        bundle,
        colmap_path=missing / "colmap",
        colmap_cuda_path=missing / "colmap",
        openmvs_root=missing / "openmvs",
    )

    assert set(report) == {
        "schema_version",
        "generated_at_utc",
        "python",
        "packages",
        "colmap",
        "colmap_cuda",
        "openmvs",
        "hardware",
        "dense_backend",
        "threads",
        "bundle",
    }
    assert report["schema_version"] == 1
    assert report["generated_at_utc"].endswith("Z")
    assert set(report["packages"]) == {"open3d", "numpy", "trimesh", "pyyaml"}
    assert set(report["colmap"]) == {"path", "banner", "has_cuda", "version"}
    assert set(report["colmap_cuda"]) == {"path", "has_cuda", "build"}
    assert report["colmap"]["has_cuda"] is False
    assert report["dense_backend"] == "unavailable_cuda"
    assert isinstance(report["hardware"]["logical_cores"], int)
    assert isinstance(report["threads"], int)
    assert report["bundle"]["image_00_count"] == 3
    assert report["bundle"]["first_frame"] == 4615
    assert report["bundle"]["last_frame"] == 4617


def test_write_env_report_is_idempotent(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path / "bundle")
    missing = tmp_path / "missing"
    report = env.probe_environment(
        bundle,
        colmap_path=missing / "colmap",
        colmap_cuda_path=missing / "colmap",
        openmvs_root=missing / "openmvs",
    )
    out = tmp_path / "artifacts" / "env_report.json"

    first = env.write_env_report(out, report)
    first_bytes = out.read_bytes()
    second = env.write_env_report(out, report)

    assert first == second
    assert len(first) == 64
    assert out.read_bytes() == first_bytes
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["dense_backend"] == "unavailable_cuda"


def test_cpu_banner_does_not_claim_cuda(tmp_path: Path) -> None:
    binary = _write_fake_colmap(
        tmp_path / "colmap", "COLMAP 4.0.4 -- SfM (Commit x on y without CUDA)"
    )

    report = env.probe_environment(
        tmp_path / "bundle",
        colmap_path=binary,
        colmap_cuda_path=tmp_path / "no-cuda",
        openmvs_root=tmp_path / "openmvs",
    )

    assert env.banner_has_cuda(report["colmap"]["banner"]) is False
    assert report["colmap"]["has_cuda"] is False
    assert report["colmap"]["version"] == "4.0.4"
    assert report["dense_backend"] == "unavailable_cuda"


def test_cuda_binary_requires_patch_match_stereo(tmp_path: Path) -> None:
    cuda_binary = _write_fake_colmap(
        tmp_path / "colmap-cuda", "COLMAP 4.2.1 (Commit x on y with CUDA)"
    )
    report = env.probe_environment(
        tmp_path / "bundle",
        colmap_path=tmp_path / "no-cpu",
        colmap_cuda_path=cuda_binary,
        openmvs_root=tmp_path / "openmvs",
    )

    assert report["colmap_cuda"]["has_cuda"] is True
    assert report["dense_backend"] == "colmap_cuda"

    broken = _write_fake_colmap(
        tmp_path / "colmap-cuda-broken",
        "COLMAP 4.2.1 (Commit x on y with CUDA)",
        patch_match_ok=False,
    )
    broken_report = env.probe_environment(
        tmp_path / "bundle",
        colmap_path=tmp_path / "no-cpu",
        colmap_cuda_path=broken,
        openmvs_root=tmp_path / "openmvs",
    )

    assert broken_report["colmap_cuda"]["has_cuda"] is False
    assert broken_report["dense_backend"] == "unavailable_cuda"


def test_banner_without_cuda_flag(tmp_path: Path) -> None:
    assert env.banner_has_cuda("COLMAP 4.0.4 ... without CUDA") is False
    assert env.banner_has_cuda("COLMAP 4.2.1 ... with CUDA") is True


def test_openmvs_metadata_is_recorded(tmp_path: Path) -> None:
    root = tmp_path / "openmvs"
    root.mkdir()
    for name in env.OPENMVS_BINARIES:
        binary = root / name
        binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        binary.chmod(0o755)
    (root / ".bundle.json").write_text(
        json.dumps({"version": "2.4.0", "zip_sha256": "f" * 64}), encoding="utf-8"
    )

    info = env._openmvs_info(root)

    assert info["version"] == "2.4.0"
    assert info["zip_sha256"] == "f" * 64
    assert set(info["binaries"]) == set(env.OPENMVS_BINARIES)
    assert all(len(entry["sha256"]) == 64 for entry in info["binaries"].values())


def test_cli_writes_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bundle = _make_bundle(tmp_path / "bundle")
    out = tmp_path / "env_report.json"

    exit_code = env.main(["--bundle-root", str(bundle), "--out", str(out)])

    assert exit_code == 0
    assert out.is_file()
    printed = capsys.readouterr().out
    assert "dense_backend:" in printed
    assert "env_report:" in printed
