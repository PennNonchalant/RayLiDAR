#pragma once

#include <raylidar/calib/kitti360.h>
#include <raylidar/export.h>

#include <Eigen/Core>

#include <cstddef>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace raylidar::lidar
{
/// Frozen thresholds for the raw-vs-curled azimuth-flatness gate.
///
/// These are populated from `configs/phase2.yaml` `uncurl:` and are never
/// overridden from the command line or the report JSON.
struct UncurlThresholds
{
    double maxAbsBinMeanM = 0.05;
    double maxSinusoidAmplitudeM = 0.03;
    double binWidthDeg = 5.0;
    double matchVoxelM = 0.2;
    double maxMatchDistanceM = 0.3;
    std::size_t minBinSamples = 100;
};

/// One azimuth bin of the signed cross-frame range error.
struct AzimuthBin
{
    double centerDeg = 0.0;
    std::size_t count = 0;
    double meanSignedRangeErrorM = 0.0;
};

/// Raw-vs-uncurled signed range error versus azimuth.
struct UncurlDiagnostic
{
    std::vector<AzimuthBin> rawBins;
    std::vector<AzimuthBin> uncurledBins;
    double rawPeakToPeakM = 0.0;
    double uncurledPeakToPeakM = 0.0;
    double rawMaxAbsBinMeanM = 0.0;
    double uncurledMaxAbsBinMeanM = 0.0;
    double rawSinusoidAmplitudeM = 0.0;
    double uncurledSinusoidAmplitudeM = 0.0;
    std::size_t rawMatchedCount = 0;
    std::size_t uncurledMatchedCount = 0;
    bool pass = false;
    std::vector<std::string> failures;
};

/// Masked raw-vs-curled ablation over consecutive frame pairs.
///
/// Algorithm (identical for both variants):
///  1. Load `maskDir/frame_{frame:010d}/points.bin` (one `uint8_t` per raw
///     point) and keep only points where `masking::IsKeptStatic` is set.
///     A missing, short, or long mask is a hard `IoError` (exit 4); there is
///     never an unmasked fallback.
///  2. Uncurl each kept point with `[timestamps[f], timestamps[f+1])` and the
///     exact poses for `f` and `f+1` (`T_W_V = ComposeTWV(calib, T_W_I)`).
///  3. For each consecutive pair `(f, g)`, transform both clouds to world with
///     `T_W_V(f)` and `T_W_V(g)`, build a `matchVoxelM` voxel hash of `g`, and
///     for every `f` point search the 27 neighbouring voxels for the nearest
///     return within `maxMatchDistanceM`.
///  4. Record `e = dot(p_W^f - q_W^g, d_W^f)` where `d_W^f` is the unit world
///     ray direction of the frame-`f` point, binned by the Velodyne azimuth in
///     `binWidthDeg` bins. Every bin is reported, including bins below
///     `minBinSamples`; only bins with at least `minBinSamples` enter the
///     `e(theta) = a0 + a1 cos(theta) + b1 sin(theta)` fit.
///  5. Repeat exactly the same pipeline on the raw (curled) points.
///
/// `pass` requires `uncurledMaxAbsBinMeanM <= maxAbsBinMeanM` and
/// `uncurledSinusoidAmplitudeM <= maxSinusoidAmplitudeM`. The raw variant is
/// always reported but never gated.
///
/// Throws `IoError` (exit 4) for missing scans/timestamps/poses/masks, a
/// non-ascending or short frame list, or a mask length mismatch.
RAYLIDAR_EXPORT UncurlDiagnostic EvaluateUncurl(
    const std::vector<int> &frames, const std::filesystem::path &datasetRoot,
    const std::filesystem::path &lidarRoot, const std::filesystem::path &maskDir,
    const calib::Kitti360Calib &calib, const std::map<int, Eigen::Matrix4d> &poses,
    const UncurlThresholds &thresholds);
} // namespace raylidar::lidar
