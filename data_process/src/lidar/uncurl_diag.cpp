#include <raylidar/common/io.h>
#include <raylidar/common/mask_bits.h>
#include <raylidar/lidar/uncurl.h>
#include <raylidar/lidar/uncurl_diag.h>
#include <raylidar/lidar/velodyne.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace raylidar::lidar
{
namespace
{
namespace fs = std::filesystem;

constexpr int kExitInvalidInput = 4;
constexpr double kPi = 3.141592653589793238462643383279;
constexpr double kMatchEpsilon = 1e-12;

[[nodiscard]] std::string FrameName(const int frame)
{
    char buffer[16] = {};
    std::snprintf(buffer, sizeof(buffer), "%010d", frame);
    return std::string(buffer);
}

[[nodiscard]] std::vector<std::uint8_t> ReadMaskFile(const std::filesystem::path &maskPath,
                                                     const std::size_t expectedPoints)
{
    std::error_code error;
    const std::uintmax_t size = std::filesystem::file_size(maskPath, error);
    if (error)
    {
        throw IoError("missing point mask " + maskPath.string() +
                          " (run mask-build before uncurl-check): " + error.message(),
                      kExitInvalidInput);
    }
    if (size != expectedPoints)
    {
        throw IoError("point mask " + maskPath.string() + " has " + std::to_string(size) + " bits but the scan has " +
                          std::to_string(expectedPoints) + " points",
                      kExitInvalidInput);
    }
    std::ifstream input(maskPath, std::ios::binary);
    if (!input.good())
    {
        throw IoError("cannot read point mask " + maskPath.string(), kExitInvalidInput);
    }
    std::vector<std::uint8_t> bits(expectedPoints);
    input.read(reinterpret_cast<char *>(bits.data()), static_cast<std::streamsize>(bits.size()));
    if (!input.good())
    {
        throw IoError("truncated point mask " + maskPath.string(), kExitInvalidInput);
    }
    return bits;
}

struct VoxelKey
{
    int x = 0;
    int y = 0;
    int z = 0;

    bool operator==(const VoxelKey &other) const noexcept
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelHash
{
    std::size_t operator()(const VoxelKey &key) const noexcept
    {
        const auto mix = [](std::uint64_t value) {
            value ^= value >> 33U;
            value *= 0xff51afd7ed558ccdULL;
            value ^= value >> 33U;
            return value;
        };
        const std::uint64_t packed = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.x)) << 32U) ^
                                     (static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.y)) << 16U) ^
                                     static_cast<std::uint32_t>(key.z);
        return static_cast<std::size_t>(mix(packed));
    }
};

/// Accumulates signed range errors per azimuth bin for one variant.
struct BinAccumulator
{
    explicit BinAccumulator(const double widthDeg)
        : widthDeg(widthDeg), binCount(static_cast<std::size_t>(std::llround(360.0 / widthDeg)))
    {
        if (!(widthDeg > 0.0) || binCount == 0U)
        {
            throw std::invalid_argument("azimuth bin width must be positive and divide 360");
        }
        sums.assign(binCount, 0.0);
        counts.assign(binCount, 0U);
    }

    void Add(const double azimuthDeg, const double signedErrorM)
    {
        double shifted = azimuthDeg + 180.0;
        if (shifted < 0.0)
        {
            shifted = 0.0;
        }
        if (shifted >= 360.0)
        {
            shifted = std::nextafter(360.0, 0.0);
        }
        std::size_t index = static_cast<std::size_t>(std::floor(shifted / widthDeg));
        if (index >= binCount)
        {
            index = binCount - 1U;
        }
        sums[index] += signedErrorM;
        counts[index] += 1U;
    }

    double widthDeg = 5.0;
    std::size_t binCount = 72U;
    std::vector<double> sums;
    std::vector<std::size_t> counts;
};

void AccumulateMatches(const std::vector<Eigen::Vector3d> &sourceSensor,
                       const std::vector<Eigen::Vector3d> &targetSensor, const Eigen::Matrix4d &T_W_V_source,
                       const Eigen::Matrix4d &T_W_V_target, const UncurlThresholds &thresholds,
                       BinAccumulator &accumulator)
{
    if (targetSensor.empty() || sourceSensor.empty())
    {
        return;
    }

    const double voxel = thresholds.matchVoxelM;
    if (!(voxel > 0.0))
    {
        throw std::invalid_argument("match voxel size must be positive");
    }
    const double maxDistance = thresholds.maxMatchDistanceM;
    if (!(maxDistance > 0.0))
    {
        throw std::invalid_argument("max match distance must be positive");
    }
    const double maxDistanceSquared = maxDistance * maxDistance;

    std::vector<Eigen::Vector3d> targetWorld;
    targetWorld.reserve(targetSensor.size());
    std::unordered_map<VoxelKey, std::vector<std::size_t>, VoxelHash> targetHash;
    targetHash.reserve(targetSensor.size());
    for (const Eigen::Vector3d &point : targetSensor)
    {
        const Eigen::Vector4d homogeneous(point.x(), point.y(), point.z(), 1.0);
        const Eigen::Vector3d world = (T_W_V_target * homogeneous).head<3>();
        targetWorld.push_back(world);
        const VoxelKey key{static_cast<int>(std::floor(world.x() / voxel)),
                           static_cast<int>(std::floor(world.y() / voxel)),
                           static_cast<int>(std::floor(world.z() / voxel))};
        targetHash[key].push_back(targetWorld.size() - 1U);
    }

    const Eigen::Vector3d sourceOrigin = T_W_V_source.block<3, 1>(0, 3);
    for (const Eigen::Vector3d &point : sourceSensor)
    {
        const Eigen::Vector4d homogeneous(point.x(), point.y(), point.z(), 1.0);
        const Eigen::Vector3d world = (T_W_V_source * homogeneous).head<3>();
        const Eigen::Vector3d direction = world - sourceOrigin;
        const double directionNorm = direction.norm();
        if (!(directionNorm > kMatchEpsilon))
        {
            continue;
        }
        const Eigen::Vector3d unitDirection = direction / directionNorm;

        const VoxelKey center{static_cast<int>(std::floor(world.x() / voxel)),
                              static_cast<int>(std::floor(world.y() / voxel)),
                              static_cast<int>(std::floor(world.z() / voxel))};
        double bestSquared = maxDistanceSquared;
        bool found = false;
        Eigen::Vector3d bestPoint = Eigen::Vector3d::Zero();
        for (int dx = -1; dx <= 1; ++dx)
        {
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dz = -1; dz <= 1; ++dz)
                {
                    const auto bucket = targetHash.find(VoxelKey{center.x + dx, center.y + dy, center.z + dz});
                    if (bucket == targetHash.end())
                    {
                        continue;
                    }
                    for (const std::size_t index : bucket->second)
                    {
                        const double squared = (targetWorld[index] - world).squaredNorm();
                        if (squared <= bestSquared)
                        {
                            bestSquared = squared;
                            bestPoint = targetWorld[index];
                            found = true;
                        }
                    }
                }
            }
        }
        if (!found)
        {
            continue;
        }

        const double signedError = (world - bestPoint).dot(unitDirection);
        const double azimuthDeg = std::atan2(point.y(), point.x()) * 180.0 / kPi;
        accumulator.Add(azimuthDeg, signedError);
    }
}

[[nodiscard]] std::vector<AzimuthBin> MakeBins(const BinAccumulator &accumulator)
{
    std::vector<AzimuthBin> bins;
    bins.reserve(accumulator.binCount);
    for (std::size_t index = 0; index < accumulator.binCount; ++index)
    {
        AzimuthBin bin;
        bin.centerDeg = -180.0 + (static_cast<double>(index) + 0.5) * accumulator.widthDeg;
        bin.count = accumulator.counts[index];
        bin.meanSignedRangeErrorM = bin.count == 0U ? 0.0 : accumulator.sums[index] / static_cast<double>(bin.count);
        bins.push_back(bin);
    }
    return bins;
}

[[nodiscard]] double PeakToPeak(const std::vector<AzimuthBin> &bins)
{
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    bool any = false;
    for (const AzimuthBin &bin : bins)
    {
        if (bin.count == 0U)
        {
            continue;
        }
        any = true;
        minimum = std::min(minimum, bin.meanSignedRangeErrorM);
        maximum = std::max(maximum, bin.meanSignedRangeErrorM);
    }
    return any ? maximum - minimum : 0.0;
}

[[nodiscard]] double MaxAbsBinMean(const std::vector<AzimuthBin> &bins)
{
    double maximum = 0.0;
    for (const AzimuthBin &bin : bins)
    {
        if (bin.count == 0U)
        {
            continue;
        }
        maximum = std::max(maximum, std::abs(bin.meanSignedRangeErrorM));
    }
    return maximum;
}

[[nodiscard]] std::size_t MatchedCount(const std::vector<AzimuthBin> &bins)
{
    std::size_t count = 0;
    for (const AzimuthBin &bin : bins)
    {
        count += bin.count;
    }
    return count;
}

bool FitSinusoid(const std::vector<AzimuthBin> &bins, const std::size_t minSamples, double *amplitude,
                 std::string *failure)
{
    Eigen::Matrix3d normal = Eigen::Matrix3d::Zero();
    Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
    std::size_t used = 0;
    for (const AzimuthBin &bin : bins)
    {
        if (bin.count < minSamples)
        {
            continue;
        }
        ++used;
        const double theta = bin.centerDeg * kPi / 180.0;
        const Eigen::Vector3d basis(1.0, std::cos(theta), std::sin(theta));
        const double weight = static_cast<double>(bin.count);
        normal += weight * basis * basis.transpose();
        rhs += weight * basis * bin.meanSignedRangeErrorM;
    }

    if (used < 3U)
    {
        *failure = "insufficient azimuth bins for the sinusoid fit (" + std::to_string(used) +
                   " bins with >= " + std::to_string(minSamples) + " samples)";
        return false;
    }

    const Eigen::FullPivLU<Eigen::Matrix3d> decomposition(normal);
    if (!decomposition.isInvertible())
    {
        *failure = "sinusoid normal equations are singular";
        return false;
    }
    const Eigen::Vector3d coefficients = decomposition.solve(rhs);
    *amplitude = std::hypot(coefficients(1), coefficients(2));
    return true;
}

struct LoadedFrame
{
    int frame = 0;
    Eigen::Matrix4d T_W_V = Eigen::Matrix4d::Identity();
    std::vector<Eigen::Vector3d> raw;
    std::vector<Eigen::Vector3d> uncurled;
};

[[nodiscard]] LoadedFrame LoadFrame(const int frame, const std::filesystem::path &lidarRoot,
                                    const std::filesystem::path &maskDir, const calib::Kitti360Calib &calib,
                                    const std::map<int, Eigen::Matrix4d> &poses,
                                    const std::vector<std::int64_t> &timestamps)
{
    const auto current = poses.find(frame);
    const auto next = poses.find(frame + 1);
    if (current == poses.end() || next == poses.end())
    {
        throw IoError("frame " + std::to_string(frame) + " needs exact poses for itself and its successor (uncurl)",
                      kExitInvalidInput);
    }
    if (static_cast<std::size_t>(frame + 1) >= timestamps.size())
    {
        throw IoError("frame " + std::to_string(frame) + " needs timestamps for itself and its successor (uncurl)",
                      kExitInvalidInput);
    }

    UncurlParams params;
    params.T_W_V_curr = calib::ComposeTWV(calib, current->second);
    params.T_W_V_next = calib::ComposeTWV(calib, next->second);
    params.t_curr_ns = timestamps[static_cast<std::size_t>(frame)];
    params.t_next_ns = timestamps[static_cast<std::size_t>(frame + 1)];

    const fs::path scanPath = lidarRoot / "data" / (FrameName(frame) + ".bin");
    const std::vector<PointXYZI> scan = LoadVelodyneScan(scanPath);
    const std::vector<std::uint8_t> maskBits =
        ReadMaskFile(maskDir / ("frame_" + FrameName(frame)) / "points.bin", scan.size());

    LoadedFrame loaded;
    loaded.frame = frame;
    loaded.T_W_V = params.T_W_V_curr;
    loaded.raw.reserve(scan.size());
    loaded.uncurled.reserve(scan.size());
    for (std::size_t index = 0; index < scan.size(); ++index)
    {
        if (!masking::IsKeptStatic(maskBits[index]))
        {
            continue;
        }
        const Eigen::Vector3d point(scan[index].x, scan[index].y, scan[index].z);
        if (!point.allFinite())
        {
            throw IoError("mask marks a non-finite point as kept static in " + scanPath.string(), kExitInvalidInput);
        }
        loaded.raw.push_back(point);
        loaded.uncurled.push_back(UncurlPoint(point, params));
    }
    return loaded;
}
} // namespace

UncurlDiagnostic EvaluateUncurl(const std::vector<int> &frames, const std::filesystem::path &datasetRoot,
                                const std::filesystem::path &lidarRoot, const std::filesystem::path &maskDir,
                                const calib::Kitti360Calib &calib, const std::map<int, Eigen::Matrix4d> &poses,
                                const UncurlThresholds &thresholds)
{
    (void)datasetRoot; // The CLI resolves poses and calibration from it.
    if (frames.size() < 2U)
    {
        throw IoError("uncurl diagnostic needs at least two frames", kExitInvalidInput);
    }
    for (std::size_t index = 1; index < frames.size(); ++index)
    {
        if (frames[index] <= frames[index - 1])
        {
            throw IoError("uncurl diagnostic frames must be strictly ascending", kExitInvalidInput);
        }
    }

    const std::vector<std::int64_t> timestamps = LoadVelodyneTimestamps(lidarRoot / "timestamps.txt");

    std::vector<LoadedFrame> loaded;
    loaded.reserve(frames.size());
    for (const int frame : frames)
    {
        loaded.push_back(LoadFrame(frame, lidarRoot, maskDir, calib, poses, timestamps));
    }

    BinAccumulator rawAccumulator(thresholds.binWidthDeg);
    BinAccumulator uncurledAccumulator(thresholds.binWidthDeg);
    for (std::size_t index = 1; index < loaded.size(); ++index)
    {
        const LoadedFrame &source = loaded[index - 1U];
        const LoadedFrame &target = loaded[index];
        AccumulateMatches(source.raw, target.raw, source.T_W_V, target.T_W_V, thresholds, rawAccumulator);
        AccumulateMatches(source.uncurled, target.uncurled, source.T_W_V, target.T_W_V, thresholds,
                          uncurledAccumulator);
    }

    UncurlDiagnostic diagnostic;
    diagnostic.rawBins = MakeBins(rawAccumulator);
    diagnostic.uncurledBins = MakeBins(uncurledAccumulator);
    diagnostic.rawPeakToPeakM = PeakToPeak(diagnostic.rawBins);
    diagnostic.uncurledPeakToPeakM = PeakToPeak(diagnostic.uncurledBins);
    diagnostic.rawMaxAbsBinMeanM = MaxAbsBinMean(diagnostic.rawBins);
    diagnostic.uncurledMaxAbsBinMeanM = MaxAbsBinMean(diagnostic.uncurledBins);
    diagnostic.rawMatchedCount = MatchedCount(diagnostic.rawBins);
    diagnostic.uncurledMatchedCount = MatchedCount(diagnostic.uncurledBins);

    std::string rawFitFailure;
    if (!FitSinusoid(diagnostic.rawBins, thresholds.minBinSamples, &diagnostic.rawSinusoidAmplitudeM, &rawFitFailure))
    {
        // The raw ablation is reported, not gated; record the reason in the
        // failure list only when the uncurled fit also fails below.
    }
    std::string uncurledFitFailure;
    if (!FitSinusoid(diagnostic.uncurledBins, thresholds.minBinSamples, &diagnostic.uncurledSinusoidAmplitudeM,
                     &uncurledFitFailure))
    {
        diagnostic.failures.push_back(uncurledFitFailure);
    }

    if (diagnostic.uncurledMaxAbsBinMeanM > thresholds.maxAbsBinMeanM)
    {
        diagnostic.failures.push_back("uncurled max abs bin mean " + std::to_string(diagnostic.uncurledMaxAbsBinMeanM) +
                                      " m exceeds " + std::to_string(thresholds.maxAbsBinMeanM) + " m");
    }
    if (diagnostic.uncurledSinusoidAmplitudeM > thresholds.maxSinusoidAmplitudeM)
    {
        diagnostic.failures.push_back("uncurled sinusoid amplitude " +
                                      std::to_string(diagnostic.uncurledSinusoidAmplitudeM) + " m exceeds " +
                                      std::to_string(thresholds.maxSinusoidAmplitudeM) + " m");
    }

    diagnostic.pass = diagnostic.failures.empty();
    return diagnostic;
}
} // namespace raylidar::lidar
