#pragma once

#include "simulation/HybridScan.h"
#include "simulation/KittiBundle.h"

#include <filesystem>
#include <functional>
#include <span>
#include <string_view>
#include <vector>

namespace raylidar::kitti
{
struct FrameScan
{
    simulation::ScanResult Result;
    std::vector<simulation::HitSource> Sources;
};

using TraceFrame = std::function<FrameScan(const simulation::Transform &, std::span<const rt::Ray>)>;

void EvaluateGtRays(const simulation::GtRays &gt, const simulation::ScanResult &result, std::string_view sceneName,
                    const std::array<double, 3> &origin, const std::filesystem::path &outputDirectory);
void PrintComparison(const simulation::GtRays &gt, const simulation::ScanResult &mesh,
                     const simulation::ScanResult &gaussian, const simulation::HybridScanResult &hybrid,
                     std::span<const std::uint8_t> untrustedTriangles);
void PrintHybridRouting(const simulation::GtRays &gt, const simulation::HybridScanResult &hybrid);
void RunVelodyneScans(const simulation::LidarSensor &sensor, std::span<const simulation::VelodynePose> poses,
                      const std::array<double, 3> &origin, const std::filesystem::path &outputDirectory,
                      std::string_view outputPrefix, std::span<const std::uint8_t> untrustedTriangles, bool gpuRayGen,
                      const TraceFrame &trace);
} // namespace raylidar::kitti
