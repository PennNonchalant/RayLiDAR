#pragma once

#include <raylidar/eval/types.h>

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace raylidar::test
{
/// Fixed-seed deterministic points covering roughly [-10, 10]^3 metres.
std::vector<Eigen::Vector3d> MakeDeterministicPoints(std::size_t count, std::uint64_t seed);

/// Fixed-seed deterministic query points covering roughly [-12, 12]^3 metres.
std::vector<Eigen::Vector3d> MakeDeterministicQueries(std::size_t count, std::uint64_t seed);

/// Fixed-seed points around (4000, 500, 100) with sub-millimetre offsets
/// (the float64 reason for the nanoflann backend).
std::vector<Eigen::Vector3d> MakeWorldScalePoints(std::size_t count, std::uint64_t seed);

/// Index of the closest point; ties resolved by the lowest index.
std::size_t BruteNearestIndex(const std::vector<Eigen::Vector3d> &points, const Eigen::Vector3d &query);

/// Unsquared Euclidean distance to the closest point.
double BruteNearestDistance(const std::vector<Eigen::Vector3d> &points, const Eigen::Vector3d &query);

/// Every point with `d <= radius`, sorted by `(distance, index)`; distances are
/// unsquared Euclidean metres.
std::vector<std::pair<std::size_t, double>> BruteRadius(const std::vector<Eigen::Vector3d> &points,
                                                        const Eigen::Vector3d &query, double radius);

/// Build a valid `RaySample` with the bucket derived from `t_gt`.
///
/// `t_pred` defaults to `+inf` (a miss). The direction defaults to `+Z`; callers
/// that need world-scale points pass an explicit origin/direction.
raylidar::eval::RaySample MakeSample(std::uint64_t ray_id, std::uint32_t frame, std::uint16_t camera, double t_gt,
                                     double t_pred, const Eigen::Vector3d &origin = Eigen::Vector3d::Zero(),
                                     const Eigen::Vector3d &dir = Eigen::Vector3d(0.0, 0.0, 1.0));

/// Identity prediction: `t_pred = t_gt`, so predicted points equal GT points.
std::vector<raylidar::eval::RaySample> MakeIdentitySamples(const std::vector<double> &ranges, std::uint32_t frame = 0,
                                                           std::uint16_t camera = 0);
} // namespace raylidar::test
