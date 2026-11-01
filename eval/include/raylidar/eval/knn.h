#pragma once

#include <raylidar/export.h>

#include <Eigen/Core>

#include <cstddef>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace raylidar::eval
{
/// Exact, double-precision nearest-neighbour / radius index over 3-D points.
///
/// Backed by `nanoflann` with `eps = 0` (exact search, no approximation). All
/// distances returned by the public API are **unsquared Euclidean metres**, so
/// they can be compared directly against the protocol's tau values.
///
/// The index copies its input points; the caller's span may be destroyed right
/// after construction. The type is move-only.
class RAYLIDAR_EXPORT PointCloudIndex
{
public:
    /// Copy `points` into the index and build the KD-tree.
    ///
    /// Throws `std::invalid_argument` for an empty cloud or a non-finite point.
    explicit PointCloudIndex(std::span<const Eigen::Vector3d> points);
    ~PointCloudIndex();

    PointCloudIndex(PointCloudIndex &&other) noexcept;
    PointCloudIndex &operator=(PointCloudIndex &&other) noexcept;
    PointCloudIndex(const PointCloudIndex &) = delete;
    PointCloudIndex &operator=(const PointCloudIndex &) = delete;

    /// Number of indexed points.
    [[nodiscard]] std::size_t size() const;

    /// Exact nearest point as `(index, distance)`; distance is Euclidean metres.
    ///
    /// Throws `std::invalid_argument` for a non-finite query or an empty index.
    [[nodiscard]] std::pair<std::size_t, double> Nearest(const Eigen::Vector3d &query) const;

    /// All points with distance `<= radius`, sorted by `(distance, index)`.
    ///
    /// `radius` is Euclidean metres. `radius == 0` returns exact duplicates.
    /// Throws `std::invalid_argument` for a negative or non-finite radius.
    /// An empty result is returned as an empty vector, never as an exception.
    void RadiusSearch(const Eigen::Vector3d &query, double radius,
                      std::vector<std::pair<std::size_t, double>> &matches) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace raylidar::eval
