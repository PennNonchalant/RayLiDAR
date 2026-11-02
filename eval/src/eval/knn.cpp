#include <raylidar/eval/knn.h>

#include <nanoflann.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace raylidar::eval
{
namespace
{
/// Row-major dataset adaptor required by nanoflann.
struct PointCloudAdaptor
{
    const std::vector<double> *coordinates = nullptr;

    [[nodiscard]] std::size_t kdtree_get_point_count() const
    {
        return coordinates->size() / 3U;
    }

    [[nodiscard]] double kdtree_get_pt(const std::size_t index, const std::size_t dimension) const
    {
        return (*coordinates)[index * 3U + dimension];
    }

    template <class BBox> bool kdtree_get_bbox(BBox & /*bbox*/) const
    {
        return false;
    }
};

using Distance = nanoflann::L2_Simple_Adaptor<double, PointCloudAdaptor, double, std::size_t>;
using Tree = nanoflann::KDTreeSingleIndexAdaptor<Distance, PointCloudAdaptor, 3, std::uint32_t>;

void RequireFiniteQuery(const Eigen::Vector3d &query)
{
    if (!query.allFinite())
    {
        throw std::invalid_argument("PointCloudIndex: query is not finite");
    }
}

void ToArray(const Eigen::Vector3d &query, double (&out)[3])
{
    out[0] = query.x();
    out[1] = query.y();
    out[2] = query.z();
}
} // namespace

struct PointCloudIndex::Impl
{
    std::vector<double> coordinates;
    PointCloudAdaptor adaptor;
    std::unique_ptr<Tree> tree;
};

PointCloudIndex::PointCloudIndex(std::span<const Eigen::Vector3d> points) : impl_(std::make_unique<Impl>())
{
    if (points.empty())
    {
        throw std::invalid_argument("PointCloudIndex: empty point cloud");
    }

    impl_->coordinates.reserve(points.size() * 3U);
    for (const auto &point : points)
    {
        if (!point.allFinite())
        {
            throw std::invalid_argument("PointCloudIndex: non-finite input point");
        }
        impl_->coordinates.push_back(point.x());
        impl_->coordinates.push_back(point.y());
        impl_->coordinates.push_back(point.z());
    }

    impl_->adaptor.coordinates = &impl_->coordinates;
    // eps defaults to zero; the index is never built with an approximate flag.
    impl_->tree = std::make_unique<Tree>(
        3, impl_->adaptor,
        nanoflann::KDTreeSingleIndexAdaptorParams(10, nanoflann::KDTreeSingleIndexAdaptorFlags::None, 1));
    impl_->tree->buildIndex();
}

PointCloudIndex::~PointCloudIndex() = default;
PointCloudIndex::PointCloudIndex(PointCloudIndex &&other) noexcept = default;
PointCloudIndex &PointCloudIndex::operator=(PointCloudIndex &&other) noexcept = default;

std::size_t PointCloudIndex::size() const
{
    return impl_->coordinates.size() / 3U;
}

std::pair<std::size_t, double> PointCloudIndex::Nearest(const Eigen::Vector3d &query) const
{
    RequireFiniteQuery(query);
    if (size() == 0U)
    {
        throw std::invalid_argument("PointCloudIndex: cannot query an empty index");
    }

    double queryArray[3];
    ToArray(query, queryArray);
    std::uint32_t index = 0;
    double squaredDistance = 0.0;
    const std::size_t found = impl_->tree->knnSearch(queryArray, 1, &index, &squaredDistance);
    if (found == 0U)
    {
        throw std::runtime_error("PointCloudIndex: nearest-neighbour search returned no result");
    }
    return {static_cast<std::size_t>(index), std::sqrt(squaredDistance)};
}

void PointCloudIndex::RadiusSearch(const Eigen::Vector3d &query, const double radius,
                                   std::vector<std::pair<std::size_t, double>> &matches) const
{
    RequireFiniteQuery(query);
    if (!std::isfinite(radius) || radius < 0.0)
    {
        throw std::invalid_argument("PointCloudIndex: radius must be finite and non-negative");
    }

    matches.clear();
    if (size() == 0U)
    {
        return;
    }

    double queryArray[3];
    ToArray(query, queryArray);
    nanoflann::SearchParameters searchParameters;
    searchParameters.eps = 0.0F; // exact: never approximate
    searchParameters.sorted = true;

    std::vector<nanoflann::ResultItem<std::uint32_t, double>> found;
    const auto foundCount = impl_->tree->radiusSearch(queryArray, radius * radius, found, searchParameters);
    (void)foundCount;

    matches.reserve(found.size());
    for (const auto &item : found)
    {
        matches.emplace_back(static_cast<std::size_t>(item.first), std::sqrt(item.second));
    }

    // nanoflann sorts by distance only; the frozen contract is (distance, index).
    std::sort(matches.begin(), matches.end(), [](const auto &lhs, const auto &rhs) {
        if (lhs.second != rhs.second)
        {
            return lhs.second < rhs.second;
        }
        return lhs.first < rhs.first;
    });
}
} // namespace raylidar::eval
