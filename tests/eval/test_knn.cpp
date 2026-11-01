#include "fixture.h"

#include <raylidar/eval/knn.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{
using raylidar::eval::PointCloudIndex;
using raylidar::test::BruteNearestDistance;
using raylidar::test::BruteNearestIndex;
using raylidar::test::BruteRadius;
using raylidar::test::MakeDeterministicPoints;
using raylidar::test::MakeDeterministicQueries;
using raylidar::test::MakeWorldScalePoints;
} // namespace

TEST_CASE("PointCloudIndex nearest agrees with brute force", "[eval][knn]")
{
    const auto points = MakeDeterministicPoints(250U, 0x5EED1234ULL);
    const auto queries = MakeDeterministicQueries(100U, 0xC0FFEEULL);

    const PointCloudIndex index{std::span<const Eigen::Vector3d>(points)};
    REQUIRE(index.size() == points.size());

    for (const auto &query : queries)
    {
        const auto [pointIndex, distance] = index.Nearest(query);
        const double reference = BruteNearestDistance(points, query);
        CHECK(distance == Catch::Approx(reference).margin(1.0e-12));
        CHECK((points[pointIndex] - query).norm() == Catch::Approx(reference).margin(1.0e-12));
        CHECK(pointIndex == BruteNearestIndex(points, query));
    }
}

TEST_CASE("PointCloudIndex radius search returns the exact brute-force set", "[eval][knn]")
{
    const auto points = MakeDeterministicPoints(250U, 0xABCDEFULL);
    const auto queries = MakeDeterministicQueries(100U, 0x1234567ULL);
    const PointCloudIndex index{std::span<const Eigen::Vector3d>(points)};

    for (const auto &query : queries)
    {
        for (const double radius : {0.5, 2.0, 6.0})
        {
            std::vector<std::pair<std::size_t, double>> matches;
            index.RadiusSearch(query, radius, matches);
            const auto reference = BruteRadius(points, query, radius);

            REQUIRE(matches.size() == reference.size());
            for (std::size_t i = 0; i < matches.size(); ++i)
            {
                CHECK(matches[i].first == reference[i].first);
                CHECK(matches[i].second == Catch::Approx(reference[i].second).margin(1.0e-12));
            }
        }
    }
}

TEST_CASE("PointCloudIndex stays exact at world-scale coordinates", "[eval][knn]")
{
    const auto points = MakeWorldScalePoints(250U, 0x777ULL);
    const auto queries = MakeWorldScalePoints(100U, 0x888ULL);
    const PointCloudIndex index{std::span<const Eigen::Vector3d>(points)};

    for (const auto &query : queries)
    {
        const auto [pointIndex, distance] = index.Nearest(query);
        const double reference = BruteNearestDistance(points, query);
        CHECK(distance == Catch::Approx(reference).margin(1.0e-12));
        CHECK((points[pointIndex] - query).norm() == Catch::Approx(reference).margin(1.0e-12));
    }
}

TEST_CASE("PointCloudIndex rejects invalid input and returns empty matches", "[eval][knn]")
{
    const auto points = MakeDeterministicPoints(32U, 0x9ULL);

    std::vector<Eigen::Vector3d> nonFinite = points;
    nonFinite[3] = Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0);
    CHECK_THROWS_AS(PointCloudIndex{std::span<const Eigen::Vector3d>(nonFinite)},
                    std::invalid_argument);

    std::vector<Eigen::Vector3d> infinite = points;
    infinite[1] = Eigen::Vector3d(std::numeric_limits<double>::infinity(), 0.0, 0.0);
    CHECK_THROWS_AS(PointCloudIndex{std::span<const Eigen::Vector3d>(infinite)},
                    std::invalid_argument);

    const std::vector<Eigen::Vector3d> empty;
    CHECK_THROWS_AS(PointCloudIndex{std::span<const Eigen::Vector3d>(empty)},
                    std::invalid_argument);

    const PointCloudIndex index{std::span<const Eigen::Vector3d>(points)};
    std::vector<std::pair<std::size_t, double>> matches;
    CHECK_THROWS_AS(index.RadiusSearch(Eigen::Vector3d::Zero(), -1.0, matches),
                    std::invalid_argument);
    CHECK_THROWS_AS(index.RadiusSearch(
                        Eigen::Vector3d::Zero(), std::numeric_limits<double>::quiet_NaN(), matches),
                    std::invalid_argument);
    CHECK_THROWS_AS(
        index.RadiusSearch(Eigen::Vector3d::Zero(), std::numeric_limits<double>::infinity(),
                           matches),
        std::invalid_argument);

    matches.resize(7U);
    index.RadiusSearch(Eigen::Vector3d(1.0e6, 1.0e6, 1.0e6), 1.0e-6, matches);
    CHECK(matches.empty());
}

TEST_CASE("PointCloudIndex is deterministic and owns its point copy", "[eval][knn]")
{
    auto build = [] {
        std::vector<Eigen::Vector3d> points = MakeDeterministicPoints(64U, 0xDEADBEEFULL);
        return PointCloudIndex{std::span<const Eigen::Vector3d>(points)};
    };

    PointCloudIndex first = build();
    PointCloudIndex second = build();
    const auto queries = MakeDeterministicQueries(16U, 0xBEEFULL);
    for (const auto &query : queries)
    {
        CHECK(first.Nearest(query).first == second.Nearest(query).first);
        CHECK(first.Nearest(query).second == Catch::Approx(second.Nearest(query).second).margin(0.0));
    }

    std::vector<Eigen::Vector3d> source = MakeDeterministicPoints(16U, 0x1ULL);
    const PointCloudIndex owned{std::span<const Eigen::Vector3d>(source)};
    source.clear();
    source.shrink_to_fit();
    CHECK(owned.size() == 16U);
    CHECK(std::isfinite(owned.Nearest(Eigen::Vector3d::Zero()).second));
}
