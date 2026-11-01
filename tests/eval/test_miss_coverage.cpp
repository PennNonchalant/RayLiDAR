#include "fixture.h"

#include <raylidar/eval/metrics.h>
#include <raylidar/eval/types.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>
#include <vector>

namespace
{
using raylidar::eval::AggregateCell;
using raylidar::eval::ComputeMissMetrics;
using raylidar::eval::RawCounts;
using raylidar::eval::RaySample;
using raylidar::test::MakeSample;
} // namespace

TEST_CASE("miss rate and ray recall use n_valid as the denominator", "[eval][miss]")
{
    std::vector<RaySample> samples;
    std::uint64_t rayId = 0;
    for (int i = 0; i < 3; ++i)
    {
        samples.push_back(MakeSample(rayId++, 0U, 0U, 10.0, 10.0));
    }
    samples.push_back(MakeSample(rayId, 0U, 0U, 10.0, std::numeric_limits<double>::infinity()));

    const auto metrics = ComputeMissMetrics(samples);
    REQUIRE(metrics.n_total == 4U);
    REQUIRE(metrics.n_hits == 3U);
    REQUIRE(metrics.n_misses == 1U);
    REQUIRE(metrics.miss_rate.has_value());
    REQUIRE(metrics.ray_recall.has_value());
    CHECK(*metrics.miss_rate == Catch::Approx(0.25).margin(1.0e-12));
    CHECK(*metrics.ray_recall == Catch::Approx(0.75).margin(1.0e-12));
}

TEST_CASE("zero valid rays make miss rate and ray recall undefined", "[eval][miss]")
{
    const std::vector<RaySample> empty;
    const auto metrics = ComputeMissMetrics(empty);
    CHECK(metrics.n_total == 0U);
    CHECK_FALSE(metrics.miss_rate.has_value());
    CHECK_FALSE(metrics.ray_recall.has_value());
}

TEST_CASE("coverage is n_valid over n_raw and null when n_raw is zero", "[eval][miss]")
{
    std::vector<RaySample> samples;
    for (std::uint64_t i = 0; i < 4U; ++i)
    {
        samples.push_back(MakeSample(i, 0U, 0U, 10.0, 10.0));
    }

    RawCounts raw;
    raw.raw_total = 8U;
    raw.raw_bucket = {8U, 0U, 0U, 0U};
    const auto cell = AggregateCell(samples, raw);
    REQUIRE(cell.coverage.has_value());
    CHECK(*cell.coverage == Catch::Approx(0.5).margin(1.0e-12));
    CHECK(cell.n_valid == 4U);

    const RawCounts unavailable;
    const auto undefined = AggregateCell(samples, unavailable);
    CHECK_FALSE(undefined.coverage.has_value());
    CHECK(undefined.n_raw == 0U);
    CHECK(undefined.n_valid == 4U);
}
