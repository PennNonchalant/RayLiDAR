#include "fixture.h"

#include <raylidar/eval/aggregate.h>
#include <raylidar/eval/metrics.h>
#include <raylidar/eval/types.h>
#include <raylidar/rays/rays.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
using raylidar::eval::AggregateCell;
using raylidar::eval::ComputeBucketMetrics;
using raylidar::eval::RawCounts;
using raylidar::eval::RaySample;
using raylidar::rays::Bucket;
using raylidar::rays::BucketOf;
using raylidar::test::MakeSample;
} // namespace

TEST_CASE("BucketOf assigns the frozen bucket edges", "[eval][buckets]")
{
    CHECK(BucketOf(0.0) == Bucket::B0_20);
    CHECK(BucketOf(19.999) == Bucket::B0_20);
    CHECK(BucketOf(20.0) == Bucket::B20_40);
    CHECK(BucketOf(39.999) == Bucket::B20_40);
    CHECK(BucketOf(40.0) == Bucket::B40_50);
    CHECK(BucketOf(50.0) == Bucket::B40_50);
    CHECK(BucketOf(50.001) == Bucket::B50_60);
    CHECK(BucketOf(60.0) == Bucket::B50_60);

    CHECK_THROWS_AS(BucketOf(-1.0), std::invalid_argument);
    CHECK_THROWS_AS(BucketOf(60.0001), std::invalid_argument);
    CHECK_THROWS_AS(BucketOf(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
}

TEST_CASE("every bucket reports a count including empty buckets", "[eval][buckets]")
{
    std::vector<RaySample> samples;
    samples.push_back(MakeSample(0U, 0U, 0U, 10.0, 10.0));
    samples.push_back(MakeSample(1U, 0U, 0U, 15.0, std::numeric_limits<double>::infinity()));

    RawCounts raw;
    raw.raw_total = 4U;
    raw.raw_bucket = {3U, 0U, 0U, 1U};
    const auto buckets = ComputeBucketMetrics(samples, raw);
    REQUIRE(buckets.size() == 4U);

    CHECK(buckets[0].n_valid == 2U);
    CHECK(buckets[0].n_hits == 1U);
    CHECK(buckets[0].n_misses == 1U);
    REQUIRE(buckets[0].coverage.has_value());
    CHECK(*buckets[0].coverage == Catch::Approx(2.0 / 3.0).margin(1.0e-12));

    for (std::size_t i = 1; i < 3U; ++i)
    {
        CHECK(buckets[i].n_valid == 0U);
        CHECK(buckets[i].n_hits == 0U);
        CHECK(buckets[i].n_misses == 0U);
        CHECK_FALSE(buckets[i].coverage.has_value());
        CHECK_FALSE(buckets[i].range.mae.has_value());
    }

    CHECK(buckets[3].n_valid == 0U);
    REQUIRE(buckets[3].coverage.has_value());
    CHECK(*buckets[3].coverage == Catch::Approx(0.0).margin(0.0));
}

TEST_CASE("primary micro excludes the secondary 50_60 bucket", "[eval][buckets]")
{
    const RaySample near = MakeSample(0U, 0U, 0U, 10.0, 10.1);
    const RaySample far = MakeSample(1U, 0U, 0U, 55.0, 55.1);

    RawCounts raw;
    raw.raw_total = 2U;
    raw.raw_bucket = {1U, 0U, 0U, 1U};

    const std::vector<RaySample> primary{near};
    const auto microPrimary = AggregateCell(primary, raw);
    CHECK(microPrimary.n_valid == 1U);

    const auto buckets = ComputeBucketMetrics(std::vector<RaySample>{near, far}, raw);
    CHECK(buckets[0].n_valid == 1U);
    CHECK(buckets[3].n_valid == 1U);
    REQUIRE(buckets[3].range.mae.has_value());
    CHECK(*buckets[3].range.mae == Catch::Approx(0.1).margin(1.0e-12));
}
