#include "fixture.h"

#include <raylidar/eval/metrics.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
using raylidar::eval::ComputeRangeMetrics;
using raylidar::eval::RangeMetrics;
using raylidar::eval::RaySample;
using raylidar::eval::ValidateSamples;
using raylidar::test::MakeSample;

RaySample RawSample()
{
    return MakeSample(1U, 0U, 0U, 10.0, 10.0);
}
} // namespace

TEST_CASE("range metrics match hand-computed values", "[eval][range]")
{
    constexpr double kGt = 10.0;
    std::vector<RaySample> samples;
    std::uint64_t rayId = 0;
    for (const double error : {-0.5, 0.1, 0.2, 2.0, -1.0})
    {
        samples.push_back(MakeSample(rayId++, 0U, 0U, kGt, kGt + error));
    }

    const RangeMetrics metrics = ComputeRangeMetrics(samples);
    REQUIRE(metrics.n_hits == 5U);
    REQUIRE(metrics.mae.has_value());
    REQUIRE(metrics.rmse.has_value());
    REQUIRE(metrics.medae.has_value());
    REQUIRE(metrics.bias.has_value());
    REQUIRE(metrics.p90.has_value());
    REQUIRE(metrics.p95.has_value());

    CHECK(*metrics.mae == Catch::Approx(0.76).margin(1.0e-12));
    CHECK(*metrics.rmse == Catch::Approx(std::sqrt(1.06)).margin(1.0e-12));
    CHECK(*metrics.medae == Catch::Approx(0.5).margin(1.0e-12));
    CHECK(*metrics.bias == Catch::Approx(0.16).margin(1.0e-12));
    // Nearest-rank: ceil(0.90 * 5) - 1 = index 4, ceil(0.95 * 5) - 1 = index 4.
    CHECK(*metrics.p90 == Catch::Approx(2.0).margin(1.0e-12));
    CHECK(*metrics.p95 == Catch::Approx(2.0).margin(1.0e-12));
}

TEST_CASE("range metrics are nullopt without hits and defined with one hit", "[eval][range]")
{
    std::vector<RaySample> misses;
    misses.push_back(MakeSample(0U, 0U, 0U, 10.0, std::numeric_limits<double>::infinity()));
    misses.push_back(MakeSample(1U, 0U, 0U, 20.0, std::numeric_limits<double>::infinity()));

    const RangeMetrics empty = ComputeRangeMetrics(misses);
    CHECK(empty.n_hits == 0U);
    CHECK_FALSE(empty.mae.has_value());
    CHECK_FALSE(empty.rmse.has_value());
    CHECK_FALSE(empty.medae.has_value());
    CHECK_FALSE(empty.bias.has_value());
    CHECK_FALSE(empty.p90.has_value());
    CHECK_FALSE(empty.p95.has_value());

    std::vector<RaySample> single;
    single.push_back(MakeSample(2U, 0U, 0U, 10.0, 9.75));
    const RangeMetrics one = ComputeRangeMetrics(single);
    REQUIRE(one.n_hits == 1U);
    REQUIRE(one.mae.has_value());
    CHECK(*one.mae == Catch::Approx(0.25).margin(1.0e-12));
    CHECK(*one.rmse == Catch::Approx(0.25).margin(1.0e-12));
    CHECK(*one.medae == Catch::Approx(0.25).margin(1.0e-12));
    CHECK(*one.bias == Catch::Approx(-0.25).margin(1.0e-12));
    CHECK(*one.p90 == Catch::Approx(0.25).margin(1.0e-12));
    CHECK(*one.p95 == Catch::Approx(0.25).margin(1.0e-12));
}

TEST_CASE("even-sized MedAE averages the two middle absolute errors", "[eval][range]")
{
    std::vector<RaySample> samples;
    std::uint64_t rayId = 0;
    for (const double error : {1.0, 2.0, 3.0, 4.0})
    {
        samples.push_back(MakeSample(rayId++, 0U, 0U, 10.0, 10.0 + error));
    }

    const RangeMetrics metrics = ComputeRangeMetrics(samples);
    REQUIRE(metrics.medae.has_value());
    CHECK(*metrics.medae == Catch::Approx(2.5).margin(1.0e-12));
}

TEST_CASE("ValidateSamples rejects malformed samples", "[eval][range]")
{
    const std::vector<RaySample> valid{MakeSample(1U, 0U, 0U, 10.0, 9.9)};
    CHECK_NOTHROW(ValidateSamples(valid));

    RaySample sample = RawSample();
    sample.t_gt = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(ValidateSamples(std::vector<RaySample>{sample}), std::invalid_argument);

    sample = RawSample();
    sample.t_min = std::numeric_limits<double>::infinity();
    CHECK_THROWS_AS(ValidateSamples(std::vector<RaySample>{sample}), std::invalid_argument);

    sample = RawSample();
    sample.t_max = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(ValidateSamples(std::vector<RaySample>{sample}), std::invalid_argument);

    sample = RawSample();
    sample.dir_W = Eigen::Vector3d(0.0, 0.0, 2.0);
    CHECK_THROWS_AS(ValidateSamples(std::vector<RaySample>{sample}), std::invalid_argument);

    sample = RawSample();
    sample.t_gt = 100.0;
    CHECK_THROWS_AS(ValidateSamples(std::vector<RaySample>{sample}), std::invalid_argument);

    sample = RawSample();
    sample.camera = 2U;
    CHECK_THROWS_AS(ValidateSamples(std::vector<RaySample>{sample}), std::invalid_argument);

    sample = RawSample();
    sample.t_gt = 25.0;
    sample.bucket = raylidar::rays::Bucket::B0_20;
    CHECK_THROWS_AS(ValidateSamples(std::vector<RaySample>{sample}), std::invalid_argument);

    sample = RawSample();
    sample.t_pred = 0.0;
    CHECK_THROWS_AS(ValidateSamples(std::vector<RaySample>{sample}), std::invalid_argument);

    sample = RawSample();
    sample.t_pred = -1.0;
    CHECK_THROWS_AS(ValidateSamples(std::vector<RaySample>{sample}), std::invalid_argument);

    sample = RawSample();
    sample.t_pred = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(ValidateSamples(std::vector<RaySample>{sample}), std::invalid_argument);

    sample = RawSample();
    sample.t_pred = std::numeric_limits<double>::infinity();
    CHECK_NOTHROW(ValidateSamples(std::vector<RaySample>{sample}));
}
