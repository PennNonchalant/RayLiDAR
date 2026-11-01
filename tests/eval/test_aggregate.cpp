#include "fixture.h"

#include <raylidar/eval/aggregate.h>
#include <raylidar/eval/metrics.h>
#include <raylidar/eval/types.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace
{
using raylidar::eval::AggregateCell;
using raylidar::eval::AggregatePerCamera;
using raylidar::eval::AggregatePerFrame;
using raylidar::eval::FrameRawCounts;
using raylidar::eval::RawCounts;
using raylidar::eval::RaySample;
using raylidar::eval::SummarizeMacro;
using raylidar::test::MakeSample;

RawCounts Frame10Raw()
{
    RawCounts raw;
    raw.raw_total = 4U;
    raw.raw_bucket = {2U, 2U, 0U, 0U};
    return raw;
}
} // namespace

TEST_CASE("micro pools rays while macro averages defined per-frame values", "[eval][aggregate]")
{
    std::vector<RaySample> samples;
    samples.push_back(MakeSample(0U, 10U, 0U, 10.0, 10.2));
    samples.push_back(MakeSample(1U, 10U, 1U, 20.0, 20.0));
    samples.push_back(MakeSample(2U, 11U, 0U, 20.0, 20.4));
    samples.push_back(MakeSample(3U, 12U, 0U, 20.0, std::numeric_limits<double>::infinity()));

    std::vector<FrameRawCounts> rawByFrame;
    rawByFrame.push_back(FrameRawCounts{10U, Frame10Raw()});
    RawCounts frame11;
    frame11.raw_total = 2U;
    frame11.raw_bucket = {0U, 2U, 0U, 0U};
    rawByFrame.push_back(FrameRawCounts{11U, frame11});
    RawCounts frame12;
    frame12.raw_total = 1U;
    frame12.raw_bucket = {0U, 1U, 0U, 0U};
    rawByFrame.push_back(FrameRawCounts{12U, frame12});

    RawCounts pooled;
    pooled.raw_total = 7U;
    pooled.raw_bucket = {2U, 5U, 0U, 0U};
    const auto micro = AggregateCell(samples, pooled);
    REQUIRE(micro.range.n_hits == 3U);
    REQUIRE(micro.range.mae.has_value());
    CHECK(*micro.range.mae == Catch::Approx(0.2).margin(1.0e-12));
    REQUIRE(micro.coverage.has_value());
    CHECK(*micro.coverage == Catch::Approx(4.0 / 7.0).margin(1.0e-12));

    const auto perFrame = AggregatePerFrame(samples, rawByFrame);
    REQUIRE(perFrame.size() == 3U);
    CHECK(perFrame[0].frame == 10U);
    REQUIRE(perFrame[0].primary.range.mae.has_value());
    CHECK(*perFrame[0].primary.range.mae == Catch::Approx(0.1).margin(1.0e-12));
    REQUIRE(perFrame[0].primary.coverage.has_value());
    CHECK(*perFrame[0].primary.coverage == Catch::Approx(0.5).margin(1.0e-12));

    REQUIRE(perFrame[1].primary.range.mae.has_value());
    CHECK(*perFrame[1].primary.range.mae == Catch::Approx(0.4).margin(1.0e-12));

    CHECK(perFrame[2].primary.n_hits == 0U);
    CHECK_FALSE(perFrame[2].primary.range.mae.has_value());
    REQUIRE(perFrame[2].primary.coverage.has_value());
    CHECK(*perFrame[2].primary.coverage == Catch::Approx(1.0).margin(1.0e-12));

    const auto macro = SummarizeMacro(perFrame);
    CHECK(macro.n_frames_total == 3U);
    CHECK(macro.n_frames_with_range == 2U);
    CHECK(macro.n_frames_without_range == 1U);

    REQUIRE(macro.mae_mean.has_value());
    REQUIRE(macro.mae_std.has_value());
    CHECK(*macro.mae_mean == Catch::Approx(0.25).margin(1.0e-12));
    // Sample standard deviation (n-1) over {0.1, 0.4}.
    CHECK(*macro.mae_std == Catch::Approx(std::sqrt(0.045)).margin(1.0e-12));

    REQUIRE(macro.miss_rate_mean.has_value());
    REQUIRE(macro.miss_rate_std.has_value());
    CHECK(*macro.miss_rate_mean == Catch::Approx(1.0 / 3.0).margin(1.0e-12));
    // Sample std over {0, 0, 1}.
    CHECK(*macro.miss_rate_std == Catch::Approx(std::sqrt(1.0 / 3.0)).margin(1.0e-12));
}

TEST_CASE("per-camera results split by camera and carry coverage or null", "[eval][aggregate]")
{
    std::vector<RaySample> samples;
    samples.push_back(MakeSample(0U, 0U, 0U, 10.0, 10.2));
    samples.push_back(MakeSample(1U, 0U, 0U, 20.0, 20.4));
    samples.push_back(MakeSample(2U, 0U, 0U, 20.0, std::numeric_limits<double>::infinity()));
    samples.push_back(MakeSample(3U, 0U, 1U, 20.0, 20.0));

    const auto perCamera = AggregatePerCamera(samples);
    REQUIRE(perCamera.size() == 2U);
    CHECK(perCamera[0].camera == 0U);
    CHECK(perCamera[1].camera == 1U);

    CHECK(perCamera[0].primary.n_valid == 3U);
    CHECK(perCamera[0].primary.n_hits == 2U);
    REQUIRE(perCamera[0].primary.range.mae.has_value());
    CHECK(*perCamera[0].primary.range.mae == Catch::Approx(0.3).margin(1.0e-12));
    CHECK_FALSE(perCamera[0].primary.coverage.has_value());

    CHECK(perCamera[1].primary.n_valid == 1U);
    REQUIRE(perCamera[1].primary.range.mae.has_value());
    CHECK(*perCamera[1].primary.range.mae == Catch::Approx(0.0).margin(0.0));
    CHECK_FALSE(perCamera[1].primary.coverage.has_value());
}

TEST_CASE("raw-aware per-camera aggregation defines coverage from raw_total",
          "[eval][aggregate]")
{
    std::vector<RaySample> samples;
    samples.push_back(MakeSample(0U, 0U, 0U, 5.0, 5.1));
    samples.push_back(MakeSample(1U, 0U, 0U, 15.0, 15.1));
    samples.push_back(MakeSample(2U, 0U, 1U, 20.0, 20.1));

    FrameRawCounts frame;
    frame.frame = 0U;
    // Camera 0 cast three rays but only two produced a bucket hit.
    frame.raw.per_camera[0].raw_total = 3U;
    frame.raw.per_camera[0].raw_bucket = {2U, 0U, 0U, 0U};
    frame.raw.per_camera[1].raw_total = 1U;
    frame.raw.per_camera[1].raw_bucket = {0U, 1U, 0U, 0U};

    const auto withRaw = AggregatePerCamera(samples, {frame});
    REQUIRE(withRaw.size() == 2U);
    REQUIRE(withRaw[0].primary.coverage.has_value());
    CHECK(withRaw[0].primary.n_raw == 3U);
    CHECK(*withRaw[0].primary.coverage == Catch::Approx(2.0 / 3.0).margin(1.0e-12));
    REQUIRE(withRaw[1].primary.coverage.has_value());
    CHECK(withRaw[1].primary.n_raw == 1U);
    CHECK(*withRaw[1].primary.coverage == Catch::Approx(1.0).margin(1.0e-12));

    // The old overload has no per-camera denominator and keeps the explicit null.
    const auto withoutRaw = AggregatePerCamera(samples);
    REQUIRE(withoutRaw.size() == 2U);
    CHECK_FALSE(withoutRaw[0].primary.coverage.has_value());
    CHECK_FALSE(withoutRaw[1].primary.coverage.has_value());
}
