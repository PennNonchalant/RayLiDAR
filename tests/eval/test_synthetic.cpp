#include "fixture.h"

#include <raylidar/eval/aggregate.h>
#include <raylidar/eval/metrics.h>
#include <raylidar/eval/types.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <tuple>
#include <vector>

namespace
{
using raylidar::eval::AggregateCell;
using raylidar::eval::CellMetrics;
using raylidar::eval::ChamferMetrics;
using raylidar::eval::ComputeChamferMetrics;
using raylidar::eval::ComputePrfMetrics;
using raylidar::eval::EvalReport;
using raylidar::eval::EvaluateAll;
using raylidar::eval::FrameRawCounts;
using raylidar::eval::GtPoints;
using raylidar::eval::MacroSummary;
using raylidar::eval::PredictedPoints;
using raylidar::eval::PrfMetrics;
using raylidar::eval::RawCounts;
using raylidar::eval::RaySample;
using raylidar::test::MakeSample;

void RequireFinite(const std::optional<double> &value)
{
    if (value.has_value())
    {
        REQUIRE(std::isfinite(*value));
    }
}

void ScanCell(const CellMetrics &cell)
{
    RequireFinite(cell.coverage);
    RequireFinite(cell.range.mae);
    RequireFinite(cell.range.rmse);
    RequireFinite(cell.range.medae);
    RequireFinite(cell.range.bias);
    RequireFinite(cell.range.p90);
    RequireFinite(cell.range.p95);
    RequireFinite(cell.miss.miss_rate);
    RequireFinite(cell.miss.ray_recall);
}

void ScanPrf(const PrfMetrics &prf)
{
    RequireFinite(prf.precision);
    RequireFinite(prf.recall);
    RequireFinite(prf.f_score);
}

void ScanChamfer(const ChamferMetrics &cd)
{
    RequireFinite(cd.cd_l1);
    RequireFinite(cd.cd_l2);
    RequireFinite(cd.accuracy_l1);
    RequireFinite(cd.completeness_l1);
    RequireFinite(cd.accuracy_l2);
    RequireFinite(cd.completeness_l2);
}

void ScanMacro(const MacroSummary &macro)
{
    RequireFinite(macro.mae_mean);
    RequireFinite(macro.mae_std);
    RequireFinite(macro.rmse_mean);
    RequireFinite(macro.rmse_std);
    RequireFinite(macro.medae_mean);
    RequireFinite(macro.medae_std);
    RequireFinite(macro.bias_mean);
    RequireFinite(macro.bias_std);
    RequireFinite(macro.p90_mean);
    RequireFinite(macro.p90_std);
    RequireFinite(macro.p95_mean);
    RequireFinite(macro.p95_std);
    RequireFinite(macro.miss_rate_mean);
    RequireFinite(macro.miss_rate_std);
    RequireFinite(macro.ray_recall_mean);
    RequireFinite(macro.ray_recall_std);
    RequireFinite(macro.coverage_mean);
    RequireFinite(macro.coverage_std);
    RequireFinite(macro.precision_mean);
    RequireFinite(macro.precision_std);
    RequireFinite(macro.recall_mean);
    RequireFinite(macro.recall_std);
    RequireFinite(macro.f_score_mean);
    RequireFinite(macro.f_score_std);
}

void ScanReport(const EvalReport &report)
{
    ScanCell(report.micro_primary);
    ScanCell(report.micro_secondary);
    for (const auto &frame : report.per_frame)
    {
        ScanCell(frame.primary);
        for (const auto &bucket : frame.buckets)
        {
            ScanCell(bucket);
        }
    }
    for (const auto &camera : report.per_camera)
    {
        ScanCell(camera.primary);
    }
    ScanMacro(report.macro);
    for (const auto &row : report.rows)
    {
        ScanCell(row.cell);
        for (const auto &prf : row.prf)
        {
            ScanPrf(prf.nearest);
            ScanPrf(prf.greedy);
        }
        ScanChamfer(row.cd);
    }
}
} // namespace

TEST_CASE("identity predictions give zero error, zero CD, and unit F", "[eval][synthetic]")
{
    const std::vector<double> ranges{5.0, 10.0, 25.0};
    std::vector<RaySample> samples;
    std::uint64_t rayId = 0;
    for (const double range : ranges)
    {
        samples.push_back(MakeSample(rayId++, 10U, 0U, range, range, Eigen::Vector3d(1.0, 2.0, 3.0)));
    }

    const std::vector<Eigen::Vector3d> predicted = PredictedPoints(samples);
    const std::vector<Eigen::Vector3d> gt = GtPoints(samples);
    REQUIRE(predicted.size() == gt.size());

    const ChamferMetrics cd = ComputeChamferMetrics(predicted, gt);
    REQUIRE(cd.cd_l1.has_value());
    CHECK(*cd.cd_l1 == Catch::Approx(0.0).margin(1.0e-12));
    CHECK(*cd.cd_l2 == Catch::Approx(0.0).margin(1.0e-12));
    for (const double tau : raylidar::eval::kTausM)
    {
        const PrfMetrics prf = ComputePrfMetrics(predicted, gt, tau, false);
        REQUIRE(prf.precision.has_value());
        REQUIRE(prf.recall.has_value());
        REQUIRE(prf.f_score.has_value());
        CHECK(*prf.precision == Catch::Approx(1.0).margin(1.0e-12));
        CHECK(*prf.recall == Catch::Approx(1.0).margin(1.0e-12));
        CHECK(*prf.f_score == Catch::Approx(1.0).margin(1.0e-12));
    }

    RawCounts raw;
    raw.raw_total = 3U;
    raw.raw_bucket = {2U, 1U, 0U, 0U};
    const EvalReport report = EvaluateAll(samples, {FrameRawCounts{10U, raw}});
    REQUIRE(report.micro_primary.range.mae.has_value());
    CHECK(*report.micro_primary.range.mae == Catch::Approx(0.0).margin(1.0e-12));
    CHECK(report.micro_primary.n_valid == 3U);
    CHECK(report.micro_primary.n_hits == 3U);
    REQUIRE(report.micro_primary.coverage.has_value());
    CHECK(*report.micro_primary.coverage == Catch::Approx(1.0).margin(1.0e-12));
    ScanReport(report);
}

TEST_CASE("an empty prediction scene is a 100 percent miss with null means", "[eval][synthetic]")
{
    std::vector<RaySample> samples;
    std::uint64_t rayId = 0;
    for (const double range : {5.0, 10.0, 25.0})
    {
        samples.push_back(MakeSample(rayId++, 20U, 0U, range, std::numeric_limits<double>::infinity()));
    }

    RawCounts raw;
    raw.raw_total = 3U;
    raw.raw_bucket = {2U, 1U, 0U, 0U};
    const EvalReport report = EvaluateAll(samples, {FrameRawCounts{20U, raw}});

    REQUIRE(report.micro_primary.miss.miss_rate.has_value());
    REQUIRE(report.micro_primary.miss.ray_recall.has_value());
    CHECK(*report.micro_primary.miss.miss_rate == Catch::Approx(1.0).margin(0.0));
    CHECK(*report.micro_primary.miss.ray_recall == Catch::Approx(0.0).margin(0.0));
    CHECK_FALSE(report.micro_primary.range.mae.has_value());
    CHECK_FALSE(report.macro.mae_mean.has_value());

    const auto predicted = PredictedPoints(samples);
    const auto gt = GtPoints(samples);
    CHECK(predicted.empty());
    CHECK(gt.size() == 3U);

    const ChamferMetrics cd = ComputeChamferMetrics(predicted, gt);
    CHECK_FALSE(cd.cd_l1.has_value());
    const PrfMetrics prf = ComputePrfMetrics(predicted, gt, raylidar::eval::kPrimaryTauM, false);
    CHECK_FALSE(prf.precision.has_value());
    REQUIRE(prf.recall.has_value());
    CHECK(*prf.recall == Catch::Approx(0.0).margin(0.0));
    CHECK_FALSE(prf.f_score.has_value());

    ScanReport(report);
}

TEST_CASE("EvaluateAll emits a complete frame x camera x bucket grid", "[eval][synthetic]")
{
    std::vector<RaySample> samples;
    samples.push_back(MakeSample(0U, 10U, 0U, 10.0, 10.1));
    samples.push_back(MakeSample(1U, 10U, 1U, 25.0, 25.1));
    samples.push_back(MakeSample(2U, 11U, 0U, 55.0, 55.1));

    FrameRawCounts frame10;
    frame10.frame = 10U;
    frame10.raw.raw_total = 2U;
    frame10.raw.raw_bucket = {1U, 1U, 0U, 0U};
    // Per-camera raw denominators: one cast per camera in frame 10.
    frame10.per_camera[0].raw_total = 1U;
    frame10.per_camera[0].raw_bucket = {1U, 0U, 0U, 0U};
    frame10.per_camera[1].raw_total = 1U;
    frame10.per_camera[1].raw_bucket = {0U, 1U, 0U, 0U};
    FrameRawCounts frame11;
    frame11.frame = 11U;
    frame11.raw.raw_total = 1U;
    frame11.raw.raw_bucket = {0U, 0U, 0U, 1U};
    frame11.per_camera[0].raw_total = 1U;
    frame11.per_camera[0].raw_bucket = {0U, 0U, 0U, 1U};

    const std::vector<FrameRawCounts> rawByFrame{frame10, frame11};
    const EvalReport report = EvaluateAll(samples, rawByFrame);

    REQUIRE(report.rows.size() == 16U);

    std::set<std::tuple<std::uint32_t, std::uint16_t, int>> seen;
    for (const auto &row : report.rows)
    {
        seen.emplace(row.frame, row.camera, static_cast<int>(row.bucket));
    }
    for (const std::uint32_t frame : {10U, 11U})
    {
        for (const std::uint16_t camera : {0U, 1U})
        {
            for (int bucket = 0; bucket < 4; ++bucket)
            {
                CHECK(seen.count({frame, camera, bucket}) == 1U);
            }
        }
    }

    for (const auto &row : report.rows)
    {
        if (row.frame == 10U && row.camera == 0U && row.bucket == raylidar::rays::Bucket::B0_20)
        {
            CHECK(row.cell.n_valid == 1U);
            CHECK(row.cell.n_hits == 1U);
            CHECK(row.cell.n_raw == 1U);
            REQUIRE(row.cell.coverage.has_value());
            CHECK(*row.cell.coverage == Catch::Approx(1.0).margin(1.0e-12));
            REQUIRE(row.cd.cd_l1.has_value());
            CHECK(*row.cd.cd_l1 == Catch::Approx(0.1).margin(1.0e-12));
            REQUIRE(row.prf[1].nearest.f_score.has_value());
            CHECK(*row.prf[1].nearest.f_score == Catch::Approx(1.0).margin(1.0e-12));
        }
        if (row.frame == 10U && row.camera == 0U && row.bucket == raylidar::rays::Bucket::B20_40)
        {
            // A populated row and an empty row are distinguishable: the empty
            // bucket has no raw denominator, so its coverage is an explicit null.
            CHECK(row.cell.n_raw == 0U);
            CHECK(row.cell.n_valid == 0U);
            CHECK_FALSE(row.cell.coverage.has_value());
        }
        if (row.frame == 11U && row.camera == 1U && row.bucket == raylidar::rays::Bucket::B50_60)
        {
            CHECK(row.cell.n_valid == 0U);
            CHECK_FALSE(row.cell.range.mae.has_value());
            CHECK_FALSE(row.cell.coverage.has_value());
            CHECK_FALSE(row.cd.cd_l1.has_value());
            CHECK_FALSE(row.prf[0].nearest.precision.has_value());
        }
    }

    // Primary micro excludes the 55 m secondary ray but its denominator is every
    // raw cast: frame 10 casts 2 + frame 11 casts 1 = 3 for 2 primary valid rays.
    REQUIRE(report.micro_primary.coverage.has_value());
    CHECK(report.micro_primary.n_valid == 2U);
    CHECK(report.micro_primary.n_raw == 3U);
    CHECK(*report.micro_primary.coverage == Catch::Approx(2.0 / 3.0).margin(1.0e-12));
    REQUIRE(report.micro_secondary.coverage.has_value());
    CHECK(report.micro_secondary.n_valid == 1U);
    CHECK(*report.micro_secondary.coverage == Catch::Approx(1.0).margin(1.0e-12));

    RawCounts primaryRaw;
    primaryRaw.raw_total = 3U;
    primaryRaw.raw_bucket = {1U, 1U, 0U, 0U};
    const auto pooledPrimary = AggregateCell(std::span<const RaySample>(samples.data(), 2U), primaryRaw);
    REQUIRE(report.micro_primary.range.mae.has_value());
    REQUIRE(pooledPrimary.range.mae.has_value());
    CHECK(*report.micro_primary.range.mae == Catch::Approx(*pooledPrimary.range.mae).margin(1.0e-12));
    CHECK(report.micro_primary.n_hits == pooledPrimary.n_hits);
    CHECK(report.micro_primary.n_valid == pooledPrimary.n_valid);

    REQUIRE(report.macro.precision_mean.has_value());
    CHECK(*report.macro.precision_mean == Catch::Approx(1.0).margin(1.0e-12));
    CHECK_FALSE(report.macro.precision_std.has_value());

    ScanReport(report);
}

TEST_CASE("coverage divides by every raw cast, not by surviving bucket hits", "[eval][synthetic]")
{
    std::vector<RaySample> samples;
    samples.push_back(MakeSample(0U, 10U, 0U, 10.0, 10.0));
    samples.push_back(MakeSample(1U, 11U, 0U, 12.0, 12.0));

    // Frame 10 casts two rays but only one produced a bucket hit; frame 11 casts
    // one. The bucket-hit population is 2, the every-raw-cast population is 3.
    RawCounts frame10;
    frame10.raw_total = 2U;
    frame10.raw_bucket = {1U, 0U, 0U, 0U};
    RawCounts frame11;
    frame11.raw_total = 1U;
    frame11.raw_bucket = {1U, 0U, 0U, 0U};

    const EvalReport report = EvaluateAll(samples, {FrameRawCounts{10U, frame10}, FrameRawCounts{11U, frame11}});

    REQUIRE(report.micro_primary.coverage.has_value());
    CHECK(report.micro_primary.n_raw == 3U);
    CHECK(report.micro_primary.n_valid == 2U);
    CHECK(*report.micro_primary.coverage == Catch::Approx(2.0 / 3.0).margin(1.0e-12));

    // Per-frame primary coverage uses that frame's raw_total as well.
    REQUIRE(report.per_frame.size() == 2U);
    REQUIRE(report.per_frame[0].primary.coverage.has_value());
    CHECK(*report.per_frame[0].primary.coverage == Catch::Approx(0.5).margin(1.0e-12));
    REQUIRE(report.per_frame[1].primary.coverage.has_value());
    CHECK(*report.per_frame[1].primary.coverage == Catch::Approx(1.0).margin(1.0e-12));

    // Bucket cells keep the pre-filter per-bucket population.
    REQUIRE(report.per_frame[0].buckets[0].coverage.has_value());
    CHECK(*report.per_frame[0].buckets[0].coverage == Catch::Approx(1.0).margin(1.0e-12));

    // Macro coverage is the mean of the per-frame primary coverages.
    REQUIRE(report.macro.coverage_mean.has_value());
    CHECK(*report.macro.coverage_mean == Catch::Approx(0.75).margin(1.0e-12));
    ScanReport(report);
}
