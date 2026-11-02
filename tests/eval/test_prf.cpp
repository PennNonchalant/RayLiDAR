#include <raylidar/eval/metrics.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
using raylidar::eval::ComputePrfMetrics;
using raylidar::eval::PrfMetrics;
using Points = std::vector<Eigen::Vector3d>;
} // namespace

TEST_CASE("P/R/F use true Euclidean distance at all three taus", "[eval][prf]")
{
    const Points gt{{0.0, 0.0, 0.0}};
    const Points predicted{{0.04, 0.0, 0.0}, {0.19, 0.0, 0.0}, {0.49, 0.0, 0.0}};

    const PrfMetrics atFive = ComputePrfMetrics(predicted, gt, 0.05, false);
    REQUIRE(atFive.precision.has_value());
    REQUIRE(atFive.recall.has_value());
    REQUIRE(atFive.f_score.has_value());
    CHECK(*atFive.precision == Catch::Approx(1.0 / 3.0).margin(1.0e-12));
    CHECK(*atFive.recall == Catch::Approx(1.0).margin(1.0e-12));
    CHECK(*atFive.f_score == Catch::Approx(2.0 * (1.0 / 3.0) / (1.0 + 1.0 / 3.0)).margin(1.0e-12));

    const PrfMetrics atTwenty = ComputePrfMetrics(predicted, gt, 0.20, false);
    CHECK(*atTwenty.precision == Catch::Approx(2.0 / 3.0).margin(1.0e-12));
    CHECK(*atTwenty.recall == Catch::Approx(1.0).margin(1.0e-12));

    const PrfMetrics atFifty = ComputePrfMetrics(predicted, gt, 0.50, false);
    CHECK(*atFifty.precision == Catch::Approx(1.0).margin(1.0e-12));
    CHECK(*atFifty.recall == Catch::Approx(1.0).margin(1.0e-12));

    CHECK(raylidar::eval::kTausM[0] == Catch::Approx(0.05));
    CHECK(raylidar::eval::kTausM[1] == Catch::Approx(0.20));
    CHECK(raylidar::eval::kTausM[2] == Catch::Approx(0.50));
    CHECK(raylidar::eval::kPrimaryTauM == Catch::Approx(0.20));
}

TEST_CASE("the tau boundary is inclusive and tau + 1e-9 is not", "[eval][prf]")
{
    const Points gt{{0.0, 0.0, 0.0}};
    const Points exactly{{0.5, 0.0, 0.0}};
    const Points justOutside{{0.5 + 1.0e-9, 0.0, 0.0}};

    const PrfMetrics inclusive = ComputePrfMetrics(exactly, gt, 0.5, false);
    REQUIRE(inclusive.precision.has_value());
    CHECK(*inclusive.precision == Catch::Approx(1.0).margin(0.0));

    const PrfMetrics exclusive = ComputePrfMetrics(justOutside, gt, 0.5, false);
    REQUIRE(exclusive.precision.has_value());
    CHECK(*exclusive.precision == Catch::Approx(0.0).margin(0.0));
}

TEST_CASE("one-to-one greedy differs from many-to-one nearest neighbour", "[eval][prf]")
{
    const Points gt{{0.0, 0.0, 0.0}};
    const Points predicted{{0.1, 0.0, 0.0}, {0.0, 0.1, 0.0}};

    const PrfMetrics nearest = ComputePrfMetrics(predicted, gt, 0.2, false);
    REQUIRE(nearest.precision.has_value());
    REQUIRE(nearest.recall.has_value());
    CHECK(*nearest.precision == Catch::Approx(1.0).margin(1.0e-12));
    CHECK(*nearest.recall == Catch::Approx(1.0).margin(1.0e-12));

    const PrfMetrics greedy = ComputePrfMetrics(predicted, gt, 0.2, true);
    REQUIRE(greedy.precision.has_value());
    REQUIRE(greedy.recall.has_value());
    CHECK(*greedy.precision == Catch::Approx(0.5).margin(1.0e-12));
    CHECK(*greedy.recall == Catch::Approx(1.0).margin(1.0e-12));
    CHECK(*greedy.f_score == Catch::Approx(2.0 * 0.5 * 1.0 / 1.5).margin(1.0e-12));
    CHECK_FALSE(greedy.greedy_truncated);

    const PrfMetrics greedyAgain = ComputePrfMetrics(predicted, gt, 0.2, true);
    CHECK(greedyAgain.tp_pred == greedy.tp_pred);
    CHECK(greedyAgain.tp_gt == greedy.tp_gt);
}

TEST_CASE("empty prediction makes precision/F undefined but recall zero", "[eval][prf]")
{
    const Points empty;
    const Points gt{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    const PrfMetrics metrics = ComputePrfMetrics(empty, gt, 0.2, false);
    CHECK(metrics.n_pred == 0U);
    CHECK(metrics.n_gt == 2U);
    CHECK_FALSE(metrics.precision.has_value());
    REQUIRE(metrics.recall.has_value());
    CHECK(*metrics.recall == Catch::Approx(0.0).margin(0.0));
    CHECK_FALSE(metrics.f_score.has_value());
}

TEST_CASE("empty GT makes recall undefined and precision zero", "[eval][prf]")
{
    const Points predicted{{0.0, 0.0, 0.0}};
    const Points empty;
    const PrfMetrics metrics = ComputePrfMetrics(predicted, empty, 0.2, false);
    REQUIRE(metrics.precision.has_value());
    CHECK(*metrics.precision == Catch::Approx(0.0).margin(0.0));
    CHECK_FALSE(metrics.recall.has_value());
    CHECK_FALSE(metrics.f_score.has_value());
}

TEST_CASE("P/R/F rejects invalid tau", "[eval][prf]")
{
    const Points cloud{{0.0, 0.0, 0.0}};
    CHECK_THROWS_AS(ComputePrfMetrics(cloud, cloud, -0.1, false), std::invalid_argument);
    CHECK_THROWS_AS(ComputePrfMetrics(cloud, cloud, std::numeric_limits<double>::quiet_NaN(), false),
                    std::invalid_argument);
    CHECK_THROWS_AS(ComputePrfMetrics(cloud, cloud, std::numeric_limits<double>::infinity(), false),
                    std::invalid_argument);
}
