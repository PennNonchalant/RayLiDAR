#include <raylidar/eval/metrics.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

namespace
{
using raylidar::eval::ChamferMetrics;
using raylidar::eval::ComputeChamferMetrics;
using Points = std::vector<Eigen::Vector3d>;
} // namespace

TEST_CASE("Chamfer matches hand-computed asymmetric 1-D cases", "[eval][chamfer]")
{
    const Points predicted{{0.0, 0.0, 0.0}, {10.0, 0.0, 0.0}};
    const Points gt{{4.0, 0.0, 0.0}};

    const ChamferMetrics metrics = ComputeChamferMetrics(predicted, gt);
    CHECK(metrics.n_pred == 2U);
    CHECK(metrics.n_gt == 1U);
    REQUIRE(metrics.accuracy_l1.has_value());
    REQUIRE(metrics.completeness_l1.has_value());
    REQUIRE(metrics.cd_l1.has_value());
    REQUIRE(metrics.accuracy_l2.has_value());
    REQUIRE(metrics.completeness_l2.has_value());
    REQUIRE(metrics.cd_l2.has_value());

    CHECK(*metrics.accuracy_l1 == Catch::Approx(5.0).margin(1.0e-12));
    CHECK(*metrics.completeness_l1 == Catch::Approx(4.0).margin(1.0e-12));
    CHECK(*metrics.cd_l1 == Catch::Approx(4.5).margin(1.0e-12));
    CHECK(*metrics.accuracy_l2 == Catch::Approx(26.0).margin(1.0e-12));
    CHECK(*metrics.completeness_l2 == Catch::Approx(16.0).margin(1.0e-12));
    CHECK(*metrics.cd_l2 == Catch::Approx(21.0).margin(1.0e-12));
}

TEST_CASE("Chamfer uses Euclidean distance, not coordinate L1", "[eval][chamfer]")
{
    const Points predicted{{3.0, 4.0, 0.0}};
    const Points gt{{0.0, 0.0, 0.0}};
    const ChamferMetrics metrics = ComputeChamferMetrics(predicted, gt);
    REQUIRE(metrics.cd_l1.has_value());
    REQUIRE(metrics.cd_l2.has_value());
    CHECK(*metrics.cd_l1 == Catch::Approx(5.0).margin(1.0e-12));
    CHECK(*metrics.cd_l2 == Catch::Approx(25.0).margin(1.0e-12));
}

TEST_CASE("identical clouds give zero Chamfer and unit F-score", "[eval][chamfer]")
{
    const Points cloud{{0.0, 0.0, 0.0}, {1.0, 2.0, 3.0}, {-4.0, 5.0, 0.5}};
    const ChamferMetrics metrics = ComputeChamferMetrics(cloud, cloud);
    REQUIRE(metrics.cd_l1.has_value());
    REQUIRE(metrics.cd_l2.has_value());
    CHECK(*metrics.cd_l1 == Catch::Approx(0.0).margin(1.0e-12));
    CHECK(*metrics.cd_l2 == Catch::Approx(0.0).margin(1.0e-12));
    CHECK(*metrics.accuracy_l1 == Catch::Approx(0.0).margin(1.0e-12));
    CHECK(*metrics.completeness_l1 == Catch::Approx(0.0).margin(1.0e-12));
}

TEST_CASE("an empty cloud leaves Chamfer undefined but keeps counts", "[eval][chamfer]")
{
    const Points empty;
    const Points gt{{1.0, 1.0, 1.0}};
    const ChamferMetrics metrics = ComputeChamferMetrics(empty, gt);
    CHECK(metrics.n_pred == 0U);
    CHECK(metrics.n_gt == 1U);
    CHECK_FALSE(metrics.cd_l1.has_value());
    CHECK_FALSE(metrics.cd_l2.has_value());
    CHECK_FALSE(metrics.accuracy_l1.has_value());
    CHECK_FALSE(metrics.completeness_l1.has_value());
}
