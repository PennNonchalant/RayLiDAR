#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <raylidar/common/io.h>
#include <raylidar/eval/scene.h>
#include <raylidar/io/sim_io.h>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>

namespace
{
namespace fs = std::filesystem;
using Catch::Approx;
using raylidar::eval::ProceduralScene;
using raylidar::eval::ProceduralStreetConfig;
using raylidar::eval::RaycastAabb;
using raylidar::eval::RaycastPlane;
using raylidar::eval::RaycastScene;
using raylidar::eval::RaycastSphere;

struct TempDir
{
    TempDir()
    {
        static std::uint64_t counter = 0;
        static std::random_device device;
        path = fs::temp_directory_path() /
               ("raylidar-eval-scene-" + std::to_string(device()) + "-" +
                std::to_string(counter++));
        fs::create_directories(path);
    }
    ~TempDir()
    {
        std::error_code error;
        fs::remove_all(path, error);
    }
    fs::path path;
};

ProceduralStreetConfig TinyConfig()
{
    ProceduralStreetConfig config;
    config.seed = 20261101;
    config.ground_length_m = 20.0;
    config.ground_width_m = 10.0;
    config.box_count = 4;
    config.thin_box_count = 2;
    return config;
}
} // namespace

TEST_CASE("RaycastPlane is bounded, two-sided, and honors t_min/t_max", "[eval][scene]")
{
    const Eigen::Vector3d u(1.0, 0.0, 0.0);
    const Eigen::Vector3d v(0.0, 1.0, 0.0);
    const Eigen::Vector3d planeOrigin = Eigen::Vector3d::Zero();
    const Eigen::Vector3d above(0.0, 0.0, 1.0);
    const Eigen::Vector3d down(0.0, 0.0, -1.0);

    const auto hit = RaycastPlane(above, down, planeOrigin, u, v, 10.0, 10.0, 1.0e-4, 60.0);
    REQUIRE(hit.has_value());
    CHECK(*hit == Approx(1.0));

    CHECK_FALSE(RaycastPlane(above, Eigen::Vector3d(1.0, 0.0, 0.0), planeOrigin, u, v, 10.0,
                             10.0, 1.0e-4, 60.0)
                    .has_value());
    CHECK_FALSE(RaycastPlane(above, Eigen::Vector3d(0.0, 0.0, 1.0), planeOrigin, u, v, 10.0,
                             10.0, 1.0e-4, 60.0)
                    .has_value());
    CHECK_FALSE(RaycastPlane(Eigen::Vector3d(20.0, 20.0, 1.0), down, planeOrigin, u, v, 10.0,
                             10.0, 1.0e-4, 60.0)
                    .has_value());
    CHECK_FALSE(RaycastPlane(above, down, planeOrigin, u, v, 10.0, 10.0, 1.5, 60.0)
                    .has_value());
    CHECK_FALSE(RaycastPlane(above, down, planeOrigin, u, v, 10.0, 10.0, 1.0e-4, 0.5)
                    .has_value());
}

TEST_CASE("RaycastSphere returns the near/exit/tangent root and rejects misses",
          "[eval][scene]")
{
    const Eigen::Vector3d center = Eigen::Vector3d::Zero();
    const Eigen::Vector3d alongZ(0.0, 0.0, 1.0);

    const auto fromOutside =
        RaycastSphere(Eigen::Vector3d(0.0, 0.0, -5.0), alongZ, center, 1.0, 1.0e-4, 60.0);
    REQUIRE(fromOutside.has_value());
    CHECK(*fromOutside == Approx(4.0));

    const auto fromInside =
        RaycastSphere(Eigen::Vector3d::Zero(), alongZ, center, 1.0, 1.0e-4, 60.0);
    REQUIRE(fromInside.has_value());
    CHECK(*fromInside == Approx(1.0));

    const auto tangent =
        RaycastSphere(Eigen::Vector3d(0.0, 1.0, -5.0), alongZ, center, 1.0, 1.0e-4, 60.0);
    REQUIRE(tangent.has_value());
    CHECK(*tangent == Approx(5.0));

    CHECK_FALSE(RaycastSphere(Eigen::Vector3d(0.0, 2.0, -5.0), alongZ, center, 1.0, 1.0e-4,
                              60.0)
                    .has_value());
    CHECK_FALSE(RaycastSphere(Eigen::Vector3d(0.0, 0.0, -5.0), alongZ, center, 1.0, 4.5,
                              60.0)
                    .has_value());
    CHECK_FALSE(RaycastSphere(Eigen::Vector3d(0.0, 0.0, -5.0), alongZ, center, 1.0, 1.0e-4,
                              3.5)
                    .has_value());
}

TEST_CASE("RaycastAabb uses the slab method for entry/exit/inside/miss", "[eval][scene]")
{
    const Eigen::Vector3d boxMin(0.0, 0.0, 0.0);
    const Eigen::Vector3d boxMax(1.0, 1.0, 1.0);
    const Eigen::Vector3d alongX(1.0, 0.0, 0.0);

    const auto entry =
        RaycastAabb(Eigen::Vector3d(-1.0, 0.5, 0.5), alongX, boxMin, boxMax, 1.0e-4, 60.0);
    REQUIRE(entry.has_value());
    CHECK(*entry == Approx(1.0));

    const auto inside =
        RaycastAabb(Eigen::Vector3d(0.5, 0.5, 0.5), alongX, boxMin, boxMax, 1.0e-4, 60.0);
    REQUIRE(inside.has_value());
    CHECK(*inside == Approx(0.5));

    CHECK_FALSE(RaycastAabb(Eigen::Vector3d(2.0, 0.5, 0.5), Eigen::Vector3d(0.0, 1.0, 0.0),
                            boxMin, boxMax, 1.0e-4, 60.0)
                    .has_value());
    CHECK_FALSE(RaycastAabb(Eigen::Vector3d(-1.0, 0.5, 0.5), alongX, boxMin, boxMax, 1.5, 60.0)
                    .has_value());
    CHECK_FALSE(RaycastAabb(Eigen::Vector3d(-1.0, 0.5, 0.5), alongX, boxMin, boxMax, 1.0e-4,
                            0.5)
                    .has_value());

    // An origin exactly on a face is deterministic: entry t=0 is below t_min,
    // so the exit root is returned.
    const auto onFace =
        RaycastAabb(Eigen::Vector3d(0.0, 0.5, 0.5), alongX, boxMin, boxMax, 1.0e-4, 60.0);
    REQUIRE(onFace.has_value());
    CHECK(*onFace == Approx(1.0));
}

TEST_CASE("the procedural street scene is deterministic and hash-stable", "[eval][scene]")
{
    const ProceduralStreetConfig config = TinyConfig();
    const ProceduralScene first = raylidar::eval::GenerateProceduralStreet(config);
    const ProceduralScene second = raylidar::eval::GenerateProceduralStreet(config);
    REQUIRE(first.triangles.size() == second.triangles.size());
    REQUIRE(first.vertices.size() == second.vertices.size());

    TempDir temp;
    const fs::path firstPly = temp.path / "first.ply";
    const fs::path secondPly = temp.path / "second.ply";
    raylidar::eval::WriteProceduralPly(firstPly, first);
    raylidar::eval::WriteProceduralPly(secondPly, second);
    CHECK(raylidar::Sha256File(firstPly) == raylidar::Sha256File(secondPly));

    ProceduralStreetConfig other = config;
    other.seed = config.seed + 1U;
    const ProceduralScene changed = raylidar::eval::GenerateProceduralStreet(other);
    const fs::path changedPly = temp.path / "changed.ply";
    raylidar::eval::WriteProceduralPly(changedPly, changed);
    CHECK(raylidar::Sha256File(changedPly) != raylidar::Sha256File(firstPly));
}

TEST_CASE("the procedural scene contains ground, boxes, and thin structures",
          "[eval][scene]")
{
    const ProceduralScene scene = raylidar::eval::GenerateProceduralStreet(TinyConfig());
    REQUIRE(scene.boxes.size() == 4U);
    REQUIRE(scene.triangles.size() == 2U + 12U * scene.boxes.size());

    std::uint64_t thinCount = 0;
    for (const auto &box : scene.boxes)
    {
        const Eigen::Vector3d size = box.second - box.first;
        CHECK(box.first.z() == Approx(0.0));
        CHECK(box.second.x() <= TinyConfig().ground_length_m);
        CHECK(box.first.y() >= -TinyConfig().ground_width_m / 2.0);
        CHECK(box.second.y() <= TinyConfig().ground_width_m / 2.0);
        const double horizontalMin = std::min(size.x(), size.y());
        if (horizontalMin >= 0.05 && horizontalMin <= 0.15 && size.z() >= 2.5 &&
            size.z() <= 4.0)
        {
            ++thinCount;
        }
    }
    CHECK(thinCount >= 2U);

    CHECK(scene.bbox_min.allFinite());
    CHECK(scene.bbox_max.allFinite());
    CHECK(scene.bbox_min.z() == Approx(0.0));
    CHECK(scene.bbox_max.x() == Approx(TinyConfig().ground_length_m));

    TempDir temp;
    const fs::path plyPath = temp.path / "scene.ply";
    raylidar::eval::WriteProceduralPly(plyPath, scene);
    const raylidar::io::MeshStats stats = raylidar::io::ValidateMeshFile(plyPath);
    CHECK(stats.vertex_count == scene.vertices.size());
    CHECK(stats.triangle_count == scene.indices.size());
    CHECK(stats.float64_positions);
    CHECK(stats.bbox_min.isApprox(scene.bbox_min));
    CHECK(stats.bbox_max.isApprox(scene.bbox_max));
}

TEST_CASE("RaycastScene returns the nearest primitive deterministically", "[eval][scene]")
{
    ProceduralStreetConfig groundOnly;
    groundOnly.seed = 7;
    groundOnly.ground_length_m = 20.0;
    groundOnly.ground_width_m = 10.0;
    groundOnly.box_count = 0;
    groundOnly.thin_box_count = 0;
    const ProceduralScene groundScene =
        raylidar::eval::GenerateProceduralStreet(groundOnly);
    const auto groundHit = RaycastScene(groundScene, Eigen::Vector3d(5.0, 0.0, 2.0),
                                        Eigen::Vector3d(0.0, 0.0, -1.0), 1.0e-4, 60.0);
    REQUIRE(groundHit.has_value());
    CHECK(groundHit->t == Approx(2.0));
    CHECK(groundHit->primitive_id == 0U);
    // Repeated calls are identical.
    const auto again = RaycastScene(groundScene, Eigen::Vector3d(5.0, 0.0, 2.0),
                                    Eigen::Vector3d(0.0, 0.0, -1.0), 1.0e-4, 60.0);
    REQUIRE(again.has_value());
    CHECK(again->t == groundHit->t);
    CHECK(again->primitive_id == groundHit->primitive_id);

    ProceduralStreetConfig twoBoxes;
    twoBoxes.seed = 11;
    twoBoxes.ground_length_m = 20.0;
    twoBoxes.ground_width_m = 10.0;
    twoBoxes.box_count = 2;
    twoBoxes.thin_box_count = 1;
    const ProceduralScene scene = raylidar::eval::GenerateProceduralStreet(twoBoxes);
    REQUIRE(scene.boxes.size() == 2U);
    const Eigen::Vector3d center = 0.5 * (scene.boxes[0].first + scene.boxes[0].second);
    const Eigen::Vector3d origin(center.x(), center.y(), scene.boxes[0].second.z() + 2.0);
    const auto boxHit = RaycastScene(scene, origin, Eigen::Vector3d(0.0, 0.0, -1.0), 1.0e-4,
                                     60.0);
    REQUIRE(boxHit.has_value());
    CHECK(boxHit->t == Approx(2.0));
    CHECK(boxHit->primitive_id == 1U);
}

TEST_CASE("the scene manifest records generator, params, counts, bbox, and hash",
          "[eval][scene]")
{
    const ProceduralScene scene = raylidar::eval::GenerateProceduralStreet(TinyConfig());
    TempDir temp;
    const fs::path plyPath = temp.path / "scene.ply";
    raylidar::eval::WriteProceduralPly(plyPath, scene);
    const nlohmann::json manifest =
        raylidar::eval::BuildSceneManifest(scene, plyPath);

    CHECK(manifest.at("generator").get<std::string>() == "procedural_street_v1");
    CHECK(manifest.at("seed").get<std::uint64_t>() == TinyConfig().seed);
    CHECK(manifest.at("ground_length_m").get<double>() ==
          Approx(TinyConfig().ground_length_m));
    CHECK(manifest.at("ground_width_m").get<double>() ==
          Approx(TinyConfig().ground_width_m));
    CHECK(manifest.at("box_count").get<int>() == TinyConfig().box_count);
    CHECK(manifest.at("thin_box_count").get<int>() == TinyConfig().thin_box_count);
    CHECK(manifest.at("vertex_count").get<std::uint64_t>() == scene.vertices.size());
    CHECK(manifest.at("triangle_count").get<std::uint64_t>() == scene.indices.size());
    CHECK(manifest.at("bbox_min_m").size() == 3U);
    CHECK(manifest.at("bbox_max_m").size() == 3U);
    CHECK(manifest.at("sha256").get<std::string>() == raylidar::Sha256File(plyPath));
}
