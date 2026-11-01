#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <raylidar/common/io.h>
#include <raylidar/eval/output.h>
#include <raylidar/rays/rays.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using Catch::Approx;
using raylidar::eval::ProtocolConfig;

fs::path SourceDir()
{
    return fs::path(RAYLIDAR_SOURCE_DIR);
}

std::string ReadText(const fs::path &path)
{
    std::ifstream input(path, std::ios::binary);
    REQUIRE(input.good());
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

struct TempConfig
{
    explicit TempConfig(const std::string &body)
    {
        static std::uint64_t counter = 0;
        static std::random_device device;
        path = fs::temp_directory_path() /
               ("raylidar-eval-protocol-" + std::to_string(device()) + "-" +
                std::to_string(counter++));
        fs::create_directories(path);
        file = path / "phase3.yaml";
        std::ofstream output(file, std::ios::trunc);
        REQUIRE(output.good());
        output << body;
    }
    ~TempConfig()
    {
        std::error_code error;
        fs::remove_all(path, error);
    }
    fs::path path;
    fs::path file;
};

std::string MakePhase3Yaml(const std::string &t_min, const std::string &r_min,
                           const std::string &t_max, const std::string &taus,
                           const std::string &primary, const std::string &buckets)
{
    std::ostringstream out;
    out << "schema_version: 1\n"
        << "phase: 3\n"
        << "artifacts:\n"
        << "  phase_dir: /tmp/phase3-artifacts\n"
        << "protocol:\n"
        << "  protocol_version: 1\n"
        << "  t_min_m: " << t_min << "\n"
        << "  r_min_m: " << r_min << "\n"
        << "  t_max_m: " << t_max << "\n"
        << "  taus_m: " << taus << "\n"
        << "  primary_tau_m: " << primary << "\n"
        << "  primary_buckets: " << buckets << "\n"
        << "  secondary_bucket: [50.0, 60.0]\n";
    return out.str();
}

std::string FrozenYaml()
{
    return MakePhase3Yaml("0.0001", "1.5", "60.0", "[0.05, 0.20, 0.50]", "0.20",
                          "[[0.0, 20.0], [20.0, 40.0], [40.0, 50.0]]");
}

void RequireProtocolDrift(const std::string &yaml)
{
    TempConfig config(yaml);
    bool rejected = false;
    try
    {
        static_cast<void>(raylidar::eval::LoadProtocolConfig(config.file));
    }
    catch (const raylidar::IoError &error)
    {
        rejected = true;
        CHECK(error.exitCode() == 6);
    }
    CHECK(rejected);
}
} // namespace

TEST_CASE("configs/phase3.yaml matches the frozen Phase-2 constants", "[eval][protocol_doc]")
{
    const ProtocolConfig protocol =
        raylidar::eval::LoadProtocolConfig(SourceDir() / "configs" / "phase3.yaml");
    CHECK(protocol.schema_version == 1);
    CHECK(protocol.t_min_m == Approx(raylidar::rays::kRayTMinM));
    CHECK(protocol.r_min_m == Approx(raylidar::rays::kRayMinRangeM));
    CHECK(protocol.t_max_m == Approx(raylidar::rays::kRayMaxRangeM));
    REQUIRE(protocol.taus_m.size() == 3U);
    CHECK(protocol.taus_m[0] == Approx(0.05));
    CHECK(protocol.taus_m[1] == Approx(0.20));
    CHECK(protocol.taus_m[2] == Approx(0.50));
    CHECK(protocol.primary_tau_m == Approx(0.20));
    CHECK(protocol.primary_buckets[0].first == Approx(0.0));
    CHECK(protocol.primary_buckets[0].second == Approx(20.0));
    CHECK(protocol.primary_buckets[1].first == Approx(20.0));
    CHECK(protocol.primary_buckets[1].second == Approx(40.0));
    CHECK(protocol.primary_buckets[2].first == Approx(40.0));
    CHECK(protocol.primary_buckets[2].second == Approx(50.0));
    CHECK(protocol.secondary_bucket.first == Approx(50.0));
    CHECK(protocol.secondary_bucket.second == Approx(60.0));
}

TEST_CASE("LoadProtocolConfig rejects any drift from the frozen protocol",
          "[eval][protocol_doc]")
{
    SECTION("correct copy loads")
    {
        TempConfig config(FrozenYaml());
        REQUIRE_NOTHROW(raylidar::eval::LoadProtocolConfig(config.file));
    }
    SECTION("changed t_min")
    {
        RequireProtocolDrift(MakePhase3Yaml("0.10", "1.5", "60.0", "[0.05, 0.20, 0.50]",
                                           "0.20", "[[0.0, 20.0], [20.0, 40.0], [40.0, 50.0]]"));
    }
    SECTION("changed r_min")
    {
        RequireProtocolDrift(MakePhase3Yaml("0.0001", "2.0", "60.0", "[0.05, 0.20, 0.50]",
                                           "0.20", "[[0.0, 20.0], [20.0, 40.0], [40.0, 50.0]]"));
    }
    SECTION("changed t_max")
    {
        RequireProtocolDrift(MakePhase3Yaml("0.0001", "1.5", "50.0", "[0.05, 0.20, 0.50]",
                                           "0.20", "[[0.0, 20.0], [20.0, 40.0], [40.0, 50.0]]"));
    }
    SECTION("changed tau set")
    {
        RequireProtocolDrift(MakePhase3Yaml("0.0001", "1.5", "60.0", "[0.05, 0.25, 0.50]",
                                           "0.20", "[[0.0, 20.0], [20.0, 40.0], [40.0, 50.0]]"));
    }
    SECTION("changed primary tau")
    {
        RequireProtocolDrift(MakePhase3Yaml("0.0001", "1.5", "60.0", "[0.05, 0.20, 0.50]",
                                           "0.25", "[[0.0, 20.0], [20.0, 40.0], [40.0, 50.0]]"));
    }
    SECTION("changed bucket edge")
    {
        RequireProtocolDrift(MakePhase3Yaml("0.0001", "1.5", "60.0", "[0.05, 0.20, 0.50]",
                                           "0.20", "[[0.0, 20.0], [20.0, 40.0], [40.0, 55.0]]"));
    }
}

TEST_CASE("docs/protocol.md freezes the required formulas, units, and caveats",
          "[eval][protocol_doc]")
{
    const std::string document = ReadText(SourceDir() / "docs" / "protocol.md");
    const std::vector<std::string> tokens = {
        "MAE",
        "RMSE",
        "MedAE",
        "CD-L1",
        "CD-L2",
        "CD-L1 =",
        "tau",
        "0.05",
        "0.20",
        "0.50",
        "[0,20)",
        "(50,60]",
        "t_min",
        "r_min",
        "1.5",
        "60",
        "schema_version",
        "cd_f_non_independent_under_gt_rays",
        "published_f_at_5cm_uses_squared_distance"};
    for (const auto &token : tokens)
    {
        INFO("missing token: " << token);
        CHECK(document.find(token) != std::string::npos);
    }
    CHECK(document.find("1e-4") != std::string::npos);
}

TEST_CASE("docs/protocol.sha256 verifies docs/protocol.md", "[eval][protocol_doc]")
{
    const std::string sidecar = ReadText(SourceDir() / "docs" / "protocol.sha256");
    REQUIRE(sidecar.size() >= 64U);
    const std::string expected = sidecar.substr(0U, 64U);
    const std::string actual = raylidar::Sha256File(SourceDir() / "docs" / "protocol.md");
    CHECK(expected == actual);
    CHECK(sidecar.find("docs/protocol.md") != std::string::npos);
}

TEST_CASE("docs/sim-io.sha256 still verifies docs/sim-io.md", "[eval][protocol_doc]")
{
    const std::string sidecar = ReadText(SourceDir() / "docs" / "sim-io.sha256");
    REQUIRE(sidecar.size() >= 64U);
    const std::string expected = sidecar.substr(0U, 64U);
    const std::string actual = raylidar::Sha256File(SourceDir() / "docs" / "sim-io.md");
    CHECK(expected == actual);
}
