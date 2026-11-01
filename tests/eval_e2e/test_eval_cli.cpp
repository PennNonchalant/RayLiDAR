#include "fixture.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <raylidar/cli/commands_phase3.h>
#include <raylidar/common/io.h>
#include <raylidar/eval/aggregate.h>
#include <raylidar/eval/output.h>
#include <raylidar/io/sim_io.h>
#include <raylidar/rays/rays.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using Catch::Approx;
using raylidar::eval::AggregateCell;
using raylidar::eval::CellMetrics;
using raylidar::eval::RaySample;
using raylidar::rays::Bucket;
using raylidar::test::Phase3Fixture;

int Run(const std::initializer_list<std::string_view> &args)
{
    return RunEval(std::span<const std::string_view>(args.begin(), args.size()));
}

std::vector<RaySample> LoadSamples(const fs::path &outDir)
{
    const auto rays = raylidar::io::ReadGtRays(outDir / "rays_placeholder.bin");
    const auto hits = raylidar::io::ReadHits(outDir / "hits_placeholder.bin");
    REQUIRE(rays.size() == hits.size());
    std::vector<RaySample> samples;
    samples.reserve(rays.size());
    for (std::size_t index = 0; index < rays.size(); ++index)
    {
        RaySample sample;
        sample.ray_id = rays[index].ray_id;
        sample.frame = rays[index].frame;
        sample.camera = rays[index].camera;
        sample.bucket = raylidar::rays::BucketOf(rays[index].t_gt);
        sample.t_gt = rays[index].t_gt;
        sample.t_min = rays[index].t_min;
        sample.t_max = rays[index].t_max;
        sample.origin_W = rays[index].origin_W;
        sample.dir_W = rays[index].dir_W;
        sample.t_pred = hits[index].hit ? hits[index].t_pred
                                        : std::numeric_limits<double>::infinity();
        samples.push_back(sample);
    }
    return samples;
}

std::vector<std::string> ReadLines(const fs::path &path)
{
    std::ifstream input(path);
    REQUIRE(input.good());
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line))
    {
        lines.push_back(line);
    }
    return lines;
}

void WriteLines(const fs::path &path, const std::vector<std::string> &lines)
{
    std::ofstream output(path, std::ios::trunc);
    REQUIRE(output.good());
    for (const auto &line : lines)
    {
        output << line << '\n';
    }
}

std::vector<std::string> SplitCsv(const std::string &line)
{
    std::vector<std::string> fields;
    std::string field;
    for (const char character : line)
    {
        if (character == ',')
        {
            fields.push_back(field);
            field.clear();
        }
        else
        {
            field.push_back(character);
        }
    }
    fields.push_back(field);
    return fields;
}

std::string JoinCsv(const std::vector<std::string> &fields)
{
    std::string line;
    for (std::size_t index = 0; index < fields.size(); ++index)
    {
        if (index != 0U)
        {
            line.push_back(',');
        }
        line += fields[index];
    }
    return line;
}

void RequirePlaceholderOk(const Phase3Fixture &fixture, const fs::path &outDir)
{
    const int result =
        Run({"placeholder", "--config", fixture.configPath.string(), "--out-dir",
             outDir.string()});
    REQUIRE(result == 0);
}
} // namespace

TEST_CASE("eval placeholder writes every Phase-3 artifact and exits 0", "[eval][cli]")
{
    Phase3Fixture fixture;
    RequirePlaceholderOk(fixture, fixture.outDir);

    for (const char *name :
         {"rays_placeholder.bin", "hits_placeholder.bin", "scene.ply",
          "scene_manifest.json", "sim_io_manifest.json", "metrics.json",
          "metrics.csv"})
    {
        INFO(name);
        CHECK(fs::is_regular_file(fixture.outDir / name));
    }
}

TEST_CASE("the emitted rays and hits satisfy the frozen sim-io contract",
          "[eval][cli]")
{
    Phase3Fixture fixture;
    RequirePlaceholderOk(fixture, fixture.outDir);

    raylidar::io::GtRaysInfo rayInfo;
    raylidar::io::HitsInfo hitInfo;
    const auto rays =
        raylidar::io::ReadGtRays(fixture.outDir / "rays_placeholder.bin", &rayInfo);
    const auto hits = raylidar::io::ReadHits(fixture.outDir / "hits_placeholder.bin",
                                             &hitInfo);
    CHECK(rayInfo.count == rays.size());
    CHECK(hitInfo.count == hits.size());
    REQUIRE(hits.size() == rays.size());
    REQUIRE_NOTHROW(raylidar::io::ValidateHitsAgainstRays(rays, hits));

    for (std::size_t index = 0; index < rays.size(); ++index)
    {
        if (hits[index].hit)
        {
            CHECK(hits[index].t_pred >= rays[index].t_min);
            CHECK(hits[index].t_pred <= rays[index].t_max);
        }
        else
        {
            CHECK(std::isinf(hits[index].t_pred));
            CHECK(hits[index].prim_id == raylidar::io::kPrimIdUnavailable);
        }
    }
}

TEST_CASE("metrics.json hashes, row grid, and micro aggregate are correct",
          "[eval][cli]")
{
    Phase3Fixture fixture;
    RequirePlaceholderOk(fixture, fixture.outDir);

    const fs::path metricsPath = fixture.outDir / "metrics.json";
    const nlohmann::json document = raylidar::ReadJson(metricsPath);
    const nlohmann::json &inputs = document.at("inputs");
    CHECK(inputs.at("config_sha256").get<std::string>() ==
          raylidar::Sha256File(fixture.configPath));
    CHECK(inputs.at("protocol_sha256").get<std::string>() ==
          raylidar::Sha256File(fixture.protocolPath));
    CHECK(inputs.at("ray_set_sha256").get<std::string>() ==
          raylidar::Sha256File(fixture.outDir / "rays_placeholder.bin"));
    CHECK(inputs.at("mesh_sha256").get<std::string>() ==
          raylidar::Sha256File(fixture.outDir / "scene.ply"));
    CHECK(inputs.at("hit_set_sha256").get<std::string>() ==
          raylidar::Sha256File(fixture.outDir / "hits_placeholder.bin"));
    CHECK(inputs.at("scene_generator").get<std::string>() == "procedural_street_v1");
    CHECK(inputs.at("predictor").get<std::string>() == "linear_range_bias_scale_v1");

    const nlohmann::json &rows = document.at("rows");
    REQUIRE(rows.size() == 2U * 2U * 4U);
    std::set<std::tuple<std::uint32_t, std::uint16_t, std::string>> seen;
    for (const auto &row : rows)
    {
        CHECK(row.contains("coverage"));
        const auto key = std::make_tuple(row.at("frame").get<std::uint32_t>(),
                                         row.at("camera").get<std::uint16_t>(),
                                         row.at("bucket").get<std::string>());
        CHECK(seen.insert(key).second);
    }
    for (std::uint32_t frame = 0; frame < 2U; ++frame)
    {
        for (std::uint16_t camera = 0; camera < 2U; ++camera)
        {
            for (int bucket = 0; bucket < 4; ++bucket)
            {
                const auto key = std::make_tuple(
                    frame, camera,
                    std::string(raylidar::rays::BucketName(static_cast<Bucket>(bucket))));
                INFO("frame " << frame << " camera " << camera << " bucket " << bucket);
                CHECK(seen.find(key) != seen.end());
            }
        }
    }

    std::vector<RaySample> primary;
    for (const auto &sample : LoadSamples(fixture.outDir))
    {
        if (sample.bucket != Bucket::B50_60)
        {
            primary.push_back(sample);
        }
    }
    REQUIRE(!primary.empty());
    raylidar::eval::RawCounts raw;
    raw.raw_total = document.at("aggregates")
                        .at("micro_primary")
                        .at("cell")
                        .at("n_raw")
                        .get<std::uint64_t>();
    const CellMetrics expected = AggregateCell(primary, raw);
    const nlohmann::json &cell =
        document.at("aggregates").at("micro_primary").at("cell");
    CHECK(cell.at("n_valid").get<std::uint64_t>() == expected.n_valid);
    CHECK(cell.at("n_hits").get<std::uint64_t>() == expected.n_hits);
    CHECK(cell.at("n_misses").get<std::uint64_t>() == expected.n_misses);
    REQUIRE(expected.range.mae.has_value());
    CHECK(cell.at("range").at("mae").get<double>() == Approx(*expected.range.mae));
    REQUIRE(expected.coverage.has_value());
    CHECK(cell.at("coverage").get<double>() == Approx(*expected.coverage));
    CHECK(document.at("aggregates").at("micro_primary").at("prf").is_object());
}

TEST_CASE("metrics.csv has the frozen header, a complete grid, and no nan/inf",
          "[eval][cli]")
{
    Phase3Fixture fixture;
    RequirePlaceholderOk(fixture, fixture.outDir);

    const std::vector<std::string> lines =
        ReadLines(fixture.outDir / "metrics.csv");
    REQUIRE(lines.size() == 1U + 2U * 2U * 4U);
    CHECK(lines[0] == std::string(raylidar::eval::MetricsCsvHeader()));

    bool sawEmptyRow = false;
    for (std::size_t index = 1; index < lines.size(); ++index)
    {
        INFO("line " << index);
        CHECK(lines[index].find("nan") == std::string::npos);
        CHECK(lines[index].find("inf") == std::string::npos);
        const std::vector<std::string> fields = SplitCsv(lines[index]);
        REQUIRE(fields.size() == 41U);
        CHECK(fields[0] == "1");
        if (fields[5] == "0" && fields[6] == "0")
        {
            sawEmptyRow = true;
            CHECK(fields[4] == "0");
            CHECK(fields[7] == "0");
        }
    }
    CHECK(sawEmptyRow);
}

TEST_CASE("a second run is deterministic for rays, hits, mesh, CSV, and metrics",
          "[eval][cli]")
{
    Phase3Fixture fixture;
    const fs::path second = fixture.root / "out2";
    RequirePlaceholderOk(fixture, fixture.outDir);
    RequirePlaceholderOk(fixture, second);

    for (const char *name : {"rays_placeholder.bin", "hits_placeholder.bin", "scene.ply"})
    {
        INFO(name);
        CHECK(raylidar::Sha256File(fixture.outDir / name) ==
              raylidar::Sha256File(second / name));
    }
    CHECK(raylidar::test::ReadFileText(fixture.outDir / "metrics.csv") ==
          raylidar::test::ReadFileText(second / "metrics.csv"));

    nlohmann::json first = raylidar::ReadJson(fixture.outDir / "metrics.json");
    nlohmann::json other = raylidar::ReadJson(second / "metrics.json");
    first.erase("generated_at_utc");
    other.erase("generated_at_utc");
    CHECK(first == other);
}

TEST_CASE("eval validate accepts clean artifacts and rejects corrupted ones",
          "[eval][cli]")
{
    Phase3Fixture fixture;
    RequirePlaceholderOk(fixture, fixture.outDir);
    const fs::path metricsPath = fixture.outDir / "metrics.json";
    const fs::path csvPath = fixture.outDir / "metrics.csv";

    CHECK(Run({"validate", "--metrics", metricsPath.string(), "--csv",
               csvPath.string()}) == 0);

    SECTION("corrupted recorded hash")
    {
        nlohmann::json document = raylidar::ReadJson(metricsPath);
        document["inputs"]["config_sha256"] = std::string(63U, 'a') + "z";
        const fs::path corrupt = fixture.outDir / "corrupt_metrics.json";
        raylidar::WriteJson(corrupt, document);
        CHECK(Run({"validate", "--metrics", corrupt.string(), "--csv",
                   csvPath.string()}) == 6);
    }
    SECTION("removed CSV row")
    {
        std::vector<std::string> lines = ReadLines(csvPath);
        lines.erase(lines.begin() + 3);
        const fs::path badCsv = fixture.outDir / "missing_row.csv";
        WriteLines(badCsv, lines);
        CHECK(Run({"validate", "--metrics", metricsPath.string(), "--csv",
                   badCsv.string()}) == 4);
    }
    SECTION("nan token in a metric cell")
    {
        std::vector<std::string> lines = ReadLines(csvPath);
        std::vector<std::string> fields = SplitCsv(lines[1]);
        fields[9] = "nan";
        lines[1] = JoinCsv(fields);
        const fs::path badCsv = fixture.outDir / "nan.csv";
        WriteLines(badCsv, lines);
        CHECK(Run({"validate", "--metrics", metricsPath.string(), "--csv",
                   badCsv.string()}) == 4);
    }
}

TEST_CASE("eval rejects unknown modes and bad paths, and writes only in --out-dir",
          "[eval][cli]")
{
    Phase3Fixture fixture;
    CHECK(Run({"explode"}) == 2);
    CHECK(Run({"placeholder", "--config", (fixture.root / "missing.yaml").string(),
               "--out-dir", (fixture.root / "nope").string()}) == 4);
    CHECK_FALSE(fs::exists(fixture.root / "nope"));

    raylidar::test::WriteFileText(fixture.root / "blocker", "not a directory");
    CHECK(Run({"placeholder", "--config", fixture.configPath.string(), "--out-dir",
               (fixture.root / "blocker" / "child").string()}) == 4);

    RequirePlaceholderOk(fixture, fixture.outDir);
    std::size_t configEntries = 0;
    for (const auto &entry : fs::directory_iterator(fixture.configPath.parent_path()))
    {
        static_cast<void>(entry);
        ++configEntries;
    }
    CHECK(configEntries == 2U);
}
