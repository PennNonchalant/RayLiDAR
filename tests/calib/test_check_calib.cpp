#include "fixture.h"

#include <raylidar/cli/commands.h>
#include <raylidar/common/io.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::ReadJson;
using raylidar::test::CalibTreeOptions;
using raylidar::test::MakeCalibTree;
using raylidar::test::MakeTempRoot;
using raylidar::test::WriteFramesConfig;

constexpr const char *kSequence = "2013_05_28_drive_0000_sync";

struct Invocation
{
    int code = 0;
    std::string out;
    std::string err;
};

Invocation Invoke(const std::vector<std::string> &args)
{
    std::vector<std::string_view> views;
    views.reserve(args.size());
    for (const std::string &arg : args)
    {
        views.emplace_back(arg);
    }
    Invocation result;
    std::ostringstream out;
    std::ostringstream err;
    std::streambuf *oldOut = std::cout.rdbuf(out.rdbuf());
    std::streambuf *oldErr = std::cerr.rdbuf(err.rdbuf());
    try
    {
        result.code = RunCheckCalib(views);
    }
    catch (...)
    {
        std::cout.rdbuf(oldOut);
        std::cerr.rdbuf(oldErr);
        throw;
    }
    std::cout.rdbuf(oldOut);
    std::cerr.rdbuf(oldErr);
    result.out = out.str();
    result.err = err.str();
    return result;
}

std::vector<std::string> Split(const std::string &line, char delimiter)
{
    std::vector<std::string> cells;
    std::istringstream stream(line);
    std::string cell;
    while (std::getline(stream, cell, delimiter))
    {
        cells.push_back(cell);
    }
    return cells;
}

std::vector<std::string> ReadLines(const fs::path &path)
{
    std::ifstream input(path);
    REQUIRE(input.good());
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line))
    {
        if (!line.empty())
        {
            lines.push_back(line);
        }
    }
    return lines;
}

bool IsNumber(const std::string &cell)
{
    if (cell.empty())
    {
        return false;
    }
    char *end = nullptr;
    (void)std::strtod(cell.c_str(), &end);
    return end != cell.c_str() && *end == '\0';
}

int IndexOf(const std::vector<std::string> &cells, const std::string &name)
{
    const auto found = std::find(cells.begin(), cells.end(), name);
    return found == cells.end() ? -1 : static_cast<int>(found - cells.begin());
}

void RequireMatrixGroup(const std::vector<std::string> &header,
                        const std::vector<std::string> &row, const std::string &prefix)
{
    const int start = IndexOf(header, prefix + "_r11");
    REQUIRE(start >= 0);
    const char *suffixes[] = {"r11", "r12", "r13", "tx", "r21", "r22",
                              "r23", "ty", "r31", "r32", "r33", "tz"};
    REQUIRE(static_cast<std::size_t>(start) + 12U <= row.size());
    for (int offset = 0; offset < 12; ++offset)
    {
        CHECK(header[static_cast<std::size_t>(start + offset)] ==
              prefix + "_" + suffixes[offset]);
        CHECK(IsNumber(row[static_cast<std::size_t>(start + offset)]));
    }
}

struct FixtureRun
{
    fs::path root;
    fs::path outDir;
    Invocation invocation;
};

FixtureRun RunFixture(const CalibTreeOptions &options, const std::vector<std::string> &args,
                      std::string_view name)
{
    FixtureRun run;
    run.root = MakeCalibTree(MakeTempRoot(name), options);
    const fs::path config = WriteFramesConfig(run.root / "frames.yaml", run.root, kSequence);
    run.outDir = run.root / "out";
    std::vector<std::string> full = {"--config", config.string(), "--out-dir", run.outDir.string()};
    full.insert(full.end(), args.begin(), args.end());
    run.invocation = Invoke(full);
    return run;
}
} // namespace

TEST_CASE("check-calib writes the three canonical artifacts", "[calib]")
{
    const FixtureRun run = RunFixture({}, {"--auto-frames", "3"}, "check_ok");
    CHECK(run.invocation.code == 0);
    CHECK(fs::is_regular_file(run.outDir / "calib.json"));
    CHECK(fs::is_regular_file(run.outDir / "frames.csv"));
    CHECK(fs::is_regular_file(run.outDir / "calib_report.json"));

    const auto report = ReadJson(run.outDir / "calib_report.json");
    CHECK(report["schema_version"] == 1);
    CHECK(report["pass"] == true);
    CHECK(report["frames"].size() == 3U);
    CHECK(report.contains("max_cam0_world_delta"));
    CHECK(report.contains("camera_height_m"));
    CHECK(report.contains("camera_height_spread_m"));
    CHECK(report["camera_height_status"] == "ok");
}

TEST_CASE("frames.csv has explicit 12-component matrix groups", "[calib]")
{
    const FixtureRun run = RunFixture({}, {"--auto-frames", "3"}, "check_csv");
    CHECK(run.invocation.code == 0);

    const std::vector<std::string> lines = ReadLines(run.outDir / "frames.csv");
    REQUIRE(lines.size() == 4U); // header + 3 frames
    const std::vector<std::string> header = Split(lines.front(), ',');
    // 3 scalar columns + 12 for T_W_I + 12 per camera (2) + 1 height = 40.
    CHECK(header.size() == 40U);

    CHECK(header[0] == "frame");
    CHECK(header[1] == "pose_frame");
    CHECK(header[2] == "pose_delta");
    CHECK(header.back() == "camera_height_m");

    for (std::size_t index = 1; index < lines.size(); ++index)
    {
        const std::vector<std::string> row = Split(lines[index], ',');
        REQUIRE(row.size() == 40U);
        CHECK(IsNumber(row[0]));
        CHECK(IsNumber(row[3])); // first T_W_I component

        RequireMatrixGroup(header, row, "T_W_I");
        RequireMatrixGroup(header, row, "image_00_rect");
        RequireMatrixGroup(header, row, "image_01_rect");
        CHECK(IsNumber(row.back()));
        // pose_delta = pose_frame - frame.
        CHECK(std::strtod(row[2].c_str(), nullptr) ==
              std::strtod(row[1].c_str(), nullptr) - std::strtod(row[0].c_str(), nullptr));
    }
}

TEST_CASE("calib.json records a 3x3 K_rect equal to P_rect.leftCols(3)", "[calib]")
{
    const FixtureRun run = RunFixture({}, {"--auto-frames", "3"}, "check_intrinsics");
    CHECK(run.invocation.code == 0);

    const auto calib = ReadJson(run.outDir / "calib.json");
    for (const std::string camera : {"image_00", "image_01"})
    {
        const auto &projection = calib["P_rect_" + camera];
        const auto &intrinsic = calib["K_rect_" + camera];
        REQUIRE(projection.size() == 3U);
        REQUIRE(intrinsic.size() == 3U);
        REQUIRE(intrinsic[0].size() == 3U);
        for (std::size_t row = 0; row < 3U; ++row)
        {
            for (std::size_t col = 0; col < 3U; ++col)
            {
                CHECK(intrinsic[row][col].get<double>() ==
                      Catch::Approx(projection[row][col].get<double>()).margin(1e-12));
            }
        }
    }
}

TEST_CASE("auto-selected frames form one contiguous window", "[calib]")
{
    SECTION("first run wins")
    {
        CalibTreeOptions options;
        options.frames = {0, 1, 2, 7};
        const FixtureRun run = RunFixture(options, {"--auto-frames", "3"}, "check_window_a");
        CHECK(run.invocation.code == 0);
        const std::vector<std::string> lines = ReadLines(run.outDir / "frames.csv");
        REQUIRE(lines.size() == 4U);
        CHECK(Split(lines[1], ',')[0] == "0");
        CHECK(Split(lines[2], ',')[0] == "1");
        CHECK(Split(lines[3], ',')[0] == "2");
    }

    SECTION("isolated head frame is skipped")
    {
        CalibTreeOptions options;
        options.frames = {0, 2, 3, 4};
        const FixtureRun run = RunFixture(options, {"--auto-frames", "3"}, "check_window_b");
        CHECK(run.invocation.code == 0);
        const std::vector<std::string> lines = ReadLines(run.outDir / "frames.csv");
        REQUIRE(lines.size() == 4U);
        CHECK(Split(lines[1], ',')[0] == "2");
        CHECK(Split(lines[2], ',')[0] == "3");
        CHECK(Split(lines[3], ',')[0] == "4");
    }
}

TEST_CASE("check-calib fails loudly on invalid input", "[calib]")
{
    SECTION("perturbed rectification exits 5")
    {
        CalibTreeOptions options;
        options.rectPerturbation = 0.01;
        const FixtureRun run = RunFixture(options, {"--auto-frames", "3"}, "check_bad_rect");
        CHECK(run.invocation.code == 5);
        CHECK(run.invocation.err.find("R_rect") != std::string::npos);
    }

    SECTION("missing LiDAR exits 5 with skipped_no_lidar")
    {
        CalibTreeOptions options;
        options.omitLidar = {1};
        const FixtureRun run = RunFixture(options, {"--frames", "0,1,2"}, "check_no_lidar");
        CHECK(run.invocation.code == 5);
        const auto report = ReadJson(run.outDir / "calib_report.json");
        CHECK(report["camera_height_status"] == "skipped_no_lidar");
        CHECK(report["pass"] == false);
    }

    SECTION("frame present in poses but absent from cam0_to_world exits 4")
    {
        CalibTreeOptions options;
        options.omitCam0 = {2};
        const FixtureRun run = RunFixture(options, {"--frames", "2"}, "check_missing_cam0");
        CHECK(run.invocation.code == 4);
        CHECK(run.invocation.err.find("cam0_to_world") != std::string::npos);
        CHECK(run.invocation.err.find("frame 2") != std::string::npos);
    }

    SECTION("frame present in cam0_to_world but absent from poses exits 4")
    {
        CalibTreeOptions options;
        options.omitPoses = {2};
        const FixtureRun run = RunFixture(options, {"--frames", "2"}, "check_missing_poses");
        CHECK(run.invocation.code == 4);
        CHECK(run.invocation.err.find("poses") != std::string::npos);
        CHECK(run.invocation.err.find("frame 2") != std::string::npos);
    }
}

TEST_CASE("check-calib --help lists its options", "[calib]")
{
    const Invocation invocation = Invoke({"--help"});
    CHECK(invocation.code == 0);
    CHECK(invocation.out.find("--config") != std::string::npos);
    CHECK(invocation.out.find("--frames") != std::string::npos);
    CHECK(invocation.out.find("--auto-frames") != std::string::npos);
    CHECK(invocation.out.find("--out-dir") != std::string::npos);
}
