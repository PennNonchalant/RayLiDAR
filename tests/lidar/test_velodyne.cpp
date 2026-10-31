#include "fixture.h"

#include <raylidar/common/io.h>
#include <raylidar/lidar/velodyne.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::IoError;
using raylidar::lidar::LoadVelodyneScan;
using raylidar::lidar::LoadVelodyneTimestamps;
using raylidar::lidar::ParseUtcTimestamp;

int ExitCodeOf(const std::string &text)
{
    try
    {
        (void)ParseUtcTimestamp(text);
    }
    catch (const IoError &error)
    {
        return error.exitCode();
    }
    return 0;
}
} // namespace

TEST_CASE("ParseUtcTimestamp returns nanoseconds since the Unix epoch", "[lidar]")
{
    CHECK(ParseUtcTimestamp("2013-05-28 00:00:00.000000000") == 1369699200000000000LL);
    CHECK(ParseUtcTimestamp("2013-05-28 00:00:01.500000000") == 1369699201500000000LL);
    CHECK(ParseUtcTimestamp("2016-02-29 12:34:56.123456789") == 1456749296123456789LL);
}

TEST_CASE("ParseUtcTimestamp rejects malformed timestamps with exit 4", "[lidar]")
{
    const std::vector<std::string> malformed = {
        "",
        "2013-05-28 00:00:00",
        "2013-05-28T00:00:00.000000000",
        "2013-05-28 00:00:00.00000000",
        "2013-05-28 00:00:00.0000000000",
        "2013-05-28 00:00:00.000000000 ",
        "2013-05-28 00:00:00.00000000x",
        "2013-05-28 0a:00:00.000000000",
        "2013-13-01 00:00:00.000000000",
        "2013-02-30 00:00:00.000000000",
        "2013-00-10 00:00:00.000000000",
        "2013-05-28 24:00:00.000000000",
        "2014-02-29 00:00:00.000000000",
        "2013-05-28 00:00:00.000000000 trailing",
        "0000-00-00 00:00:00.000000000",
    };

    for (const std::string &text : malformed)
    {
        INFO("timestamp: " << text);
        CHECK(ExitCodeOf(text) == 4);
    }
}

TEST_CASE("LoadVelodyneScan preserves point order and intensity", "[lidar]")
{
    const fs::path root = raylidar::test::MakeLidarTempRoot("load_scan");
    const fs::path bin = raylidar::test::WriteVelodyneBin(
        root / "0000000001.bin",
        {{1.0F, 2.0F, 3.0F, 0.25F}, {-4.0F, 5.0F, 6.0F, 0.75F}});

    const std::vector<raylidar::lidar::PointXYZI> points = LoadVelodyneScan(bin);
    REQUIRE(points.size() == 2U);
    CHECK(points[0].x == 1.0F);
    CHECK(points[0].y == 2.0F);
    CHECK(points[0].z == 3.0F);
    CHECK(points[0].intensity == 0.25F);
    CHECK(points[1].x == -4.0F);
    CHECK(points[1].y == 5.0F);
    CHECK(points[1].z == 6.0F);
    CHECK(points[1].intensity == 0.75F);
}

TEST_CASE("LoadVelodyneScan rejects truncated, empty, and oversized scans", "[lidar]")
{
    const fs::path root = raylidar::test::MakeLidarTempRoot("load_scan_bad");

    const fs::path truncated = root / "truncated.bin";
    raylidar::test::WriteRawBytes(truncated, std::vector<std::uint8_t>(20U, 0U));
    CHECK_THROWS_AS(LoadVelodyneScan(truncated), IoError);

    const fs::path empty = root / "empty.bin";
    raylidar::test::WriteRawBytes(empty, {});
    CHECK_THROWS_AS(LoadVelodyneScan(empty), IoError);

    const fs::path oversized = raylidar::test::WriteVelodyneBin(
        root / "oversized.bin",
        {{0.0F, 0.0F, 0.0F, 0.0F}, {1.0F, 1.0F, 1.0F, 1.0F}});
    CHECK_THROWS_AS(LoadVelodyneScan(oversized, 1U), IoError);

    CHECK_THROWS_AS(LoadVelodyneScan(root / "missing.bin"), IoError);
}

TEST_CASE("LoadVelodyneTimestamps parses every non-empty line", "[lidar]")
{
    const fs::path root = raylidar::test::MakeLidarTempRoot("timestamps");
    const fs::path timestamps = root / "timestamps.txt";
    raylidar::test::WriteTextFile(timestamps,
                                  "2013-05-28 00:00:00.000000000\n"
                                  "\n"
                                  "2013-05-28 00:00:01.000000000\n"
                                  "2013-05-28 00:00:02.000000000\n");

    const std::vector<std::int64_t> values = LoadVelodyneTimestamps(timestamps);
    REQUIRE(values.size() == 3U);
    CHECK(values[0] == 1369699200000000000LL);
    CHECK(values[1] - values[0] == 1000000000LL);
    CHECK(values[2] - values[1] == 1000000000LL);
}

TEST_CASE("LoadVelodyneTimestamps fails loudly on malformed lines", "[lidar]")
{
    const fs::path root = raylidar::test::MakeLidarTempRoot("timestamps_bad");

    const fs::path malformed = root / "malformed.txt";
    raylidar::test::WriteTextFile(malformed,
                                  "2013-05-28 00:00:00.000000000\n"
                                  "not a timestamp\n");
    CHECK_THROWS_AS(LoadVelodyneTimestamps(malformed), IoError);

    const fs::path empty = root / "empty.txt";
    raylidar::test::WriteTextFile(empty, "\n\n");
    CHECK_THROWS_AS(LoadVelodyneTimestamps(empty), IoError);

    CHECK_THROWS_AS(LoadVelodyneTimestamps(root / "missing.txt"), IoError);
}
