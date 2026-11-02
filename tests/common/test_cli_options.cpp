#include <raylidar/common/cli_options.h>

#include <catch2/catch_test_macros.hpp>

#include <set>
#include <string>
#include <string_view>
#include <vector>

using raylidar::ParseArgs;
using raylidar::UsageError;

namespace
{
std::vector<std::string_view> Views(std::initializer_list<std::string_view> values)
{
    return std::vector<std::string_view>(values);
}

const std::set<std::string> kValueKeys = {"config", "frame"};
const std::set<std::string> kFlagKeys = {"strict"};
} // namespace

TEST_CASE("ParseArgs handles space-separated values", "[common]")
{
    const auto args = Views({"--config", "dataset.yaml"});
    const auto parsed = ParseArgs(args, kValueKeys, kFlagKeys);
    CHECK(parsed.Has("config"));
    CHECK(parsed.Get("config") == "dataset.yaml");
}

TEST_CASE("ParseArgs handles inline values", "[common]")
{
    const auto args = Views({"--config=dataset.yaml"});
    const auto parsed = ParseArgs(args, kValueKeys, kFlagKeys);
    CHECK(parsed.Get("config") == "dataset.yaml");
}

TEST_CASE("ParseArgs collects repeatable keys", "[common]")
{
    const auto args = Views({"--frame", "1", "--frame=2", "--frame", "3"});
    const auto parsed = ParseArgs(args, kValueKeys, kFlagKeys);
    const auto frames = parsed.GetAll("frame");
    REQUIRE(frames.size() == 3U);
    CHECK(frames[0] == "1");
    CHECK(frames[1] == "2");
    CHECK(frames[2] == "3");
    CHECK(parsed.Get("frame") == "3");
}

TEST_CASE("ParseArgs records flags and built-in help", "[common]")
{
    const auto args = Views({"--strict", "--help"});
    const auto parsed = ParseArgs(args, kValueKeys, kFlagKeys);
    CHECK(parsed.Flag("strict"));
    CHECK(parsed.Flag("help"));
    CHECK(parsed.Has("strict"));
}

TEST_CASE("ParseArgs returns the fallback for absent keys", "[common]")
{
    const auto args = Views({});
    const auto parsed = ParseArgs(args, kValueKeys, kFlagKeys);
    CHECK_FALSE(parsed.Has("config"));
    CHECK(parsed.Get("config", "default.yaml") == "default.yaml");
    CHECK(parsed.GetAll("frame").empty());
}

TEST_CASE("ParseArgs rejects unknown keys", "[common]")
{
    const auto args = Views({"--nope", "1"});
    try
    {
        (void)ParseArgs(args, kValueKeys, kFlagKeys);
        FAIL("unknown key did not throw");
    }
    catch (const UsageError &error)
    {
        CHECK(error.exitCode() == 2);
    }
}

TEST_CASE("ParseArgs rejects a missing value", "[common]")
{
    const auto args = Views({"--config"});
    try
    {
        (void)ParseArgs(args, kValueKeys, kFlagKeys);
        FAIL("missing value did not throw");
    }
    catch (const UsageError &error)
    {
        CHECK(error.exitCode() == 2);
    }
}
