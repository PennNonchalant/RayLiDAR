#include "fixture.h"

#include <raylidar/cli/commands_phase2.h>
#include <raylidar/common/io.h>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::ReadJson;
using raylidar::test::MakeSyntheticG2Tree;
using raylidar::test::SyntheticG2Tree;

int RunG2(const std::vector<std::string> &arguments)
{
    std::vector<std::string_view> views;
    views.reserve(arguments.size());
    for (const std::string &argument : arguments)
    {
        views.emplace_back(argument);
    }
    return ::RunG2Check(std::span<const std::string_view>(views.data(), views.size()));
}

std::string ReadText(const fs::path &path)
{
    std::ifstream input(path, std::ios::binary);
    REQUIRE(input.good());
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}
} // namespace

TEST_CASE("g2-check writes JSON and Markdown and exits 0 on a passing tree", "[g2]")
{
    const SyntheticG2Tree tree = MakeSyntheticG2Tree("cli_pass");
    const fs::path out = tree.root / "g2" / "G2_REPORT";
    const int code = RunG2({"--config", tree.configPath.string(), "--artifacts-dir",
                            tree.phaseDir.string(), "--out", out.string()});
    REQUIRE(code == 0);
    const nlohmann::json report = ReadJson(out.string() + ".json");
    REQUIRE(report.at("pass").get<bool>());
    const std::string markdown = ReadText(out.string() + ".md");
    REQUIRE(markdown.find("G2 PASSED") != std::string::npos);
}

TEST_CASE("g2-check exits non-zero and reports NOT PASSED on missing evidence", "[g2]")
{
    const SyntheticG2Tree tree = MakeSyntheticG2Tree("cli_fail");
    std::error_code error;
    fs::remove(tree.phaseDir / "uncurl" / "uncurl_report.json", error);
    REQUIRE_FALSE(error);
    const fs::path out = tree.root / "g2" / "G2_REPORT";
    const int code = RunG2({"--config", tree.configPath.string(), "--artifacts-dir",
                            tree.phaseDir.string(), "--out", out.string()});
    REQUIRE(code != 0);
    const nlohmann::json report = ReadJson(out.string() + ".json");
    REQUIRE_FALSE(report.at("pass").get<bool>());
    REQUIRE(ReadText(out.string() + ".md").find("G2 NOT PASSED") != std::string::npos);
}

TEST_CASE("g2-check --help exits 0", "[g2]")
{
    REQUIRE(RunG2({"--help"}) == 0);
}
