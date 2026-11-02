#include <raylidar/cli/commands.h>
#include <raylidar/prereg/prereg.h>

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
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

std::atomic<unsigned long> gCounter{0};

fs::path MakeTempDir(std::string_view name)
{
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const unsigned long counter = gCounter.fetch_add(1U);
    const fs::path dir = fs::temp_directory_path() / ("raylidar_prereg_cli_" + std::string(name) + "_" +
                                                      std::to_string(stamp) + "_" + std::to_string(counter));
    std::error_code error;
    fs::remove_all(dir, error);
    fs::create_directories(dir, error);
    REQUIRE_FALSE(error);
    return dir;
}

void WriteFile(const fs::path &path, std::string_view text)
{
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    REQUIRE(out.good());
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
}

std::string ReadFile(const fs::path &path)
{
    std::ifstream in(path, std::ios::binary);
    REQUIRE(in.good());
    std::ostringstream contents;
    contents << in.rdbuf();
    return contents.str();
}

struct DecisionSet
{
    fs::path dir;
    fs::path decisions;
    fs::path protocol;
    fs::path sidecar;
};

DecisionSet MakeDecisionSet(std::string_view name)
{
    DecisionSet set;
    set.dir = MakeTempDir(name);
    set.decisions = set.dir / "docs" / "decisions.md";
    set.protocol = set.dir / "configs" / "protocol.yaml";
    set.sidecar = set.dir / "artifacts" / "decisions.sha256";
    WriteFile(set.decisions, "# Phase 1 — Pre-registered Evaluation Decisions\n"
                             "\n"
                             "**Status:** PRE-REGISTERED\n"
                             "**Frozen at (UTC):** PENDING\n"
                             "\n"
                             "## Change Policy\n"
                             "\n"
                             "frozen before reconstruction\n");
    WriteFile(set.protocol, "primary_tau_m: 0.20\nmax_range_m: 60.0\n");
    return set;
}

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
        result.code = RunPrereg(views);
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

std::vector<std::string> CommonArgs(const DecisionSet &set, std::string_view subcommand)
{
    return {std::string(subcommand), "--decisions", set.decisions.string(), "--protocol",
            set.protocol.string(),   "--sidecar",   set.sidecar.string()};
}
} // namespace

TEST_CASE("prereg freeze then verify succeeds", "[prereg]")
{
    const DecisionSet set = MakeDecisionSet("ok");

    const Invocation freeze = Invoke(CommonArgs(set, "freeze"));
    CHECK(freeze.code == 0);
    CHECK(freeze.out.find("frozen ") != std::string::npos);

    const Invocation verify = Invoke(CommonArgs(set, "verify"));
    CHECK(verify.code == 0);
    CHECK(verify.out.find("OK ") != std::string::npos);
}

TEST_CASE("a tampered decisions file returns exit 6 with MISMATCH", "[prereg]")
{
    const DecisionSet set = MakeDecisionSet("tamper");
    const Invocation freeze = Invoke(CommonArgs(set, "freeze"));
    REQUIRE(freeze.code == 0);

    std::string text = ReadFile(set.decisions);
    const std::size_t position = text.find("frozen");
    REQUIRE(position != std::string::npos);
    text[position] = 'F';
    WriteFile(set.decisions, text);

    const Invocation verify = Invoke(CommonArgs(set, "verify"));
    CHECK(verify.code == 6);
    CHECK(verify.err.find("MISMATCH") != std::string::npos);
}

TEST_CASE("a missing sidecar returns exit 4", "[prereg]")
{
    const DecisionSet set = MakeDecisionSet("missing");
    const Invocation verify = Invoke(CommonArgs(set, "verify"));
    CHECK(verify.code == 4);
}

TEST_CASE("no subcommand returns exit 2", "[prereg]")
{
    const Invocation invocation = Invoke({});
    CHECK(invocation.code == 2);
}
