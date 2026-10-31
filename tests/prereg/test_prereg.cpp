#include <raylidar/common/io.h>
#include <raylidar/prereg/prereg.h>

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

namespace
{
namespace fs = std::filesystem;
using raylidar::IoError;
using raylidar::ReadJson;
using raylidar::Sha256File;
using raylidar::prereg::FreezeDecisions;
using raylidar::prereg::ReadFrozenAt;
using raylidar::prereg::Sha256Decisions;
using raylidar::prereg::VerifyDecisions;

std::atomic<unsigned long> gCounter{0};

fs::path MakeTempDir(std::string_view name)
{
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const unsigned long counter = gCounter.fetch_add(1U);
    const fs::path dir = fs::temp_directory_path() /
                         ("raylidar_prereg_" + std::string(name) + "_" + std::to_string(stamp) +
                          "_" + std::to_string(counter));
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
    WriteFile(set.decisions,
              "# Phase 1 — Pre-registered Evaluation Decisions\n"
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

constexpr const char *kUtc = "2026-10-31T12:00:00Z";
} // namespace

TEST_CASE("Sha256Decisions tracks the exact file bytes", "[prereg]")
{
    const DecisionSet set = MakeDecisionSet("sha");
    const std::string first = Sha256Decisions(set.decisions);
    REQUIRE(first.size() == 64U);

    WriteFile(set.decisions, ReadFile(set.decisions) + "x");
    const std::string second = Sha256Decisions(set.decisions);
    CHECK(first != second);
}

TEST_CASE("freeze timestamps, writes the sidecar, and verifies", "[prereg]")
{
    const DecisionSet set = MakeDecisionSet("freeze");
    const nlohmann::json record =
        FreezeDecisions(set.decisions, set.protocol, set.sidecar, kUtc);

    CHECK(ReadFrozenAt(set.decisions) == kUtc);
    CHECK(ReadFile(set.decisions).find("PENDING") == std::string::npos);
    CHECK(record["frozen_at_utc"] == kUtc);

    REQUIRE(fs::is_regular_file(set.sidecar));
    const std::string sidecar = ReadFile(set.sidecar);
    REQUIRE(sidecar.size() > 66U);
    CHECK(sidecar.substr(0U, 64U) == record["decisions_sha256"].get<std::string>());
    CHECK(sidecar[64] == ' ');
    CHECK(sidecar[65] == ' ');
    CHECK(sidecar.find(set.decisions.generic_string()) != std::string::npos);

    const nlohmann::json verify = VerifyDecisions(set.decisions, set.sidecar);
    CHECK(verify["ok"] == true);
    CHECK(verify["frozen_at_utc"] == kUtc);
    CHECK(verify["expected"] == verify["actual"]);
}

TEST_CASE("tampering after freeze is detected", "[prereg]")
{
    const DecisionSet set = MakeDecisionSet("tamper");
    (void)FreezeDecisions(set.decisions, set.protocol, set.sidecar, kUtc);

    std::string text = ReadFile(set.decisions);
    const std::size_t position = text.find("frozen");
    REQUIRE(position != std::string::npos);
    text[position] = 'F';
    WriteFile(set.decisions, text);

    const nlohmann::json verify = VerifyDecisions(set.decisions, set.sidecar);
    CHECK(verify["ok"] == false);
    CHECK(verify["expected"] != verify["actual"]);
}

TEST_CASE("a missing or malformed sidecar is an IoError", "[prereg]")
{
    const DecisionSet set = MakeDecisionSet("sidecar");

    SECTION("missing")
    {
        CHECK_THROWS_AS(VerifyDecisions(set.decisions, set.sidecar), IoError);
        try
        {
            (void)VerifyDecisions(set.decisions, set.sidecar);
        }
        catch (const IoError &error)
        {
            CHECK(error.exitCode() == 4);
        }
    }

    SECTION("malformed")
    {
        WriteFile(set.sidecar, "not-a-valid-sha256-digest\n");
        CHECK_THROWS_AS(VerifyDecisions(set.decisions, set.sidecar), IoError);
        try
        {
            (void)VerifyDecisions(set.decisions, set.sidecar);
        }
        catch (const IoError &error)
        {
            CHECK(error.exitCode() == 4);
        }
    }
}

TEST_CASE("freeze is deterministic when the timestamp is injected", "[prereg]")
{
    const DecisionSet first = MakeDecisionSet("det_a");
    const DecisionSet second = MakeDecisionSet("det_b");

    const nlohmann::json a = FreezeDecisions(first.decisions, first.protocol, first.sidecar, kUtc);
    const nlohmann::json b = FreezeDecisions(second.decisions, second.protocol, second.sidecar, kUtc);

    CHECK(a["decisions_sha256"] == b["decisions_sha256"]);
    CHECK(a["protocol_sha256"] == b["protocol_sha256"]);
}

TEST_CASE("the sidecar JSON records both decisions and protocol hashes", "[prereg]")
{
    const DecisionSet set = MakeDecisionSet("json");
    const nlohmann::json record =
        FreezeDecisions(set.decisions, set.protocol, set.sidecar, kUtc);

    const nlohmann::json json = ReadJson(fs::path(set.sidecar.string() + ".json"));
    CHECK(json["schema_version"] == 1);
    CHECK(json["frozen_at_utc"] == kUtc);
    CHECK(json["decisions_sha256"] == Sha256Decisions(set.decisions));
    CHECK(json["protocol_sha256"] == Sha256File(set.protocol));
    CHECK(json["decisions_sha256"] == record["decisions_sha256"]);
    CHECK(json["protocol_sha256"] == record["protocol_sha256"]);
}
