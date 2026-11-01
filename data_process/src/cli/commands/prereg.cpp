#include <raylidar/cli/command.h>
#include <raylidar/cli/commands.h>
#include <raylidar/common/artifacts.h>
#include <raylidar/common/cli_options.h>
#include <raylidar/common/io.h>
#include <raylidar/prereg/prereg.h>

#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::IoError;
using raylidar::ParsedArgs;
using raylidar::ParseArgs;
using raylidar::PhaseArtifact;
using raylidar::UsageError;
using raylidar::prereg::FreezeDecisions;
using raylidar::prereg::VerifyDecisions;

constexpr int kExitMismatch = 6;
constexpr int kExitInvalidInput = 4;
constexpr int kExitUsage = 2;

void PrintHelp(std::ostream &out)
{
    out << "Usage: raylidar_cli prereg <freeze|verify> [options]\n"
        << "\n"
        << "Freeze or verify the pre-registered evaluation protocol.\n"
        << "\n"
        << "Subcommands:\n"
        << "  freeze    Timestamp docs/decisions.md, then write the SHA-256 sidecar\n"
        << "  verify    Recompute the decisions hash and compare it with the sidecar\n"
        << "\n"
        << "Options:\n"
        << "  --decisions <path>  Decisions document (default: docs/decisions.md)\n"
        << "  --protocol <path>   Protocol YAML (default: configs/protocol.yaml)\n"
        << "  --sidecar <path>    Hash sidecar (default: <artifacts>/01/decisions.sha256)\n"
        << "  --help              Show this help\n"
        << "\n"
        << "Exit codes: 0 success, 2 usage, 4 invalid/missing input, 6 hash mismatch.\n";
}

struct Options
{
    fs::path decisions = "docs/decisions.md";
    fs::path protocol = "configs/protocol.yaml";
    fs::path sidecar;
};

Options ParseOptions(const ParsedArgs &parsed)
{
    Options options;
    options.decisions = parsed.Get("decisions", options.decisions.string());
    options.protocol = parsed.Get("protocol", options.protocol.string());
    options.sidecar = fs::path(parsed.Get("sidecar", PhaseArtifact(1, "decisions.sha256").string()));
    return options;
}

int Execute(std::span<const std::string_view> args)
{
    const ParsedArgs parsed = ParseArgs(args, {"decisions", "protocol", "sidecar"}, {"help"});
    if (parsed.Flag("help"))
    {
        PrintHelp(std::cout);
        return 0;
    }

    const std::vector<std::string> positional = parsed.GetAll("");
    if (positional.empty())
    {
        std::cerr << "error: missing prereg subcommand (expected 'freeze' or 'verify')\n\n";
        PrintHelp(std::cerr);
        return kExitUsage;
    }

    const std::string &subcommand = positional.front();
    const Options options = ParseOptions(parsed);

    if (subcommand == "freeze")
    {
        const nlohmann::json record =
            FreezeDecisions(options.decisions, options.protocol, options.sidecar);
        std::cout << "frozen " << record["decisions_sha256"].get<std::string>() << " at "
                  << record["frozen_at_utc"].get<std::string>() << '\n';
        return 0;
    }

    if (subcommand == "verify")
    {
        const nlohmann::json record = VerifyDecisions(options.decisions, options.sidecar);
        if (record["ok"].get<bool>())
        {
            std::cout << "OK " << record["actual"].get<std::string>() << '\n';
            return 0;
        }
        std::cerr << "MISMATCH expected=" << record["expected"].get<std::string>()
                  << " actual=" << record["actual"].get<std::string>() << '\n';
        return kExitMismatch;
    }

    std::cerr << "error: unknown prereg subcommand '" << subcommand << "'\n\n";
    PrintHelp(std::cerr);
    return kExitUsage;
}
} // namespace

int RunPrereg(std::span<const std::string_view> args)
{
    try
    {
        return Execute(args);
    }
    catch (const UsageError &error)
    {
        std::cerr << "error: " << error.what() << "\n\n";
        PrintHelp(std::cerr);
        return error.exitCode();
    }
    catch (const IoError &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return error.exitCode();
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return kExitInvalidInput;
    }
}

RAYLIDAR_REGISTER_COMMAND(
    "prereg", "Freeze or verify the pre-registered evaluation protocol", RunPrereg);
