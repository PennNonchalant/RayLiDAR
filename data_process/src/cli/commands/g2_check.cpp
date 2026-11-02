#include <raylidar/cli/command.h>
#include <raylidar/cli/commands_phase2.h>
#include <raylidar/common/artifacts.h>
#include <raylidar/common/cli_options.h>
#include <raylidar/common/io.h>
#include <raylidar/g2/g2.h>

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace
{
namespace fs = std::filesystem;
using raylidar::IoError;
using raylidar::ParseArgs;
using raylidar::ParsedArgs;
using raylidar::PhaseArtifactsDir;
using raylidar::UsageError;

constexpr int kExitInvalidInput = 4;
constexpr int kExitAssertionFailure = 5;

void PrintHelp(std::ostream &out)
{
    out << "Usage: raylidar_cli g2-check [options]\n"
        << "\n"
        << "Recompute every Phase-2 G2 success criterion from stored artifacts and\n"
        << "write g2/G2_REPORT.json + g2/G2_REPORT.md. Missing evidence is a failure;\n"
        << "there is no CLI flag that can set `pass`.\n"
        << "\n"
        << "Options:\n"
        << "  --config <path>          Phase-2 YAML (default: configs/phase2.yaml)\n"
        << "  --artifacts-dir <path>   Phase-2 artifact directory\n"
        << "                           (default: artifacts.phase_dir from the config)\n"
        << "  --out <path>             Report base path\n"
        << "                           (default: <artifacts-dir>/g2/G2_REPORT)\n"
        << "  --help                   Show this help\n"
        << "\n"
        << "Exit codes: 0 all checks pass, 2 usage, 4 invalid/missing input,\n"
        << "            5 one or more G2 checks failed.\n";
}

std::string NodeString(const YAML::Node &node, const char *key)
{
    if (!node || !node.IsMap())
    {
        return {};
    }
    const YAML::Node value = node[key];
    if (!value || !value.IsScalar())
    {
        return {};
    }
    return value.as<std::string>();
}

void WriteText(const fs::path &path, const std::string &text)
{
    if (path.has_parent_path())
    {
        raylidar::EnsureDir(path.parent_path());
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
    {
        throw IoError("cannot write " + path.string(), kExitInvalidInput);
    }
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!output)
    {
        throw IoError("failed to write " + path.string(), kExitInvalidInput);
    }
}

int Execute(std::span<const std::string_view> args)
{
    const ParsedArgs parsed = ParseArgs(args, {"config", "artifacts-dir", "out"}, {"help"});
    if (parsed.Flag("help"))
    {
        PrintHelp(std::cout);
        return 0;
    }

    const fs::path configPath = parsed.Get("config", "configs/phase2.yaml");
    fs::path artifactsDir = parsed.Get("artifacts-dir", "");
    if (artifactsDir.empty())
    {
        std::error_code error;
        if (fs::is_regular_file(configPath, error))
        {
            const YAML::Node config = raylidar::ReadYaml(configPath);
            const std::string configured = NodeString(config["artifacts"], "phase_dir");
            if (!configured.empty())
            {
                artifactsDir = configured;
            }
        }
        if (artifactsDir.empty())
        {
            artifactsDir = PhaseArtifactsDir(2);
        }
    }

    const fs::path outBase = parsed.Get("out", (artifactsDir / "g2" / "G2_REPORT").string());
    if (outBase.empty())
    {
        throw IoError("--out requires a path", kExitInvalidInput);
    }

    const nlohmann::json report = raylidar::g2::EvaluateG2(artifactsDir, configPath);
    raylidar::WriteJson(outBase.string() + ".json", report);
    const std::string markdown = raylidar::g2::RenderG2Markdown(report);
    WriteText(outBase.string() + ".md", markdown);
    std::cout << markdown;

    return report.value("pass", false) ? 0 : kExitAssertionFailure;
}
} // namespace

int RunG2Check(std::span<const std::string_view> args)
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

RAYLIDAR_REGISTER_COMMAND("g2-check", "Recompute the Phase-2 G2 evidence and write G2_REPORT", RunG2Check);
