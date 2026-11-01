#include <raylidar/cli/command.h>
#include <raylidar/cli/commands.h>
#include <raylidar/common/artifacts.h>
#include <raylidar/common/cli_options.h>
#include <raylidar/common/io.h>
#include <raylidar/dataset/acquire.h>
#include <raylidar/dataset/preflight.h>

#include <yaml-cpp/yaml.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::ArtifactsRoot;
using raylidar::IoError;
using raylidar::ParsedArgs;
using raylidar::ParseArgs;
using raylidar::PhaseArtifactsDir;
using raylidar::UsageError;
using raylidar::WriteJson;
using raylidar::dataset::AcquisitionError;
using raylidar::dataset::InstallStaging;
using raylidar::dataset::PreflightExitCode;
using raylidar::dataset::PreflightReport;
using raylidar::dataset::PreflightReportJson;
using raylidar::dataset::RenderSummary;
using raylidar::dataset::RunPreflight;

constexpr const char *kDefaultStaging = "/data-20T/hybrid-lidar-sim/staging";

const std::vector<std::string> kDefaultForbiddenNames = {
    "calib_cam_to_cam.txt",
    "calib_velo_to_cam.txt",
    "calib_imu_to_velo.txt",
};

const std::set<std::string> kValueKeys = {
    "config", "staging", "root",  "sequence", "policy",
    "link-source", "estimated-extract-bytes", "manifest", "json",
};

const std::set<std::string> kFlagKeys = {"stream-extract", "dry-run", "strict"};

void PrintHelp(std::ostream &out)
{
    out << "Usage: raylidar_cli dataset <install|verify> [options]\n"
        << "\n"
        << "install  Install the official per-sequence subset from staged archives\n"
        << "verify   Run the dataset preflight and report readiness\n"
        << "\n"
        << "install options:\n"
        << "  --config <path>                  Dataset YAML (default: configs/dataset.yaml)\n"
        << "  --staging <path>                 Directory holding the staged archives\n"
        << "  --root <path>                    Override the dataset root from the config\n"
        << "  --sequence <name>                Override the sequence from the config\n"
        << "  --policy <auto|official|reuse_local_after_g1>\n"
        << "  --link-source <KEY=PATH>         Override one link source (repeatable;\n"
        << "                                   keys image_00, image_01, lidar,\n"
        << "                                   image_00_timestamps, image_01_timestamps)\n"
        << "  --stream-extract                 Extract archives in a single streaming pass\n"
        << "  --estimated-extract-bytes <n>    Required with --stream-extract\n"
        << "  --manifest <path>                Readiness manifest output path\n"
        << "  --dry-run                        Print the resolved plan and exit\n"
        << "\n"
        << "verify options:\n"
        << "  --config <path>                  Dataset YAML (default: configs/dataset.yaml)\n"
        << "  --json <path>                    Write the JSON preflight report\n"
        << "  --strict                         Escalate non-blocking failures to exit 2\n"
        << "\n"
        << "Exit codes: 0 success, 2 usage, 4 invalid config/link source,\n"
        << "            7 missing official component, 8 unsafe archive.\n";
}

std::string NodeString(const YAML::Node &node, const char *key, const std::string &fallback)
{
    if (!node || !node.IsMap())
    {
        return fallback;
    }
    const YAML::Node value = node[key];
    if (!value || !value.IsScalar())
    {
        return fallback;
    }
    return value.as<std::string>();
}

bool NodeBool(const YAML::Node &node, const char *key, bool fallback)
{
    if (!node || !node.IsMap())
    {
        return fallback;
    }
    const YAML::Node value = node[key];
    if (!value || !value.IsScalar())
    {
        return fallback;
    }
    try
    {
        return value.as<bool>();
    }
    catch (const YAML::Exception &)
    {
        return fallback;
    }
}

std::vector<std::string> NodeStringList(const YAML::Node &node, const char *key)
{
    std::vector<std::string> values;
    if (!node || !node.IsMap())
    {
        return values;
    }
    const YAML::Node sequence = node[key];
    if (!sequence || !sequence.IsSequence())
    {
        return values;
    }
    for (const YAML::Node &value : sequence)
    {
        if (value.IsScalar())
        {
            values.push_back(value.as<std::string>());
        }
    }
    return values;
}

std::map<std::string, std::string> NodeStringMap(const YAML::Node &node, const char *key)
{
    std::map<std::string, std::string> values;
    if (!node || !node.IsMap())
    {
        return values;
    }
    const YAML::Node map = node[key];
    if (!map || !map.IsMap())
    {
        return values;
    }
    for (const auto &entry : map)
    {
        if (entry.first.IsScalar() && entry.second.IsScalar())
        {
            values[entry.first.as<std::string>()] = entry.second.as<std::string>();
        }
    }
    return values;
}

YAML::Node LoadConfig(const ParsedArgs &parsed)
{
    const std::string path = parsed.Get("config", "configs/dataset.yaml");
    std::error_code error;
    if (!fs::exists(path, error))
    {
        if (parsed.Has("config"))
        {
            throw IoError("cannot open YAML file: " + path);
        }
        return YAML::Node(YAML::NodeType::Map);
    }
    return raylidar::ReadYaml(path);
}

YAML::Node DatasetNode(const YAML::Node &config)
{
    if (config && config.IsMap() && config["dataset"])
    {
        return config["dataset"];
    }
    return config;
}

std::string ResolveRoot(const ParsedArgs &parsed, const YAML::Node &dataset)
{
    std::string root = parsed.Get("root");
    if (root.empty())
    {
        root = NodeString(dataset, "root", "");
    }
    return root;
}

std::string ResolveSequence(const ParsedArgs &parsed, const YAML::Node &dataset)
{
    std::string sequence = parsed.Get("sequence");
    if (sequence.empty())
    {
        sequence = NodeString(dataset, "sequence", "");
    }
    return sequence;
}

std::map<std::string, std::string> ResolveLinkSources(const ParsedArgs &parsed,
                                                      const YAML::Node &dataset)
{
    std::map<std::string, std::string> sources = NodeStringMap(dataset, "link_sources");
    for (const std::string &override : parsed.GetAll("link-source"))
    {
        const std::size_t equals = override.find('=');
        if (equals == std::string::npos || equals == 0U)
        {
            throw UsageError("invalid --link-source '" + override + "'; expected KEY=PATH");
        }
        sources[override.substr(0U, equals)] = override.substr(equals + 1U);
    }
    return sources;
}

std::uintmax_t ParseBytes(const std::string &text)
{
    if (text.empty())
    {
        return 0U;
    }
    try
    {
        std::size_t consumed = 0;
        const unsigned long long value = std::stoull(text, &consumed);
        if (consumed != text.size())
        {
            throw UsageError("invalid byte count: " + text);
        }
        return static_cast<std::uintmax_t>(value);
    }
    catch (const UsageError &)
    {
        throw;
    }
    catch (const std::exception &)
    {
        throw UsageError("invalid byte count: " + text);
    }
}

int RunInstall(const ParsedArgs &parsed, const YAML::Node &dataset)
{
    const std::string root = ResolveRoot(parsed, dataset);
    const std::string sequence = ResolveSequence(parsed, dataset);
    if (root.empty())
    {
        std::cerr << "error: dataset root is not configured; pass --root\n";
        return 4;
    }
    if (sequence.empty())
    {
        std::cerr << "error: dataset sequence is not configured; pass --sequence\n";
        return 4;
    }

    std::string staging = parsed.Get("staging");
    if (staging.empty())
    {
        staging = NodeString(dataset, "staging_dir", kDefaultStaging);
    }

    std::string policy = parsed.Get("policy");
    if (policy.empty())
    {
        policy = "auto";
    }
    const bool acceptReuse = NodeBool(dataset, "accept_reuse_after_g1", false);

    std::map<std::string, std::string> linkSources = ResolveLinkSources(parsed, dataset);
    if (policy == "auto" && !acceptReuse)
    {
        // Auto may only fall back to reuse after G1 when the configuration
        // explicitly accepts it; without that, official images are required.
        if (linkSources.find("image_00") != linkSources.end())
        {
            linkSources["image_00"].clear();
        }
        if (linkSources.find("image_01") != linkSources.end())
        {
            linkSources["image_01"].clear();
        }
    }

    std::vector<std::string> cameras = NodeStringList(dataset, "cameras");
    if (cameras.empty())
    {
        cameras = {"image_00", "image_01"};
    }

    std::string manifest = parsed.Get("manifest");
    if (manifest.empty())
    {
        manifest = (PhaseArtifactsDir(1) / "readiness.json").string();
    }

    const std::uintmax_t estimated =
        ParseBytes(parsed.Get("estimated-extract-bytes"));

    if (parsed.Flag("dry-run"))
    {
        std::cout << "dataset install plan\n"
                  << "root: " << root << '\n'
                  << "sequence: " << sequence << '\n'
                  << "staging: " << staging << '\n'
                  << "policy: " << policy << '\n'
                  << "accept_reuse_after_g1: " << (acceptReuse ? "true" : "false") << '\n'
                  << "manifest: " << manifest << '\n'
                  << "stream_extract: " << (parsed.Flag("stream-extract") ? "true" : "false")
                  << '\n'
                  << "estimated_extract_bytes: " << estimated << "\n\nlinks:\n";
        const std::vector<std::string> keys = {
            "image_00", "image_01", "lidar", "image_00_timestamps", "image_01_timestamps"};
        for (const std::string &key : keys)
        {
            const auto found = linkSources.find(key);
            const std::string value = found == linkSources.end() ? std::string() : found->second;
            std::error_code error;
            const bool exists = !value.empty() && fs::exists(value, error);
            std::cout << "  " << key << ": " << (value.empty() ? "(none)" : value)
                      << (value.empty() ? "" : (exists ? " [exists]" : " [MISSING]")) << '\n';
        }
        std::cout << "\narchives:\n";
        std::error_code error;
        fs::directory_iterator iterator(staging, fs::directory_options::skip_permission_denied,
                                        error);
        if (error)
        {
            std::cout << "  (staging directory unavailable: " << staging << ")\n";
        }
        else
        {
            for (; iterator != fs::directory_iterator(); iterator.increment(error))
            {
                if (error)
                {
                    break;
                }
                std::error_code fileError;
                if (iterator->is_regular_file(fileError))
                {
                    std::cout << "  " << iterator->path().filename().string() << " ("
                              << iterator->file_size(fileError) << " bytes)\n";
                }
            }
        }
        return 0;
    }

    const nlohmann::json manifestJson =
        InstallStaging(staging, root, sequence, cameras, policy, linkSources, manifest,
                       parsed.Flag("stream-extract"), estimated);

    std::cout << "dataset install complete\n"
              << "sequence: " << manifestJson.value("sequence", sequence) << '\n'
              << "policy: " << manifestJson.value("policy", policy) << '\n'
              << "image_provenance: "
              << manifestJson.value("image_provenance", std::string()) << '\n'
              << "archives: " << manifestJson["archives"].size() << '\n'
              << "skipped_archives: " << manifestJson["skipped_archives"].size() << '\n'
              << "manifest: " << manifest << '\n';
    return 0;
}

int RunVerify(const ParsedArgs &parsed, const YAML::Node &dataset)
{
    std::string root = ResolveRoot(parsed, dataset);
    const std::string sequence = ResolveSequence(parsed, dataset);
    if (root.empty())
    {
        std::cerr << "error: dataset root is not configured; pass --root\n";
        return 4;
    }
    if (sequence.empty())
    {
        std::cerr << "error: dataset sequence is not configured; pass --sequence\n";
        return 4;
    }

    std::vector<std::string> cameras = NodeStringList(dataset, "cameras");
    if (cameras.empty())
    {
        cameras = {"image_00", "image_01"};
    }

    std::vector<std::string> forbiddenNames =
        NodeStringList(dataset, "forbidden_calibration_names");
    if (forbiddenNames.empty())
    {
        forbiddenNames = kDefaultForbiddenNames;
    }

    std::vector<fs::path> scanRoots;
    for (const std::string &value : NodeStringList(dataset, "scan_roots"))
    {
        scanRoots.emplace_back(value);
    }

    const PreflightReport report =
        RunPreflight(root, sequence, cameras, forbiddenNames, scanRoots);
    std::cout << RenderSummary(report);

    const std::string jsonPath = parsed.Get("json");
    if (!jsonPath.empty())
    {
        WriteJson(jsonPath, PreflightReportJson(report));
    }
    return PreflightExitCode(report, parsed.Flag("strict"));
}
} // namespace

int RunDataset(std::span<const std::string_view> args)
{
    try
    {
        const ParsedArgs parsed = ParseArgs(args, kValueKeys, kFlagKeys);
        if (parsed.Flag("help"))
        {
            PrintHelp(std::cout);
            return 0;
        }

        const std::string subcommand = parsed.Get("");
        if (subcommand.empty())
        {
            PrintHelp(std::cerr);
            return 2;
        }
        if (subcommand != "install" && subcommand != "verify")
        {
            std::cerr << "error: unknown dataset subcommand '" << subcommand << "'\n\n";
            PrintHelp(std::cerr);
            return 2;
        }

        const YAML::Node config = LoadConfig(parsed);
        const YAML::Node dataset = DatasetNode(config);
        if (subcommand == "install")
        {
            return RunInstall(parsed, dataset);
        }
        return RunVerify(parsed, dataset);
    }
    catch (const UsageError &error)
    {
        std::cerr << "error: " << error.what() << "\n\n";
        PrintHelp(std::cerr);
        return error.exitCode();
    }
    catch (const AcquisitionError &error)
    {
        std::cerr << "error: " << error.what() << '\n';
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
        return 4;
    }
}

RAYLIDAR_REGISTER_COMMAND(
    "dataset", "Install the official per-sequence subset or verify dataset readiness",
    RunDataset);
