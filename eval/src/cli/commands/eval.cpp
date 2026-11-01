#include <raylidar/cli/command.h>
#include <raylidar/cli/commands_phase3.h>

#include <raylidar/common/artifacts.h>
#include <raylidar/common/cli_options.h>
#include <raylidar/common/io.h>
#include <raylidar/eval/aggregate.h>
#include <raylidar/eval/output.h>
#include <raylidar/eval/scene.h>
#include <raylidar/io/sim_io.h>
#include <raylidar/rays/rays.h>

#include <yaml-cpp/yaml.h>

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::IoError;
using raylidar::ParsedArgs;
using raylidar::ParseArgs;
using raylidar::PhaseArtifactsDir;
using raylidar::UsageError;
using Eigen::Vector3d;
using raylidar::eval::EvalReport;
using raylidar::eval::FrameRawCounts;
using raylidar::eval::MetricsInputs;
using raylidar::eval::ProceduralScene;
using raylidar::eval::ProceduralStreetConfig;
using raylidar::eval::ProtocolConfig;
using raylidar::eval::RaySample;

constexpr int kExitInvalidInput = 4;
constexpr int kExitProtocolDrift = 6;
constexpr std::uint32_t kMaxGridAxis = 4096;
constexpr std::uint64_t kMaxGridRays = 4000000;
constexpr int kMaxFrames = 1024;

[[noreturn]] void Fail(const std::string &message, const int exitCode = kExitInvalidInput)
{
    throw IoError(message, exitCode);
}

// ---------------------------------------------------------------------------
// YAML helpers
// ---------------------------------------------------------------------------

YAML::Node RequireNode(const YAML::Node &parent, const char *key, const std::string &where)
{
    const YAML::Node node = parent[key];
    if (!node || !node.IsDefined())
    {
        Fail(where + " is missing '" + key + "'");
    }
    return node;
}

std::string NodeString(const YAML::Node &parent, const char *key, const std::string &where)
{
    const YAML::Node node = RequireNode(parent, key, where);
    if (!node.IsScalar())
    {
        Fail(where + "." + key + " must be a scalar");
    }
    return node.as<std::string>();
}

double NodeDouble(const YAML::Node &parent, const char *key, const std::string &where)
{
    const YAML::Node node = RequireNode(parent, key, where);
    if (!node.IsScalar())
    {
        Fail(where + "." + key + " must be a number");
    }
    try
    {
        return node.as<double>();
    }
    catch (const std::exception &)
    {
        Fail(where + "." + key + " must be a number");
    }
}

int NodeInt(const YAML::Node &parent, const char *key, const std::string &where)
{
    const YAML::Node node = RequireNode(parent, key, where);
    if (!node.IsScalar())
    {
        Fail(where + "." + key + " must be an integer");
    }
    try
    {
        return node.as<int>();
    }
    catch (const std::exception &)
    {
        Fail(where + "." + key + " must be an integer");
    }
}

std::uint64_t NodeUint(const YAML::Node &parent, const char *key, const std::string &where)
{
    const YAML::Node node = RequireNode(parent, key, where);
    if (!node.IsScalar())
    {
        Fail(where + "." + key + " must be an unsigned integer");
    }
    try
    {
        return node.as<std::uint64_t>();
    }
    catch (const std::exception &)
    {
        Fail(where + "." + key + " must be an unsigned integer");
    }
}

// ---------------------------------------------------------------------------
// Placeholder configuration
// ---------------------------------------------------------------------------

struct PlaceholderSettings
{
    std::string generator;
    std::uint64_t seed = 0;
    double groundLengthM = 0.0;
    double groundWidthM = 0.0;
    int boxCount = 0;
    int thinBoxCount = 0;
    std::array<Vector3d, 2> cameras = {Vector3d::Zero(), Vector3d::Zero()};
    int gridWidth = 0;
    int gridHeight = 0;
    double hfovDeg = 0.0;
    double vfovDeg = 0.0;
    int frames = 0;
    double frameStrideM = 0.0;
    std::string predictorName;
    double rangeBiasM = 0.0;
    double rangeScale = 0.0;
    int missEveryN = 0;
};

Vector3d NodeVector3(const YAML::Node &parent, const char *key, const std::string &where)
{
    const YAML::Node node = RequireNode(parent, key, where);
    if (!node.IsSequence() || node.size() != 3U)
    {
        Fail(where + "." + key + " must hold three numbers");
    }
    try
    {
        return Vector3d(node[0].as<double>(), node[1].as<double>(), node[2].as<double>());
    }
    catch (const std::exception &)
    {
        Fail(where + "." + key + " must hold three numbers");
    }
}

PlaceholderSettings LoadPlaceholder(const YAML::Node &root,
                                    const std::optional<std::uint64_t> seedOverride,
                                    const std::optional<int> framesOverride)
{
    const YAML::Node node = RequireNode(root, "placeholder", "config");
    if (!node.IsMap())
    {
        Fail("config.placeholder must be a mapping");
    }
    const std::string where = "config.placeholder";

    PlaceholderSettings settings;
    settings.generator = NodeString(node, "generator", where);
    if (settings.generator != "procedural_street_v1")
    {
        Fail("unsupported placeholder generator '" + settings.generator + "'");
    }
    settings.seed = NodeUint(node, "seed", where);
    settings.groundLengthM = NodeDouble(node, "ground_length_m", where);
    settings.groundWidthM = NodeDouble(node, "ground_width_m", where);
    settings.boxCount = NodeInt(node, "box_count", where);
    settings.thinBoxCount = NodeInt(node, "thin_box_count", where);

    const YAML::Node cameras = RequireNode(node, "cameras", where);
    if (!cameras.IsSequence() || cameras.size() != 2U)
    {
        Fail(where + ".cameras must hold exactly two cameras");
    }
    for (std::size_t index = 0; index < 2U; ++index)
    {
        const YAML::Node camera = cameras[index];
        if (!camera.IsMap())
        {
            Fail(where + ".cameras entries must be mappings");
        }
        const int id = NodeInt(camera, "id", where + ".cameras");
        if (id != static_cast<int>(index))
        {
            Fail(where + ".cameras ids must be 0 then 1");
        }
        settings.cameras[index] = NodeVector3(camera, "origin_m", where + ".cameras");
    }

    const YAML::Node grid = RequireNode(node, "ray_grid", where);
    if (!grid.IsMap())
    {
        Fail(where + ".ray_grid must be a mapping");
    }
    settings.gridWidth = NodeInt(grid, "width", where + ".ray_grid");
    settings.gridHeight = NodeInt(grid, "height", where + ".ray_grid");
    settings.hfovDeg = NodeDouble(grid, "hfov_deg", where + ".ray_grid");
    settings.vfovDeg = NodeDouble(grid, "vfov_deg", where + ".ray_grid");

    settings.frames = NodeInt(node, "frames", where);
    settings.frameStrideM = NodeDouble(node, "frame_stride_m", where);

    const YAML::Node predictor = RequireNode(node, "predictor", where);
    if (!predictor.IsMap())
    {
        Fail(where + ".predictor must be a mapping");
    }
    settings.predictorName = NodeString(predictor, "name", where + ".predictor");
    settings.rangeBiasM = NodeDouble(predictor, "range_bias_m", where + ".predictor");
    settings.rangeScale = NodeDouble(predictor, "range_scale", where + ".predictor");
    settings.missEveryN = NodeInt(predictor, "miss_every_n", where + ".predictor");

    if (seedOverride.has_value())
    {
        settings.seed = *seedOverride;
    }
    if (framesOverride.has_value())
    {
        settings.frames = *framesOverride;
    }

    if (settings.gridWidth < 1 ||
        static_cast<std::uint32_t>(settings.gridWidth) > kMaxGridAxis ||
        settings.gridHeight < 1 ||
        static_cast<std::uint32_t>(settings.gridHeight) > kMaxGridAxis)
    {
        Fail("ray_grid width/height must be in [1, 4096]");
    }
    if (static_cast<std::uint64_t>(settings.gridWidth) *
            static_cast<std::uint64_t>(settings.gridHeight) >
        kMaxGridRays)
    {
        Fail("ray_grid exceeds the 4,000,000-ray cap");
    }
    if (settings.frames < 1 || settings.frames > kMaxFrames)
    {
        Fail("placeholder.frames must be in [1, 1024]");
    }
    if (!std::isfinite(settings.frameStrideM) || settings.frameStrideM < 0.0)
    {
        Fail("placeholder.frame_stride_m must be a finite non-negative number");
    }
    if (!(settings.groundLengthM >= 4.0) || settings.groundLengthM > 10000.0 ||
        !(settings.groundWidthM >= 2.0) || settings.groundWidthM > 10000.0)
    {
        Fail("placeholder ground dimensions are invalid");
    }
    if (settings.boxCount < 0 || settings.boxCount > 100000 ||
        settings.thinBoxCount < 0 || settings.thinBoxCount > settings.boxCount)
    {
        Fail("placeholder box counts are invalid");
    }
    if (!(settings.hfovDeg > 0.0) || settings.hfovDeg > 180.0 ||
        !(settings.vfovDeg > 0.0) || settings.vfovDeg > 180.0)
    {
        Fail("ray_grid FOV must be in (0, 180] degrees");
    }
    if (!std::isfinite(settings.rangeBiasM) || !std::isfinite(settings.rangeScale))
    {
        Fail("predictor range parameters must be finite");
    }
    if (settings.missEveryN < 0)
    {
        Fail("predictor.miss_every_n must be non-negative");
    }
    return settings;
}

// ---------------------------------------------------------------------------
// Ray generation
// ---------------------------------------------------------------------------

Vector3d RayDirection(const int column, const int row, const int width, const int height,
                      const double hfovRad, const double vfovRad)
{
    const double yaw =
        width > 1 ? (static_cast<double>(column) / static_cast<double>(width - 1) - 0.5) *
                        hfovRad
                  : 0.0;
    const double pitch =
        height > 1 ? (0.5 - static_cast<double>(row) / static_cast<double>(height - 1)) *
                         vfovRad
                   : 0.0;
    const double cosPitch = std::cos(pitch);
    return Vector3d(cosPitch * std::cos(yaw), cosPitch * std::sin(yaw), std::sin(pitch));
}

std::uint32_t PrimitiveToTriangleId(const std::uint32_t primitiveId)
{
    // The PLY layout is 2 ground triangles, then 12 per box; `prim_id` in
    // hits.bin is a 0-based triangle id (docs/sim-io.md).
    return primitiveId == 0U ? 0U : 2U + (primitiveId - 1U) * 12U;
}

struct GeneratedArtifacts
{
    std::vector<raylidar::io::GtRay> rays;
    std::vector<raylidar::io::Hit> hits;
    std::vector<RaySample> samples;
    std::vector<FrameRawCounts> rawByFrame;
};

GeneratedArtifacts GenerateRays(const ProceduralScene &scene,
                                const PlaceholderSettings &settings,
                                const ProtocolConfig &protocol)
{
    GeneratedArtifacts artifacts;
    const double hfovRad = settings.hfovDeg * (std::acos(-1.0) / 180.0);
    const double vfovRad = settings.vfovDeg * (std::acos(-1.0) / 180.0);
    const std::uint64_t gridRays = static_cast<std::uint64_t>(settings.gridWidth) *
                                   static_cast<std::uint64_t>(settings.gridHeight);

    for (int frame = 0; frame < settings.frames; ++frame)
    {
        FrameRawCounts raw;
        raw.frame = static_cast<std::uint32_t>(frame);
        for (std::uint16_t camera = 0; camera < 2U; ++camera)
        {
            Vector3d origin = settings.cameras[camera];
            origin.x() += static_cast<double>(frame) * settings.frameStrideM;
            for (int row = 0; row < settings.gridHeight; ++row)
            {
                for (int column = 0; column < settings.gridWidth; ++column)
                {
                    const std::uint64_t gridIndex =
                        static_cast<std::uint64_t>(row) *
                            static_cast<std::uint64_t>(settings.gridWidth) +
                        static_cast<std::uint64_t>(column);
                    const Vector3d direction = RayDirection(column, row, settings.gridWidth,
                                                            settings.gridHeight, hfovRad,
                                                            vfovRad);
                    ++raw.raw.raw_total;
                    const auto hit = raylidar::eval::RaycastScene(
                        scene, origin, direction, protocol.t_min_m, protocol.t_max_m);
                    if (!hit.has_value())
                    {
                        continue;
                    }
                    const raylidar::rays::Bucket bucket =
                        raylidar::rays::BucketOf(hit->t);
                    ++raw.raw.raw_bucket[static_cast<std::size_t>(bucket)];
                    if (hit->t < protocol.r_min_m || hit->t > protocol.t_max_m)
                    {
                        continue;
                    }

                    // Unique within a frame across both cameras; the low 32
                    // bits are camera * gridRays + grid index.
                    const std::uint64_t frameIndex =
                        static_cast<std::uint64_t>(camera) * gridRays + gridIndex;
                    const std::uint64_t rayId =
                        (static_cast<std::uint64_t>(frame) << 32U) | frameIndex;
                    raylidar::io::GtRay ray;
                    ray.ray_id = rayId;
                    ray.frame = static_cast<std::uint32_t>(frame);
                    ray.camera = camera;
                    ray.origin_W = origin;
                    ray.dir_W = direction;
                    ray.t_gt = hit->t;
                    ray.t_min = protocol.t_min_m;
                    ray.t_max = protocol.t_max_m;
                    artifacts.rays.push_back(ray);

                    raylidar::io::Hit prediction;
                    prediction.ray_id = rayId;
                    const bool miss = settings.missEveryN > 0 &&
                                      (frameIndex %
                                           static_cast<std::uint64_t>(settings.missEveryN) ==
                                       0U);
                    if (miss)
                    {
                        prediction.hit = false;
                        prediction.t_pred = std::numeric_limits<double>::infinity();
                        prediction.prim_id = raylidar::io::kPrimIdUnavailable;
                    }
                    else
                    {
                        double tPred = hit->t * (1.0 + settings.rangeScale) +
                                       settings.rangeBiasM +
                                       0.03 * std::sin(0.001 * static_cast<double>(rayId));
                        tPred = std::min(std::max(tPred, protocol.t_min_m),
                                         protocol.t_max_m);
                        prediction.hit = true;
                        prediction.t_pred = tPred;
                        prediction.prim_id =
                            PrimitiveToTriangleId(hit->primitive_id);
                    }
                    artifacts.hits.push_back(prediction);

                    RaySample sample;
                    sample.ray_id = rayId;
                    sample.frame = static_cast<std::uint32_t>(frame);
                    sample.camera = camera;
                    sample.bucket = bucket;
                    sample.t_gt = hit->t;
                    sample.t_min = protocol.t_min_m;
                    sample.t_max = protocol.t_max_m;
                    sample.origin_W = origin;
                    sample.dir_W = direction;
                    sample.t_pred = prediction.hit
                                        ? prediction.t_pred
                                        : std::numeric_limits<double>::infinity();
                    sample.occluded = false;
                    artifacts.samples.push_back(sample);
                }
            }
        }
        artifacts.rawByFrame.push_back(raw);
    }
    return artifacts;
}

// ---------------------------------------------------------------------------
// Paths and sidecars
// ---------------------------------------------------------------------------

fs::path ResolveOutDir(const ParsedArgs &parsed, const YAML::Node &root)
{
    if (parsed.Has("out-dir"))
    {
        const std::string given = parsed.Get("out-dir");
        if (given.empty())
        {
            throw UsageError("--out-dir must not be empty");
        }
        return fs::path(given);
    }
    const YAML::Node artifacts = root["artifacts"];
    if (artifacts && artifacts.IsMap())
    {
        const YAML::Node configured = artifacts["phase_dir"];
        if (configured && configured.IsScalar())
        {
            return fs::path(configured.as<std::string>()) / "eval";
        }
    }
    return PhaseArtifactsDir(3) / "eval";
}

fs::path ResolveProtocolDoc(const ParsedArgs &parsed, const fs::path &configPath)
{
    const std::string given = parsed.Get("protocol-doc");
    if (!given.empty())
    {
        return fs::path(given);
    }
    const std::array<fs::path, 3> candidates = {
        configPath.parent_path() / "protocol.md",
        configPath.parent_path().parent_path() / "docs" / "protocol.md",
        fs::path("docs") / "protocol.md",
    };
    for (const auto &candidate : candidates)
    {
        std::error_code error;
        if (fs::is_regular_file(candidate, error))
        {
            return candidate;
        }
    }
    Fail("cannot locate docs/protocol.md for the protocol hash");
}

bool IsSha256Hex(const std::string &text)
{
    if (text.size() != 64U)
    {
        return false;
    }
    for (const char character : text)
    {
        const bool digit = character >= '0' && character <= '9';
        const bool lower = character >= 'a' && character <= 'f';
        if (!digit && !lower)
        {
            return false;
        }
    }
    return true;
}

std::string Trim(const std::string &text)
{
    const std::string whitespace = " \t\r\n";
    const std::size_t first = text.find_first_not_of(whitespace);
    if (first == std::string::npos)
    {
        return {};
    }
    const std::size_t last = text.find_last_not_of(whitespace);
    return text.substr(first, last - first + 1U);
}

void VerifySidecar(const fs::path &path)
{
    std::ifstream input(path);
    if (!input)
    {
        Fail("cannot read sidecar " + path.string());
    }
    std::string line;
    bool sawEntry = false;
    while (std::getline(input, line))
    {
        const std::string trimmed = Trim(line);
        if (trimmed.empty())
        {
            continue;
        }
        const std::size_t split = trimmed.find_first_of(" \t");
        if (split == std::string::npos)
        {
            Fail("malformed sidecar line in " + path.string(), kExitProtocolDrift);
        }
        const std::string hash = trimmed.substr(0U, split);
        if (!IsSha256Hex(hash))
        {
            Fail("malformed SHA-256 in " + path.string(), kExitProtocolDrift);
        }
        std::string target = Trim(trimmed.substr(split));
        if (!target.empty() && target.front() == '*')
        {
            target.erase(target.begin());
        }
        target = Trim(target);
        if (target.empty())
        {
            Fail("sidecar line has no path in " + path.string(), kExitProtocolDrift);
        }
        const std::string actual = raylidar::Sha256File(target);
        if (actual != hash)
        {
            Fail("hash mismatch for " + target, kExitProtocolDrift);
        }
        sawEntry = true;
    }
    if (!sawEntry)
    {
        Fail("sidecar is empty: " + path.string());
    }
}

// ---------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------

void PrintHelp(std::ostream &out)
{
    out << "Usage: raylidar_cli eval <placeholder|validate> [options]\n"
        << "\n"
        << "Placeholder mode generates the deterministic procedural street scene,\n"
        << "casts the configured ray grid, runs the synthetic predictor, and writes\n"
        << "rays_placeholder.bin, hits_placeholder.bin, scene.ply, scene_manifest.json,\n"
        << "sim_io_manifest.json, metrics.json, and metrics.csv under --out-dir. Every\n"
        << "artifact is validated before exit 0.\n"
        << "\n"
        << "Validate mode re-checks metrics.json/metrics.csv and optional SHA-256\n"
        << "sidecars without rewriting anything.\n"
        << "\n"
        << "Options (placeholder):\n"
        << "  --config <path>        Phase-3 YAML (default: configs/phase3.yaml)\n"
        << "  --out-dir <path>       Output directory (default: <phase_dir>/eval)\n"
        << "  --seed <uint64>        Override the procedural scene seed\n"
        << "  --frames <int>         Override the frame count\n"
        << "  --protocol-doc <path>  Protocol markdown to hash\n"
        << "\n"
        << "Options (validate):\n"
        << "  --metrics <path>       metrics.json to validate (required)\n"
        << "  --csv <path>           metrics.csv to validate (required)\n"
        << "  --protocol <path>      optional protocol SHA-256 sidecar\n"
        << "  --sim-io <path>        optional sim-io SHA-256 sidecar\n"
        << "\n"
        << "  --help                 Show this help\n"
        << "\n"
        << "Exit codes: 0 success, 2 usage, 4 invalid/missing input, 6 schema/hash or\n"
        << "            protocol drift.\n";
}

int ExecutePlaceholder(const ParsedArgs &parsed)
{
    const fs::path configPath = parsed.Get("config", "configs/phase3.yaml");
    const ProtocolConfig protocol = raylidar::eval::LoadProtocolConfig(configPath);
    const YAML::Node root = raylidar::ReadYaml(configPath);

    std::optional<std::uint64_t> seedOverride;
    const std::string seedText = parsed.Get("seed");
    if (!seedText.empty())
    {
        try
        {
            std::size_t consumed = 0;
            const auto value = std::stoull(seedText, &consumed);
            if (consumed != seedText.size())
            {
                throw std::invalid_argument("trailing characters");
            }
            seedOverride = static_cast<std::uint64_t>(value);
        }
        catch (const std::exception &)
        {
            throw UsageError("--seed must be an unsigned integer");
        }
    }
    std::optional<int> framesOverride;
    const std::string framesText = parsed.Get("frames");
    if (!framesText.empty())
    {
        try
        {
            std::size_t consumed = 0;
            const int value = std::stoi(framesText, &consumed);
            if (consumed != framesText.size() || value < 1 || value > kMaxFrames)
            {
                throw std::invalid_argument("out of range");
            }
            framesOverride = value;
        }
        catch (const std::exception &)
        {
            throw UsageError("--frames must be an integer in [1, 1024]");
        }
    }

    const PlaceholderSettings settings =
        LoadPlaceholder(root, seedOverride, framesOverride);
    const fs::path outDir = ResolveOutDir(parsed, root);
    const fs::path protocolDoc = ResolveProtocolDoc(parsed, configPath);

    ProceduralStreetConfig sceneConfig;
    sceneConfig.seed = settings.seed;
    sceneConfig.ground_length_m = settings.groundLengthM;
    sceneConfig.ground_width_m = settings.groundWidthM;
    sceneConfig.box_count = settings.boxCount;
    sceneConfig.thin_box_count = settings.thinBoxCount;
    const ProceduralScene scene = raylidar::eval::GenerateProceduralStreet(sceneConfig);

    raylidar::EnsureDir(outDir);

    const fs::path scenePath = outDir / "scene.ply";
    const fs::path sceneManifestPath = outDir / "scene_manifest.json";
    const fs::path raysPath = outDir / "rays_placeholder.bin";
    const fs::path hitsPath = outDir / "hits_placeholder.bin";
    const fs::path simIoPath = outDir / "sim_io_manifest.json";
    const fs::path metricsPath = outDir / "metrics.json";
    const fs::path csvPath = outDir / "metrics.csv";

    raylidar::eval::WriteProceduralPly(scenePath, scene);
    raylidar::WriteJson(sceneManifestPath,
                        raylidar::eval::BuildSceneManifest(scene, scenePath));

    const GeneratedArtifacts artifacts = GenerateRays(scene, settings, protocol);
    raylidar::io::WriteGtRays(raysPath, artifacts.rays);
    raylidar::io::WriteHits(hitsPath, artifacts.hits);

    const std::string configSha = raylidar::Sha256File(configPath);
    const std::string protocolSha = raylidar::Sha256File(protocolDoc);
    const std::string raySha = raylidar::Sha256File(raysPath);
    const std::string meshSha = raylidar::Sha256File(scenePath);
    const std::string hitSha = raylidar::Sha256File(hitsPath);

    nlohmann::json simIo;
    simIo["schema_version"] = 1;
    simIo["frame"] = "W";
    simIo["units"] = "m";
    nlohmann::json entries = nlohmann::json::array();
    entries.push_back({{"kind", "gt_rays"},
                       {"path", raysPath.filename().string()},
                       {"sha256", raySha},
                       {"version", raylidar::io::kGtRaysVersion},
                       {"record_size", raylidar::io::kGtRaysRecordSize},
                       {"count", artifacts.rays.size()},
                       {"frame", "W"},
                       {"units", "m"}});
    entries.push_back({{"kind", "hits"},
                       {"path", hitsPath.filename().string()},
                       {"sha256", hitSha},
                       {"version", raylidar::io::kHitsVersion},
                       {"record_size", raylidar::io::kHitsRecordSize},
                       {"count", artifacts.hits.size()},
                       {"frame", "W"},
                       {"units", "m"}});
    simIo["entries"] = std::move(entries);
    raylidar::WriteJson(simIoPath, simIo);

    const EvalReport report =
        raylidar::eval::EvaluateAll(artifacts.samples, artifacts.rawByFrame);

    MetricsInputs inputs;
    inputs.mode = "placeholder";
    inputs.config_sha256 = configSha;
    inputs.protocol_sha256 = protocolSha;
    inputs.ray_set_sha256 = raySha;
    inputs.mesh_sha256 = meshSha;
    inputs.hit_set_sha256 = hitSha;
    inputs.scene_generator = settings.generator;
    inputs.scene_seed = settings.seed;
    inputs.scene_box_count = static_cast<std::uint64_t>(settings.boxCount);
    inputs.scene_thin_box_count = static_cast<std::uint64_t>(settings.thinBoxCount);
    inputs.predictor = settings.predictorName;
    inputs.predictor_parameters = {{"range_bias_m", settings.rangeBiasM},
                                   {"range_scale", settings.rangeScale},
                                   {"miss_every_n", settings.missEveryN}};

    const nlohmann::json document = raylidar::eval::BuildMetricsJson(
        report, inputs, protocol, raylidar::UtcNowIso8601());
    raylidar::eval::WriteMetricsJson(metricsPath, document);
    raylidar::eval::WriteMetricsCsv(csvPath, report, protocol);

    // Validate every artifact and every recorded hash before exit 0.
    raylidar::io::GtRaysInfo rayInfo;
    raylidar::io::HitsInfo hitInfo;
    const auto readRays = raylidar::io::ReadGtRays(raysPath, &rayInfo);
    const auto readHits = raylidar::io::ReadHits(hitsPath, &hitInfo);
    raylidar::io::ValidateHitsAgainstRays(readRays, readHits);
    static_cast<void>(raylidar::io::ValidateMeshFile(scenePath));
    raylidar::eval::ValidateMetricsJson(document);
    raylidar::eval::ValidateMetricsCsv(csvPath, document);

    const std::array<std::pair<std::string, std::string>, 5> recorded = {
        std::make_pair(rayInfo.sha256, raylidar::Sha256File(raysPath)),
        std::make_pair(hitInfo.sha256, raylidar::Sha256File(hitsPath)),
        std::make_pair(meshSha, raylidar::Sha256File(scenePath)),
        std::make_pair(configSha, raylidar::Sha256File(configPath)),
        std::make_pair(protocolSha, raylidar::Sha256File(protocolDoc))};
    for (const auto &[recordedSha, actualSha] : recorded)
    {
        if (recordedSha != actualSha)
        {
            Fail("recorded hash does not match the emitted file", kExitProtocolDrift);
        }
    }
    if (rayInfo.count != artifacts.rays.size() || hitInfo.count != artifacts.hits.size())
    {
        Fail("emitted record count disagrees with the in-memory artifact",
             kExitProtocolDrift);
    }

    std::cout << "eval placeholder: wrote " << outDir.string() << " ("
              << artifacts.rays.size() << " rays, " << report.rows.size()
              << " rows)\n";
    return 0;
}

int ExecuteValidate(const ParsedArgs &parsed)
{
    const std::string metricsText = parsed.Get("metrics");
    const std::string csvText = parsed.Get("csv");
    if (metricsText.empty())
    {
        throw UsageError("eval validate requires --metrics <path>");
    }
    if (csvText.empty())
    {
        throw UsageError("eval validate requires --csv <path>");
    }
    const nlohmann::json document = raylidar::ReadJson(metricsText);
    raylidar::eval::ValidateMetricsJson(document);
    raylidar::eval::ValidateMetricsCsv(csvText, document);
    for (const char *key : {"protocol", "sim-io"})
    {
        const std::string sidecar = parsed.Get(key);
        if (!sidecar.empty())
        {
            VerifySidecar(sidecar);
        }
    }
    std::cout << "eval validate: OK\n";
    return 0;
}

int Execute(const std::span<const std::string_view> args)
{
    const ParsedArgs parsed = ParseArgs(
        args,
        {"config", "out-dir", "seed", "frames", "metrics", "csv", "protocol", "sim-io",
         "protocol-doc"},
        {"help"});
    if (parsed.Flag("help"))
    {
        PrintHelp(std::cout);
        return 0;
    }
    const std::vector<std::string> modes = parsed.GetAll("");
    if (modes.size() != 1U)
    {
        throw UsageError("expected exactly one mode: placeholder|validate");
    }
    if (modes.front() == "placeholder")
    {
        return ExecutePlaceholder(parsed);
    }
    if (modes.front() == "validate")
    {
        return ExecuteValidate(parsed);
    }
    throw UsageError("unknown eval mode '" + modes.front() +
                     "' (expected placeholder|validate)");
}
} // namespace

int RunEval(std::span<const std::string_view> args)
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

RAYLIDAR_REGISTER_COMMAND("eval", "Run or validate the Phase-3 metrics harness", RunEval);
