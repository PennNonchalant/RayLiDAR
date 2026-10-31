#include "fixture.h"

#include <raylidar/common/io.h>
#include <raylidar/gt/cache.h>
#include <raylidar/prereg/prereg.h>
#include <raylidar/rays/freeze.h>
#include <raylidar/rays/rays.h>
#include <raylidar/rays/waterfall.h>

#include <stb_image_write.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace raylidar::test
{
namespace
{
namespace fs = std::filesystem;

constexpr const char *kSequence = "2013_05_28_drive_0000_sync";
constexpr const char *kCameras[] = {"image_00", "image_01"};
constexpr int kImageWidth = 64;
constexpr int kImageHeight = 48;
constexpr int kFirstFrame = 4615;

std::atomic<unsigned long> gCounter{0};

std::string FrameName(const int frame)
{
    std::ostringstream out;
    out << std::setw(10) << std::setfill('0') << frame;
    return out.str();
}

void WritePng(const fs::path &path)
{
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
    const std::vector<std::uint8_t> pixels(
        static_cast<std::size_t>(kImageWidth) * static_cast<std::size_t>(kImageHeight), 0U);
    if (stbi_write_png(path.string().c_str(), kImageWidth, kImageHeight, 1, pixels.data(),
                       kImageWidth) == 0)
    {
        throw std::runtime_error("cannot write fixture PNG " + path.string());
    }
}

void WriteCalibration(const fs::path &datasetRoot)
{
    const fs::path calibration = datasetRoot / "calibration";
    WriteG2TextFile(calibration / "calib_cam_to_velo.txt",
                    "1 0 0 0 0 1 0 0 0 0 1 0\n");
    WriteG2TextFile(calibration / "calib_cam_to_pose.txt",
                    "image_00: 1 0 0 0 0 1 0 0 0 0 1 0\n"
                    "image_01: 1 0 0 0 0 1 0 0 0 0 1 0\n");
    WriteG2TextFile(calibration / "perspective.txt",
                    "P_rect_00: 100 0 32 0 0 100 24 0 0 0 1 0\n"
                    "P_rect_01: 100 0 32 0 0 100 24 0 0 0 1 0\n"
                    "R_rect_00: 1 0 0 0 1 0 0 0 0 1\n"
                    "R_rect_01: 1 0 0 0 1 0 0 0 0 1\n");
}

void WritePoses(const fs::path &datasetRoot, const std::vector<int> &frames)
{
    std::ostringstream text;
    for (const int frame : frames)
    {
        text << frame << " 1 0 0 0 0 1 0 0 0 0 1 0\n";
    }
    WriteG2TextFile(datasetRoot / "data_poses" / kSequence / "poses.txt", text.str());
}

std::vector<int> RaysForFrame(const G2FixtureOptions &options, const std::size_t index)
{
    if (!options.raysPerFrame.empty())
    {
        if (index >= options.raysPerFrame.size())
        {
            return {};
        }
        return std::vector<int>(1, options.raysPerFrame[index]);
    }
    if (options.overriddenFrameRays > 0 &&
        index == static_cast<std::size_t>(options.overriddenFrame))
    {
        return std::vector<int>(1, options.overriddenFrameRays);
    }
    return std::vector<int>(1, options.raysPerFrameValue);
}

std::vector<rays::Ray> MakeRays(const int frame, const int count, const bool image01)
{
    std::vector<rays::Ray> rays;
    rays.reserve(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index)
    {
        rays::Ray ray;
        ray.ray_id = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(frame)) << 32U) |
                     static_cast<std::uint32_t>(index);
        ray.p_L = Eigen::Vector3f(0.0F, 0.0F, 10.0F);
        ray.p_C = Eigen::Vector3f(0.0F, 0.0F, 10.0F);
        ray.dir_C = Eigen::Vector3f(0.0F, 0.0F, 1.0F);
        ray.t_gt = 10.0F;
        ray.bucket = static_cast<std::uint8_t>(rays::Bucket::B0_20);
        ray.occluded = 0U;
        ray.camera = (image01 && (index % 2 == 1)) ? 1U : 0U;
        ray.semanticId = 0U;
        rays.push_back(ray);
    }
    return rays;
}

nlohmann::json WriteRays(const SyntheticG2Tree &tree, const G2FixtureOptions &options,
                         std::vector<nlohmann::json> &entries)
{
    std::uint64_t totalRays = 0;
    std::uint64_t totalOccluded = 0;
    std::uint64_t totalUnoccluded = 0;
    std::array<std::uint64_t, 4> bucketTotals{0U, 0U, 0U, 0U};
    std::array<std::uint64_t, 2> cameraTotals{0U, 0U};

    for (std::size_t index = 0; index < tree.frames.size(); ++index)
    {
        const int frame = tree.frames[index];
        const int count = RaysForFrame(options, index).front();
        const std::vector<rays::Ray> rays = MakeRays(frame, count, options.image01);

        std::array<std::uint64_t, 4> buckets{0U, 0U, 0U, 0U};
        std::array<std::uint64_t, 2> cameras{0U, 0U};
        std::uint64_t occluded = 0;
        std::uint64_t unoccludedPrimary = 0;
        std::vector<std::uint8_t> primaryMask;
        std::vector<std::uint8_t> secondaryMask;
        std::vector<std::uint8_t> unoccludedMask;
        primaryMask.reserve(rays.size());
        secondaryMask.reserve(rays.size());
        unoccludedMask.reserve(rays.size());
        for (const rays::Ray &ray : rays)
        {
            ++buckets[static_cast<std::size_t>(ray.bucket)];
            ++cameras[static_cast<std::size_t>(ray.camera)];
            const bool primary = ray.bucket != static_cast<std::uint8_t>(rays::Bucket::B50_60);
            primaryMask.push_back(primary ? 1U : 0U);
            secondaryMask.push_back(primary ? 0U : 1U);
            unoccludedMask.push_back(primary && ray.occluded == 0U ? 1U : 0U);
            if (ray.occluded != 0U)
            {
                ++occluded;
            }
            if (primary && ray.occluded == 0U)
            {
                ++unoccludedPrimary;
            }
        }

        const std::string name = FrameName(frame);
        const fs::path rayPath = tree.phaseDir / "rays" / ("frame_" + name + ".bin");
        const fs::path primaryPath = tree.phaseDir / "rays" / ("primary_mask_" + name + ".bin");
        const fs::path secondaryPath =
            tree.phaseDir / "rays" / ("secondary_mask_" + name + ".bin");
        const fs::path unoccludedPath =
            tree.phaseDir / "rays" / ("primary_unoccluded_mask_" + name + ".bin");
        rays::WriteRayFrame(rayPath, frame, rays);
        rays::WriteMaskFile(primaryPath, primaryMask);
        rays::WriteMaskFile(secondaryPath, secondaryMask);
        rays::WriteMaskFile(unoccludedPath, unoccludedMask);

        nlohmann::json entry;
        entry["frame"] = frame;
        entry["n_rays"] = rays.size();
        entry["bucket_counts"] = {{"0_20", buckets[0]},
                                  {"20_40", buckets[1]},
                                  {"40_50", buckets[2]},
                                  {"50_60", buckets[3]}};
        entry["camera_counts"] = {{"image_00", cameras[0]}, {"image_01", cameras[1]}};
        entry["occluded_count"] = occluded;
        entry["occluded_excluded_count"] = unoccludedPrimary;
        entry["full_count"] = rays.size();
        entry["sha256"] = {{"ray", Sha256File(rayPath)},
                           {"primary_mask", Sha256File(primaryPath)},
                           {"secondary_mask", Sha256File(secondaryPath)},
                           {"primary_unoccluded_mask", Sha256File(unoccludedPath)}};
        entry["files"] = {{"ray", rayPath.filename().string()},
                          {"primary_mask", primaryPath.filename().string()},
                          {"secondary_mask", secondaryPath.filename().string()},
                          {"primary_unoccluded_mask", unoccludedPath.filename().string()}};

        WriteJson(tree.phaseDir / "rays" / ("frame_" + name + "_meta.json"), entry);
        entries.push_back(entry);

        totalRays += rays.size();
        totalOccluded += occluded;
        totalUnoccluded += unoccludedPrimary;
        for (std::size_t bucket = 0; bucket < bucketTotals.size(); ++bucket)
        {
            bucketTotals[bucket] += buckets[bucket];
        }
        cameraTotals[0] += cameras[0];
        cameraTotals[1] += cameras[1];
    }

    nlohmann::json totals;
    totals["frames"] = entries.size();
    totals["n_rays"] = totalRays;
    totals["full_count"] = totalRays;
    totals["bucket_counts"] = {{"0_20", bucketTotals[0]},
                               {"20_40", bucketTotals[1]},
                               {"40_50", bucketTotals[2]},
                               {"50_60", bucketTotals[3]}};
    totals["camera_counts"] = {{"image_00", cameraTotals[0]}, {"image_01", cameraTotals[1]}};
    totals["occluded_count"] = totalOccluded;
    totals["occluded_excluded_count"] = totalUnoccluded;
    return totals;
}

void WriteSegment(const SyntheticG2Tree &tree)
{
    nlohmann::json selected;
    selected["start_frame"] = tree.frames.front();
    selected["end_frame"] = tree.frames.back();
    selected["frame_count"] = tree.frames.size();
    selected["frames"] = tree.frames;
    selected["dynamic_points"] = 0;
    selected["static_points"] = 1;
    selected["score"] = 0.0;
    selected["rationale"] = "synthetic lowest dynamic fraction";

    nlohmann::json document;
    document["schema_version"] = 1;
    document["sequence"] = kSequence;
    document["selection_mode"] = "full";
    document["criteria"] = {{"window_frames", 60}, {"step_frames", 10}};
    document["candidate_windows"] = nlohmann::json::array(
        {{{"start_frame", tree.frames.front()}, {"end_frame", tree.frames.back()},
          {"frame_count", tree.frames.size()}}});
    document["selected"] = selected;
    document["per_frame"] = nlohmann::json::array();
    document["failures"] = nlohmann::json::array();
    WriteJson(tree.phaseDir / "segment" / "segment_manifest.json", document);
}

void WriteUncurl(const SyntheticG2Tree &tree)
{
    nlohmann::json bins = nlohmann::json::array();
    for (int index = 0; index < 4; ++index)
    {
        bins.push_back({{"azimuth_deg", -180.0 + 5.0 * index},
                        {"count", 10},
                        {"mean_signed_range_error_m", 0.0}});
    }
    nlohmann::json document;
    document["schema_version"] = 1;
    document["sequence"] = kSequence;
    document["pass"] = true;
    document["frames"] = tree.frames;
    document["mask_dir"] = (tree.phaseDir / "masks").string();
    document["masks_required"] = true;
    document["raw"] = {{"bins", bins}, {"matched_count", 40}, {"peak_to_peak_m", 0.01},
                       {"max_abs_bin_mean_m", 0.01}, {"sinusoid_amplitude_m", 0.001}};
    document["uncurled"] = {{"bins", bins}, {"matched_count", 40}, {"peak_to_peak_m", 0.009},
                            {"max_abs_bin_mean_m", 0.009}, {"sinusoid_amplitude_m", 0.001}};
    document["rawPeakToPeakM"] = 0.01;
    document["uncurledPeakToPeakM"] = 0.009;
    document["rawMaxAbsBinMeanM"] = 0.01;
    document["uncurledMaxAbsBinMeanM"] = 0.009;
    document["rawSinusoidAmplitudeM"] = 0.001;
    document["uncurledSinusoidAmplitudeM"] = 0.001;
    document["failures"] = nlohmann::json::array();
    WriteJson(tree.phaseDir / "uncurl" / "uncurl_report.json", document);
}

void WriteMasks(const SyntheticG2Tree &tree, const G2FixtureOptions &options)
{
    nlohmann::json entries = nlohmann::json::array();
    for (std::size_t index = 0; index < tree.frames.size(); ++index)
    {
        const int frame = tree.frames[index];
        const int rayCount = RaysForFrame(options, index).front();
        const fs::path frameDir = tree.phaseDir / "masks" / ("frame_" + FrameName(frame));
        std::vector<std::uint8_t> bits(static_cast<std::size_t>(rayCount), 0U);
        WriteG2TextFile(frameDir / "points.bin",
                        std::string_view(reinterpret_cast<const char *>(bits.data()),
                                         bits.size()));
        WritePng(frameDir / "image_00_dynamic.png");
        WritePng(frameDir / "image_01_dynamic.png");

        nlohmann::json counts;
        counts["nonfinite"] = 0;
        counts["ego_roi"] = 0;
        counts["dynamic_2d"] = 0;
        counts["dynamic_bbox"] = 0;
        counts["dynamic_union"] = 0;
        counts["kept_static"] = rayCount;
        counts["overlap_2d_only"] = 0;
        counts["overlap_bbox_only"] = 0;
        counts["overlap_both"] = 0;

        nlohmann::json entry;
        entry["schema_version"] = 1;
        entry["frame"] = frame;
        entry["sequence"] = kSequence;
        entry["point_count"] = rayCount;
        entry["counts"] = counts;
        entry["residual_dynamic"] = {{"either", 0},
                                     {"fraction", 0.0},
                                     {"inside_bbox", 0},
                                     {"kept_static", rayCount},
                                     {"radius_m", 1.0},
                                     {"within_radius", 0}};
        entry["applies_to"] = {"images", "gt", "projections", "evaluation"};
        entry["files"] = {{"points_sha256", Sha256File(frameDir / "points.bin")},
                          {"image_00_dynamic_sha256",
                           Sha256File(frameDir / "image_00_dynamic.png")},
                          {"image_01_dynamic_sha256",
                           Sha256File(frameDir / "image_01_dynamic.png")}};
        WriteJson(frameDir / "mask.json", entry);
        entries.push_back(entry);
    }

    nlohmann::json document;
    document["schema_version"] = 1;
    document["sequence"] = kSequence;
    document["applies_to"] = {"images", "gt", "projections", "evaluation"};
    document["frames"] = entries;
    document["totals"] = nlohmann::json::object();
    document["config_sha256"] = Sha256File(tree.configPath);
    WriteJson(tree.phaseDir / "masks" / "mask_manifest.json", document);
}

void WriteGt(const SyntheticG2Tree &tree, const G2FixtureOptions &options)
{
    nlohmann::json entries = nlohmann::json::array();
    std::uint64_t totalPoints = 0;
    for (const int frame : tree.frames)
    {
        std::vector<gt::GtPoint> points;
        points.reserve(static_cast<std::size_t>(options.gtPointsPerFrame));
        for (int index = 0; index < options.gtPointsPerFrame; ++index)
        {
            gt::GtPoint point;
            point.p_L = Eigen::Vector3f(0.0F, 0.0F, 10.0F);
            point.frame = static_cast<std::uint32_t>(frame);
            point.semanticId = 0U;
            point.camera = (index % 2 == 1) ? 1U : 0U;
            point.valid = 1U;
            point.r_gt = 10.0F;
            points.push_back(point);
        }
        const fs::path path = tree.phaseDir / "gt" / ("frame_" + FrameName(frame) + ".bin");
        gt::WriteGtFrame(path, frame, points);
        entries.push_back({{"frame", frame},
                           {"count", points.size()},
                           {"camera_00", options.gtPointsPerFrame / 2},
                           {"camera_01", options.gtPointsPerFrame / 2},
                           {"camera_unknown", 0},
                           {"sha256", Sha256File(path)}});
        totalPoints += points.size();
    }
    nlohmann::json document;
    document["schema_version"] = 1;
    document["sequence"] = kSequence;
    document["config_sha256"] = Sha256File(tree.configPath);
    document["frames"] = entries;
    document["totals"] = {{"frames", entries.size()},
                          {"points", totalPoints},
                          {"camera_00", totalPoints / 2},
                          {"camera_01", totalPoints / 2},
                          {"camera_unknown", 0}};
    WriteJson(tree.phaseDir / "gt" / "gt_manifest.json", document);
}

void WriteProtocol(const SyntheticG2Tree &tree, const G2FixtureOptions &options)
{
    std::ostringstream protocol;
    protocol << "schema_version: 1\n"
             << "phase: 2\n"
             << "status: preregistered\n"
             << "run_id: p2-60f-step10\n"
             << "sequence_policy:\n"
             << "  tie_break: earliest_window\n"
             << "frame_window:\n"
             << "  window_frames: " << options.protocolWindowFrames << "\n"
             << "  step_frames: 10\n"
             << "dynamic_scoring:\n"
             << "  sources: [3d_bboxes, 2d_semantics]\n";
    WriteG2TextFile(tree.protocolPath, protocol.str());
}

void WriteConfig(const SyntheticG2Tree &tree, const G2FixtureOptions &options)
{
    std::ostringstream config;
    config << "sequence: " << kSequence << "\n"
           << "dataset_root: " << tree.datasetRoot.string() << "\n"
           << "cameras: [image_00, image_01]\n"
           << "artifacts:\n"
           << "  phase_dir: " << tree.phaseDir.string() << "\n"
           << "mask:\n"
           << "  ego_roi_velodyne_m:\n"
           << "    x: [-2.5, 2.5]\n"
           << "    y: [-1.2, 1.2]\n"
           << "    z: [-2.0, 0.3]\n"
           << "gt_cache:\n"
           << "  min_points_per_frame: 1\n"
           << "  expected_points_range: [1, 100000]\n"
           << "g2:\n"
           << "  require_image_00_and_01: true\n"
           << "  min_valid_rays_per_frame: " << options.minValidRays << "\n"
           << "  waterfall_exact: true\n";
    WriteG2TextFile(tree.configPath, config.str());
}
} // namespace

fs::path MakeG2TempRoot(const std::string_view name)
{
    std::error_code error;
    const fs::path base = fs::temp_directory_path(error) / "raylidar_g2_tests";
    fs::create_directories(base, error);
    const unsigned long counter = gCounter.fetch_add(1U) + 1U;
    const fs::path root = base / (std::string(name) + "_" + std::to_string(counter));
    fs::remove_all(root, error);
    fs::create_directories(root, error);
    return root;
}

void WriteG2TextFile(const fs::path &path, const std::string_view text)
{
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output.good())
    {
        throw std::runtime_error("cannot write " + path.string());
    }
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
}

void RebuildG2Freeze(const SyntheticG2Tree &tree)
{
    rays::FreezeRaySet(tree.raysManifestPath, tree.configPath, tree.freezePath, "test");
}

SyntheticG2Tree MakeSyntheticG2Tree(const std::string_view name,
                                    const G2FixtureOptions &options)
{
    SyntheticG2Tree tree;
    tree.root = MakeG2TempRoot(name);
    tree.datasetRoot = tree.root / "dataset";
    tree.phaseDir = tree.root / "artifacts" / "02";
    tree.phase1Dir = tree.root / "artifacts" / "01";
    tree.configsDir = tree.root / "configs";
    tree.configPath = tree.configsDir / "phase2.yaml";
    tree.protocolPath = tree.configsDir / "protocol.yaml";
    tree.decisionsPath = tree.root / "docs" / "decisions.md";
    tree.sidecarPath = tree.phase1Dir / "decisions.sha256";
    tree.raysManifestPath = tree.phaseDir / "rays" / "rays_manifest.json";
    tree.freezePath = tree.phaseDir / "rays" / "RAY_SET_FREEZE.json";
    tree.sequence = kSequence;

    for (int index = 0; index < options.frameCount; ++index)
    {
        tree.frames.push_back(kFirstFrame + index);
    }

    WriteCalibration(tree.datasetRoot);
    WritePoses(tree.datasetRoot, tree.frames);
    WriteProtocol(tree, options);
    WriteConfig(tree, options);

    WriteG2TextFile(tree.decisionsPath,
                    "# Synthetic Decisions\n\n**Frozen at (UTC):** 2026-10-31T00:00:00Z\n\n"
                    "Synthetic preregistration for G2 tests.\n");
    if (!options.omitPrereg)
    {
        prereg::FreezeDecisions(tree.decisionsPath, tree.protocolPath, tree.sidecarPath,
                                "2026-10-31T00:00:00Z");
    }

    WriteSegment(tree);
    WriteUncurl(tree);
    WriteMasks(tree, options);
    WriteGt(tree, options);

    std::vector<nlohmann::json> entries;
    const nlohmann::json totals = WriteRays(tree, options, entries);
    nlohmann::json manifest;
    manifest["schema_version"] = 1;
    manifest["sequence"] = kSequence;
    manifest["window"] = tree.frames;
    manifest["frames"] = entries;
    manifest["totals"] = totals;
    manifest["config_sha256"] = Sha256File(tree.configPath);
    WriteJson(tree.raysManifestPath, manifest);

    // The waterfall includes the camera-255 drops, but this synthetic tree has
    // no such points, so every ray is valid and every exclusion is zero.
    std::vector<rays::FrameWaterfall> waterfalls;
    waterfalls.reserve(entries.size());
    for (std::size_t index = 0; index < entries.size(); ++index)
    {
        rays::FrameWaterfall waterfall;
        waterfall.frame = tree.frames[index];
        waterfall.raw = entries[index]["n_rays"].get<std::int64_t>();
        waterfall.nonfinite = 0;
        waterfall.ego_roi = 0;
        waterfall.dynamic_union = 0;
        waterfall.no_camera_fov = 0;
        waterfall.range_out_of_domain = 0;
        waterfall.valid = waterfall.raw;
        waterfalls.push_back(waterfall);
    }
    nlohmann::json document = rays::AggregateWaterfalls(waterfalls);
    document["sequence"] = kSequence;
    document["config_sha256"] = Sha256File(tree.configPath);
    document["window"] = tree.frames;
    WriteJson(tree.phaseDir / "rays" / "waterfalls.json", document);

    rays::FreezeRaySet(tree.raysManifestPath, tree.configPath, tree.freezePath, "test");
    return tree;
}
} // namespace raylidar::test
