#include <raylidar/segment/select.h>

#include <raylidar/common/io.h>
#include <raylidar/common/mask_bits.h>
#include <raylidar/lidar/uncurl.h>
#include <raylidar/lidar/velodyne.h>
#include <raylidar/masking/semantics.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace raylidar::segment
{
namespace
{
namespace fs = std::filesystem;

constexpr int kExitInvalidInput = 4;

[[nodiscard]] std::string FrameName(const int frame)
{
    char buffer[16] = {};
    std::snprintf(buffer, sizeof(buffer), "%010d", frame);
    return std::string(buffer);
}

[[nodiscard]] int ConfigInt(const nlohmann::json &config, const char *key, const int fallback)
{
    if (!config.is_object() || !config.contains("selection"))
    {
        return fallback;
    }
    const nlohmann::json &selection = config.at("selection");
    if (!selection.is_object() || !selection.contains(key) ||
        !selection.at(key).is_number_integer())
    {
        return fallback;
    }
    return selection.at(key).get<int>();
}

[[nodiscard]] std::string ConfigString(const nlohmann::json &config, const char *key,
                                       const std::string &fallback)
{
    if (!config.is_object() || !config.contains("selection"))
    {
        return fallback;
    }
    const nlohmann::json &selection = config.at("selection");
    if (!selection.is_object() || !selection.contains(key) || !selection.at(key).is_string())
    {
        return fallback;
    }
    return selection.at(key).get<std::string>();
}

[[nodiscard]] nlohmann::json WindowJson(const WindowScore &window)
{
    nlohmann::json entry;
    entry["start_frame"] = window.start_frame;
    entry["end_frame"] = window.end_frame;
    entry["frame_count"] = window.frame_count;
    entry["dynamic_points"] = window.dynamic_points;
    entry["static_points"] = window.static_points;
    entry["score"] = window.score;
    return entry;
}

[[nodiscard]] nlohmann::json FrameJson(const FrameScore &frame)
{
    nlohmann::json entry;
    entry["frame"] = frame.frame;
    entry["dynamic_2d"] = frame.dynamic_2d;
    entry["dynamic_bbox"] = frame.dynamic_bbox;
    entry["dynamic_union"] = frame.dynamic_union;
    entry["ego_roi"] = frame.ego_roi;
    entry["kept_static"] = frame.kept_static;
    entry["nonfinite"] = frame.nonfinite;
    entry["score"] = frame.score;
    entry["usable"] = frame.usable;
    entry["unusable_reason"] = frame.unusable_reason;
    return entry;
}
} // namespace

FrameScore ScoreFrame(const int frame, const fs::path &datasetRoot, const fs::path &lidarRoot,
                      const calib::Kitti360Calib &calib,
                      const std::map<int, Eigen::Matrix4d> &poses,
                      const std::vector<masking::DynamicBBox> &boxes,
                      const SegmentInputs &inputs)
{
    if (inputs.sequence.empty())
    {
        throw IoError("segment scoring needs a sequence", kExitInvalidInput);
    }
    if (inputs.cameras.empty())
    {
        throw IoError("segment scoring needs at least one camera", kExitInvalidInput);
    }

    std::vector<std::int64_t> loadedTimestamps;
    const std::vector<std::int64_t> *timestamps = inputs.timestamps;
    if (timestamps == nullptr)
    {
        loadedTimestamps = lidar::LoadVelodyneTimestamps(lidarRoot / "timestamps.txt");
        timestamps = &loadedTimestamps;
    }

    const auto currentPose = poses.find(frame);
    const auto nextPose = poses.find(frame + 1);
    if (currentPose == poses.end() || nextPose == poses.end())
    {
        throw IoError("frame " + std::to_string(frame) +
                          " needs exact poses for itself and its successor (uncurl)",
                      kExitInvalidInput);
    }
    if (static_cast<std::size_t>(frame + 1) >= timestamps->size() ||
        static_cast<std::size_t>(frame) >= timestamps->size())
    {
        throw IoError("frame " + std::to_string(frame) +
                          " needs timestamps for itself and its successor (uncurl)",
                      kExitInvalidInput);
    }

    const std::vector<lidar::PointXYZI> scan =
        lidar::LoadVelodyneScan(lidarRoot / "data" / (FrameName(frame) + ".bin"));

    lidar::UncurlParams params;
    params.T_W_V_curr = calib::ComposeTWV(calib, currentPose->second);
    params.T_W_V_next = calib::ComposeTWV(calib, nextPose->second);
    params.t_curr_ns = (*timestamps)[static_cast<std::size_t>(frame)];
    params.t_next_ns = (*timestamps)[static_cast<std::size_t>(frame + 1)];

    std::size_t rejected = 0;
    const std::vector<Eigen::Vector3d> finite = lidar::UncurlScan(scan, params, &rejected);
    const Eigen::Vector3d placeholder(std::numeric_limits<double>::quiet_NaN(),
                                      std::numeric_limits<double>::quiet_NaN(),
                                      std::numeric_limits<double>::quiet_NaN());
    std::vector<Eigen::Vector3d> uncurled(scan.size(), placeholder);
    std::size_t cursor = 0;
    for (std::size_t index = 0; index < scan.size(); ++index)
    {
        const Eigen::Vector3d raw(scan[index].x, scan[index].y, scan[index].z);
        if (raw.allFinite())
        {
            if (cursor >= finite.size())
            {
                throw IoError("uncurl output is shorter than the finite input", kExitInvalidInput);
            }
            uncurled[index] = finite[cursor++];
        }
    }
    if (cursor != finite.size())
    {
        throw IoError("uncurl output does not match the scan", kExitInvalidInput);
    }

    std::map<std::string, masking::SemanticImage> semantics;
    std::map<std::string, std::pair<int, int>> imageSizes;
    std::map<std::string, Eigen::Matrix4d> cameraTransforms;
    for (const std::string &camera : inputs.cameras)
    {
        const fs::path semanticsPath = datasetRoot / "data_2d_semantics" / "train" /
                                       inputs.sequence / camera / "semantic" /
                                       (FrameName(frame) + ".png");
        masking::SemanticImage image = masking::LoadSemantics(semanticsPath);
        imageSizes[camera] = {image.width, image.height};
        semantics.emplace(camera, std::move(image));
        cameraTransforms[camera] = calib::ComposeTCkRectV(calib, currentPose->second, camera);
    }

    const masking::MaskResult mask =
        masking::BuildMask(scan, uncurled, frame, calib, params.T_W_V_curr, boxes, semantics,
                           imageSizes, cameraTransforms, inputs.mask);

    FrameScore score;
    score.frame = frame;
    for (const std::uint8_t bits : mask.bits)
    {
        if ((bits & static_cast<std::uint8_t>(masking::kInvalidNonFinite)) != 0U)
        {
            ++score.nonfinite;
            continue;
        }
        const bool ego = (bits & static_cast<std::uint8_t>(masking::kEgoRoi)) != 0U;
        if (ego)
        {
            ++score.ego_roi;
        }
        if ((bits & static_cast<std::uint8_t>(masking::kDynamic2D)) != 0U)
        {
            ++score.dynamic_2d;
        }
        if ((bits & static_cast<std::uint8_t>(masking::kDynamicBBox)) != 0U)
        {
            ++score.dynamic_bbox;
        }
        if (!ego && (bits & static_cast<std::uint8_t>(masking::kDynamicUnion)) != 0U)
        {
            ++score.dynamic_union;
        }
        if ((bits & static_cast<std::uint8_t>(masking::kKeptStatic)) != 0U)
        {
            ++score.kept_static;
        }
    }

    const std::uint64_t scenePoints = score.dynamic_union + score.kept_static;
    score.score = scenePoints == 0U ? 0.0
                                    : static_cast<double>(score.dynamic_union) /
                                          static_cast<double>(scenePoints);
    if (scenePoints < static_cast<std::uint64_t>(inputs.minScenePointsPerFrame))
    {
        score.usable = false;
        score.unusable_reason = "too_few_scene_points";
    }
    return score;
}

std::vector<WindowScore> CandidateWindows(const std::vector<FrameScore> &scores,
                                          const int windowFrames, const int stepFrames)
{
    std::vector<WindowScore> windows;
    if (windowFrames <= 0 || stepFrames <= 0 ||
        scores.size() < static_cast<std::size_t>(windowFrames))
    {
        return windows;
    }

    const std::size_t width = static_cast<std::size_t>(windowFrames);
    const std::size_t step = static_cast<std::size_t>(stepFrames);
    for (std::size_t start = 0; start + width <= scores.size(); start += step)
    {
        WindowScore window;
        window.start_frame = scores[start].frame;
        window.end_frame = scores[start + width - 1U].frame;
        window.frame_count = width;
        bool complete = true;
        for (std::size_t offset = 0; offset < width; ++offset)
        {
            const FrameScore &frame = scores[start + offset];
            if (!frame.usable)
            {
                complete = false;
                break;
            }
            window.dynamic_points += frame.dynamic_union;
            window.static_points += frame.kept_static;
        }
        if (!complete)
        {
            continue;
        }
        const std::uint64_t denominator = window.dynamic_points + window.static_points;
        window.score = denominator == 0U ? 0.0
                                         : static_cast<double>(window.dynamic_points) /
                                               static_cast<double>(denominator);
        windows.push_back(std::move(window));
    }
    return windows;
}

std::optional<WindowScore> SelectWindow(const std::vector<WindowScore> &windows)
{
    if (windows.empty())
    {
        return std::nullopt;
    }
    const WindowScore *best = &windows.front();
    for (const WindowScore &window : windows)
    {
        if (window.score < best->score ||
            (window.score == best->score && window.start_frame < best->start_frame))
        {
            best = &window;
        }
    }
    return *best;
}

nlohmann::json SelectionManifest(const std::vector<FrameScore> &frames,
                                 const std::vector<WindowScore> &windows,
                                 const std::optional<WindowScore> &selected,
                                 const std::string_view sequence, const std::string_view mode,
                                 const nlohmann::json &config)
{
    nlohmann::json document;
    document["schema_version"] = 1;
    document["sequence"] = std::string(sequence);
    document["generated_at_utc"] = UtcNowIso8601();
    document["selection_mode"] = std::string(mode);

    nlohmann::json criteria;
    criteria["window_frames"] = ConfigInt(config, "window_frames", 60);
    criteria["step_frames"] = ConfigInt(config, "step_frames", 10);
    criteria["score"] = ConfigString(config, "score", "micro_dynamic_fraction_after_ego");
    criteria["tie_break"] = ConfigString(config, "tie_break", "earliest_window");
    criteria["dynamic_sources"] = nlohmann::json::array({"3d_bboxes", "2d_semantics"});
    document["criteria"] = std::move(criteria);

    nlohmann::json candidateWindows = nlohmann::json::array();
    for (const WindowScore &window : windows)
    {
        candidateWindows.push_back(WindowJson(window));
    }
    document["candidate_windows"] = std::move(candidateWindows);

    if (selected.has_value())
    {
        const std::size_t width = selected->frame_count;
        std::size_t startIndex = frames.size();
        for (std::size_t index = 0; index < frames.size(); ++index)
        {
            if (frames[index].frame == selected->start_frame)
            {
                startIndex = index;
                break;
            }
        }
        if (startIndex == frames.size() || startIndex + width > frames.size())
        {
            throw std::invalid_argument(
                "selected window does not match the per-frame score list");
        }
        if (frames[startIndex + width - 1U].frame != selected->end_frame)
        {
            throw std::invalid_argument(
                "selected end_frame does not match the per-frame score slice");
        }

        nlohmann::json chosen;
        chosen["start_frame"] = selected->start_frame;
        chosen["end_frame"] = selected->end_frame;
        chosen["frame_count"] = width;
        nlohmann::json selectedFrames = nlohmann::json::array();
        for (std::size_t offset = 0; offset < width; ++offset)
        {
            const int id = frames[startIndex + offset].frame;
            if (!selectedFrames.empty() && id <= selectedFrames.back().get<int>())
            {
                throw std::invalid_argument("selected.frames must be ascending");
            }
            selectedFrames.push_back(id);
        }
        chosen["frames"] = std::move(selectedFrames);
        chosen["dynamic_points"] = selected->dynamic_points;
        chosen["static_points"] = selected->static_points;
        chosen["score"] = selected->score;
        chosen["rationale"] =
            "lowest micro dynamic fraction after ego removal; ties select the earliest window";
        document["selected"] = std::move(chosen);
    }
    else
    {
        document["selected"] = nullptr;
    }

    nlohmann::json perFrame = nlohmann::json::array();
    nlohmann::json failures = nlohmann::json::array();
    for (const FrameScore &frame : frames)
    {
        perFrame.push_back(FrameJson(frame));
        if (!frame.usable)
        {
            nlohmann::json failure;
            failure["frame"] = frame.frame;
            failure["reason"] = frame.unusable_reason;
            failures.push_back(std::move(failure));
        }
    }
    document["per_frame"] = std::move(perFrame);
    document["failures"] = std::move(failures);
    return document;
}
} // namespace raylidar::segment
