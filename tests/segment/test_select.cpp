#include "fixture.h"

#include <raylidar/calib/kitti360.h>
#include <raylidar/masking/bbox.h>
#include <raylidar/segment/select.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::segment::CandidateWindows;
using raylidar::segment::FrameScore;
using raylidar::segment::ScoreFrame;
using raylidar::segment::SegmentInputs;
using raylidar::segment::SelectionManifest;
using raylidar::segment::SelectWindow;
using raylidar::segment::WindowScore;
using raylidar::test::MakeSyntheticFrameIndex;
using raylidar::test::MakeSyntheticSegmentTree;
using raylidar::test::SyntheticSegmentTree;

FrameScore MakeScore(const int frame, const std::uint64_t dynamic, const std::uint64_t kept,
                     const bool usable = true, std::string reason = {})
{
    FrameScore score;
    score.frame = frame;
    score.dynamic_union = dynamic;
    score.kept_static = kept;
    const std::uint64_t denominator = dynamic + kept;
    score.score = denominator == 0U ? 0.0
                                    : static_cast<double>(dynamic) /
                                          static_cast<double>(denominator);
    score.usable = usable;
    score.unusable_reason = std::move(reason);
    return score;
}

nlohmann::json TestConfig()
{
    nlohmann::json config;
    config["selection"] = {{"window_frames", 2},
                           {"step_frames", 1},
                           {"score", "micro_dynamic_fraction_after_ego"},
                           {"tie_break", "earliest_window"}};
    return config;
}
} // namespace

TEST_CASE("window score is point-weighted, not a mean of per-frame ratios", "[segment]")
{
    const std::vector<FrameScore> scores = {
        MakeScore(0, 0U, 100U), MakeScore(1, 10U, 10U), MakeScore(2, 1U, 1U),
        MakeScore(3, 0U, 100U)};

    const std::vector<WindowScore> windows = CandidateWindows(scores, 2, 1);
    REQUIRE(windows.size() == 3U);
    CHECK(windows[0].start_frame == 0);
    CHECK(windows[0].dynamic_points == 10U);
    CHECK(windows[0].static_points == 110U);
    CHECK(windows[0].score == Catch::Approx(10.0 / 120.0));
    CHECK(windows[2].start_frame == 2);
    CHECK(windows[2].score == Catch::Approx(1.0 / 102.0));

    const std::optional<WindowScore> selected = SelectWindow(windows);
    REQUIRE(selected.has_value());
    CHECK(selected->start_frame == 2);
    CHECK(selected->end_frame == 3);
    CHECK(selected->score == Catch::Approx(1.0 / 102.0));
    // A mean of per-frame ratios would be (0.5 + 0.0) / 2 = 0.25 here.
    CHECK(selected->score < 0.02);
}

TEST_CASE("identical window scores select the earliest window", "[segment]")
{
    std::vector<FrameScore> scores;
    for (int frame = 0; frame < 6; ++frame)
    {
        scores.push_back(MakeScore(frame, 5U, 5U));
    }

    const std::vector<WindowScore> windows = CandidateWindows(scores, 2, 2);
    REQUIRE(windows.size() == 3U);
    const std::optional<WindowScore> selected = SelectWindow(windows);
    REQUIRE(selected.has_value());
    CHECK(selected->start_frame == 0);
    CHECK(selected->end_frame == 1);
}

TEST_CASE("fewer than one window of usable frames yields no candidate", "[segment]")
{
    std::vector<FrameScore> scores;
    for (int frame = 0; frame < 59; ++frame)
    {
        scores.push_back(MakeScore(frame, 1U, 10U));
    }

    CHECK(CandidateWindows(scores, 60, 10).empty());
    CHECK_FALSE(SelectWindow(CandidateWindows(scores, 60, 10)).has_value());
}

TEST_CASE("a window containing an unusable frame is not a candidate", "[segment]")
{
    const std::vector<FrameScore> scores = {
        MakeScore(100, 1U, 10U), MakeScore(101, 1U, 10U, false, "too_few_scene_points"),
        MakeScore(102, 0U, 10U), MakeScore(103, 0U, 10U)};

    const std::vector<WindowScore> windows = CandidateWindows(scores, 2, 1);
    REQUIRE(windows.size() == 1U);
    CHECK(windows[0].start_frame == 102);
    CHECK(windows[0].end_frame == 103);
}

TEST_CASE("step windows start on the usable-frame grid and skip the partial tail", "[segment]")
{
    std::vector<FrameScore> scores;
    for (int frame = 100; frame < 108; ++frame)
    {
        scores.push_back(MakeScore(frame, 1U, 10U));
    }

    const std::vector<WindowScore> windows = CandidateWindows(scores, 3, 2);
    REQUIRE(windows.size() == 3U);
    CHECK(windows[0].start_frame == 100);
    CHECK(windows[1].start_frame == 102);
    CHECK(windows[2].start_frame == 104);
    CHECK(windows[2].end_frame == 106);
    for (const WindowScore &window : windows)
    {
        CHECK(window.frame_count == 3U);
    }
}

TEST_CASE("ego points are excluded from the numerator and denominator", "[segment]")
{
    const SyntheticSegmentTree tree = MakeSyntheticSegmentTree("ego_exclusion");
    const raylidar::calib::Kitti360Calib calib =
        raylidar::calib::LoadKitti360Calib(tree.datasetRoot / "calibration");
    const std::map<int, Eigen::Matrix4d> poses =
        raylidar::calib::LoadPoses(tree.datasetRoot / "data_poses" / tree.sequence / "poses.txt");
    const std::vector<raylidar::masking::DynamicBBox> boxes =
        raylidar::masking::LoadDynamicBBoxes(tree.bboxPath);

    SegmentInputs inputs;
    inputs.sequence = tree.sequence;
    inputs.cameras = {"image_00", "image_01"};
    inputs.minScenePointsPerFrame = 0;

    const FrameScore score = ScoreFrame(0, tree.datasetRoot, tree.lidarRoot, calib, poses, boxes,
                                        inputs);
    CHECK(score.frame == 0);
    CHECK(score.ego_roi == 1U);
    CHECK(score.dynamic_union == 1U);
    CHECK(score.kept_static == 2U);
    CHECK(score.score == Catch::Approx(1.0 / 3.0));
    CHECK(score.usable);
}

TEST_CASE("manifest records every candidate and a canonical selected.frames", "[segment]")
{
    std::vector<FrameScore> scores = {MakeScore(10, 4U, 4U), MakeScore(11, 1U, 20U),
                                      MakeScore(13, 2U, 2U), MakeScore(17, 0U, 50U),
                                      MakeScore(18, 0U, 50U), MakeScore(21, 0U, 50U)};

    const std::vector<WindowScore> windows = CandidateWindows(scores, 2, 1);
    REQUIRE(windows.size() == 5U);
    const std::optional<WindowScore> selected = SelectWindow(windows);
    REQUIRE(selected.has_value());
    CHECK(selected->start_frame == 17);

    const nlohmann::json manifest =
        SelectionManifest(scores, windows, selected, "seq", "full", TestConfig());
    CHECK(manifest["selection_mode"] == "full");
    CHECK(manifest["candidate_windows"].size() == windows.size());
    REQUIRE(manifest["selected"].is_object());
    const nlohmann::json &chosen = manifest["selected"];
    CHECK(chosen["rationale"] ==
          "lowest micro dynamic fraction after ego removal; ties select the earliest window");
    CHECK(chosen["frame_count"].get<std::size_t>() == 2U);
    REQUIRE(chosen["frames"].size() == 2U);
    CHECK(chosen["frames"][0].get<int>() == 17);
    CHECK(chosen["frames"][1].get<int>() == 18);
    CHECK(chosen["frames"].size() == chosen["frame_count"].get<std::size_t>());

    const nlohmann::json &perFrame = manifest["per_frame"];
    std::vector<int> sliced;
    for (const nlohmann::json &entry : perFrame)
    {
        const int frame = entry["frame"].get<int>();
        if (frame >= chosen["start_frame"].get<int>() && frame <= chosen["end_frame"].get<int>())
        {
            sliced.push_back(frame);
        }
    }
    CHECK(sliced == chosen["frames"].get<std::vector<int>>());
    for (std::size_t index = 1U; index < sliced.size(); ++index)
    {
        CHECK(sliced[index - 1U] < sliced[index]);
    }
}
