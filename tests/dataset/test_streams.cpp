#include "fixture.h"

#include <raylidar/cli/commands_phase2.h>
#include <raylidar/common/io.h>
#include <raylidar/dataset/streams.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::dataset::BuildStreamIndex;
using raylidar::dataset::FrameRecord;
using raylidar::dataset::StreamIndex;
using raylidar::dataset::StreamIndexToJson;
using raylidar::dataset::WriteStreamIndexCsv;

constexpr const char *kSequence = "2013_05_28_drive_0000_sync";
const std::vector<std::string> kCameras = {"image_00", "image_01"};

struct Tree
{
    fs::path root;
    fs::path lidar;
};

void WriteText(const fs::path &path, const std::string &text)
{
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    REQUIRE(output.good());
    output << text;
}

void Touch(const fs::path &path)
{
    WriteText(path, "x");
}

void WriteBin(const fs::path &path)
{
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    REQUIRE(output.good());
    const float point[4] = {1.0F, 2.0F, 3.0F, 0.5F};
    output.write(reinterpret_cast<const char *>(point), sizeof(point));
}

std::string FrameName(int frame)
{
    char buffer[16] = {};
    std::snprintf(buffer, sizeof(buffer), "%010d", frame);
    return std::string(buffer);
}

Tree MakeTree(std::string_view name)
{
    Tree tree;
    tree.root = raylidar::test::MakeTempRoot(name);
    tree.lidar = tree.root / "lidar";
    return tree;
}

fs::path ImagePath(const Tree &tree, std::string_view camera, int frame)
{
    return tree.root / "data_2d_raw" / kSequence / std::string(camera) / "data_rect" / (FrameName(frame) + ".png");
}

fs::path SemanticsPath(const Tree &tree, std::string_view camera, int frame)
{
    return tree.root / "data_2d_semantics" / "train" / kSequence / std::string(camera) / "semantic" /
           (FrameName(frame) + ".png");
}

void WritePoses(const Tree &tree, const std::vector<int> &frames)
{
    std::string poses;
    std::string cam0;
    for (const int frame : frames)
    {
        poses += std::to_string(frame) + " 1 0 0 0 0 1 0 0 0 0 1 0\n";
        cam0 += std::to_string(frame) + " 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n";
    }
    WriteText(tree.root / "data_poses" / kSequence / "poses.txt", poses);
    WriteText(tree.root / "data_poses" / kSequence / "cam0_to_world.txt", cam0);
}

void WriteTimestamps(const Tree &tree, const std::vector<std::string> &lines)
{
    std::string text;
    for (const std::string &line : lines)
    {
        text += line;
        text += '\n';
    }
    WriteText(tree.lidar / "timestamps.txt", text);
}

void WriteFrameStreams(const Tree &tree, int frame)
{
    WriteBin(tree.lidar / "data" / (FrameName(frame) + ".bin"));
    for (const std::string &camera : kCameras)
    {
        Touch(ImagePath(tree, camera, frame));
        Touch(SemanticsPath(tree, camera, frame));
    }
}

StreamIndex Build(const Tree &tree)
{
    return BuildStreamIndex(tree.root, tree.lidar, kSequence, kCameras, true, true, true);
}

const FrameRecord *FindRecord(const std::vector<FrameRecord> &records, int frame)
{
    const auto found = std::find_if(records.begin(), records.end(),
                                    [frame](const FrameRecord &record) { return record.frame == frame; });
    return found == records.end() ? nullptr : &*found;
}

std::vector<int> Frames(const std::vector<FrameRecord> &records)
{
    std::vector<int> frames;
    frames.reserve(records.size());
    for (const FrameRecord &record : records)
    {
        frames.push_back(record.frame);
    }
    return frames;
}
} // namespace

TEST_CASE("A frame present in every stream is usable and a missing stream is a drop reason", "[streams]")
{
    Tree tree = MakeTree("streams_all_present");
    WriteFrameStreams(tree, 0);
    WriteFrameStreams(tree, 1);
    WriteTimestamps(tree, {"t0", "t1", "t2"});
    WritePoses(tree, {0, 1, 2});

    StreamIndex index = Build(tree);
    REQUIRE(Frames(index.usable) == std::vector<int>{0, 1});
    REQUIRE(index.dropped.empty());

    std::error_code error;
    fs::remove(ImagePath(tree, "image_01", 1), error);
    index = Build(tree);
    REQUIRE(Frames(index.usable) == std::vector<int>{0});
    REQUIRE(Frames(index.dropped) == std::vector<int>{1});
    REQUIRE(index.dropped.front().missing == std::vector<std::string>{"image_01"});
}

TEST_CASE("A pose within the nearest-frame delta but not exact is dropped", "[streams]")
{
    Tree tree = MakeTree("streams_inexact_pose");
    WriteFrameStreams(tree, 0);
    WriteFrameStreams(tree, 1);
    WriteTimestamps(tree, {"t0", "t1", "t2"});
    WritePoses(tree, {1, 2});

    const StreamIndex index = Build(tree);
    REQUIRE(Frames(index.usable) == std::vector<int>{1});
    const FrameRecord *dropped = FindRecord(index.dropped, 0);
    REQUIRE(dropped != nullptr);
    REQUIRE(dropped->missing == std::vector<std::string>{"pose", "cam0_to_world"});
}

TEST_CASE("A frame without an exact successor pose is dropped", "[streams]")
{
    Tree tree = MakeTree("streams_missing_next_pose");
    WriteFrameStreams(tree, 0);
    WriteFrameStreams(tree, 1);
    WriteTimestamps(tree, {"t0", "t1", "t2"});
    WritePoses(tree, {0});

    const StreamIndex index = Build(tree);
    REQUIRE(index.usable.empty());
    const FrameRecord *dropped = FindRecord(index.dropped, 0);
    REQUIRE(dropped != nullptr);
    REQUIRE(dropped->missing == std::vector<std::string>{"next_pose"});
    REQUIRE_FALSE(dropped->present.next_pose);
}

TEST_CASE("A frame without a non-empty successor timestamp is dropped", "[streams]")
{
    Tree tree = MakeTree("streams_missing_next_timestamp");
    WriteFrameStreams(tree, 0);
    WriteFrameStreams(tree, 1);
    WriteTimestamps(tree, {"t0", ""});
    WritePoses(tree, {0, 1, 2});

    const StreamIndex index = Build(tree);
    const FrameRecord *dropped = FindRecord(index.dropped, 0);
    REQUIRE(dropped != nullptr);
    REQUIRE(dropped->present.pose);
    REQUIRE(dropped->present.next_pose);
    REQUIRE(dropped->missing == std::vector<std::string>{"next_timestamp"});
}

TEST_CASE("Missing semantics is a drop reason even when every other stream exists", "[streams]")
{
    Tree tree = MakeTree("streams_missing_semantics");
    WriteFrameStreams(tree, 0);
    WriteFrameStreams(tree, 1);
    std::error_code error;
    fs::remove(SemanticsPath(tree, "image_01", 1), error);
    WriteTimestamps(tree, {"t0", "t1", "t2"});
    WritePoses(tree, {0, 1, 2});

    const StreamIndex index = Build(tree);
    REQUIRE(Frames(index.usable) == std::vector<int>{0});
    const FrameRecord *dropped = FindRecord(index.dropped, 1);
    REQUIRE(dropped != nullptr);
    REQUIRE(dropped->missing == std::vector<std::string>{"semantics_01"});
}

TEST_CASE("Frames are sorted and missing reasons are ordered and de-duplicated", "[streams]")
{
    Tree tree = MakeTree("streams_ordering");
    WriteFrameStreams(tree, 2);
    WriteFrameStreams(tree, 0);
    WriteTimestamps(tree, {"t0", "t1", "t2"});
    WritePoses(tree, {});

    const StreamIndex index = Build(tree);
    REQUIRE(Frames(index.dropped) == std::vector<int>{0, 2});

    const FrameRecord *frame0 = FindRecord(index.dropped, 0);
    REQUIRE(frame0 != nullptr);
    REQUIRE(frame0->missing == std::vector<std::string>{"pose", "next_pose", "cam0_to_world"});
    REQUIRE(std::adjacent_find(frame0->missing.begin(), frame0->missing.end()) == frame0->missing.end());

    const FrameRecord *frame2 = FindRecord(index.dropped, 2);
    REQUIRE(frame2 != nullptr);
    REQUIRE(frame2->missing == std::vector<std::string>{"next_timestamp", "pose", "next_pose", "cam0_to_world"});
}

TEST_CASE("JSON counts and CSV columns match the frozen schema", "[streams]")
{
    Tree tree = MakeTree("streams_outputs");
    WriteFrameStreams(tree, 0);
    WriteFrameStreams(tree, 1);
    WriteTimestamps(tree, {"t0", "t1", "t2"});
    WritePoses(tree, {0, 1, 2});

    const StreamIndex index = Build(tree);
    const nlohmann::json document = StreamIndexToJson(index, kSequence, tree.root, tree.lidar);
    REQUIRE(document.at("schema_version").get<int>() == 1);
    REQUIRE(document.at("counts").at("total").get<int>() == 2);
    REQUIRE(document.at("counts").at("usable").get<int>() == 2);
    REQUIRE(document.at("counts").at("dropped").get<int>() == 0);
    REQUIRE(document.at("usable_frames") == nlohmann::json::array({0, 1}));

    const fs::path csv = tree.root / "frame_index.csv";
    WriteStreamIndexCsv(csv, index);
    std::ifstream input(csv);
    REQUIRE(input.good());
    std::string header;
    std::getline(input, header);
    REQUIRE(header == "frame,image_00,image_01,velodyne,timestamp,next_timestamp,pose,next_pose,"
                      "cam0_to_world,semantics_00,semantics_01");
}

namespace
{
void WriteCliConfig(const fs::path &path, const Tree &tree)
{
    const std::string text = "schema_version: 1\n"
                             "phase: 2\n"
                             "sequence: " +
                             std::string(kSequence) +
                             "\n"
                             "dataset_root: " +
                             tree.root.string() +
                             "\n"
                             "lidar_root: " +
                             tree.lidar.string() +
                             "\n"
                             "cameras: [image_00, image_01]\n"
                             "streams:\n"
                             "  require_exact_pose: true\n"
                             "  require_next_pose: true\n"
                             "  require_semantics: true\n";
    WriteText(path, text);
}

int RunFrameIndexCli(const std::vector<std::string> &owned)
{
    std::vector<std::string_view> args;
    args.reserve(owned.size());
    for (const std::string &argument : owned)
    {
        args.emplace_back(argument);
    }
    std::span<const std::string_view> view(args);
    return RunFrameIndex(view);
}
} // namespace

TEST_CASE("frame-index writes all three artifacts on a synthetic tree", "[streams][cli]")
{
    Tree tree = MakeTree("streams_cli_ok");
    WriteFrameStreams(tree, 0);
    WriteFrameStreams(tree, 1);
    WriteTimestamps(tree, {"t0", "t1", "t2"});
    WritePoses(tree, {0, 1, 2});

    const fs::path config = tree.root / "phase2.yaml";
    WriteCliConfig(config, tree);
    const fs::path outDir = tree.root / "streams";

    const std::vector<std::string> args = {"--config",     config.string(),     "--root",     tree.root.string(),
                                           "--lidar-root", tree.lidar.string(), "--sequence", kSequence,
                                           "--out-dir",    outDir.string()};
    REQUIRE(RunFrameIndexCli(args) == 0);

    REQUIRE(fs::is_regular_file(outDir / "frame_index.json"));
    REQUIRE(fs::is_regular_file(outDir / "frame_index.csv"));
    REQUIRE(fs::is_regular_file(outDir / "dropped_frames.json"));

    const nlohmann::json frameIndex = raylidar::ReadJson(outDir / "frame_index.json");
    REQUIRE(frameIndex.at("usable_frames") == nlohmann::json::array({0, 1}));
    REQUIRE(frameIndex.at("counts").at("dropped").get<int>() == 0);

    std::ifstream csv(outDir / "frame_index.csv");
    REQUIRE(csv.good());
    std::string header;
    std::getline(csv, header);
    REQUIRE(header.rfind("frame,image_00,image_01,velodyne,timestamp,next_timestamp,pose,"
                         "next_pose,cam0_to_world,semantics_00,semantics_01",
                         0) == 0);

    std::error_code error;
    fs::remove(ImagePath(tree, "image_01", 1), error);
    const fs::path droppedDir = tree.root / "streams_dropped";
    const std::vector<std::string> droppedArgs = {"--config",     config.string(),     "--root",     tree.root.string(),
                                                  "--lidar-root", tree.lidar.string(), "--sequence", kSequence,
                                                  "--out-dir",    droppedDir.string()};
    REQUIRE(RunFrameIndexCli(droppedArgs) == 0);

    const nlohmann::json dropped = raylidar::ReadJson(droppedDir / "dropped_frames.json");
    REQUIRE(dropped.at("dropped").is_array());
    REQUIRE(dropped.at("dropped").size() == 1U);
    REQUIRE(dropped.at("dropped").front().at("missing").is_array());
    REQUIRE_FALSE(dropped.at("dropped").front().at("missing").empty());
}

TEST_CASE("frame-index exits 4 when the config is missing", "[streams][cli]")
{
    Tree tree = MakeTree("streams_cli_missing_config");
    const std::vector<std::string> args = {"--config",     (tree.root / "absent.yaml").string(),
                                           "--root",       tree.root.string(),
                                           "--lidar-root", tree.lidar.string(),
                                           "--sequence",   kSequence,
                                           "--out-dir",    (tree.root / "streams").string()};
    REQUIRE(RunFrameIndexCli(args) == 4);
}

TEST_CASE("frame-index rejects an unknown option with exit 2", "[streams][cli]")
{
    Tree tree = MakeTree("streams_cli_unknown_option");
    const fs::path config = tree.root / "phase2.yaml";
    WriteCliConfig(config, tree);
    const std::vector<std::string> args = {"--config", config.string(), "--bogus", "x"};
    REQUIRE(RunFrameIndexCli(args) == 2);
}
