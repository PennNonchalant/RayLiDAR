#include <raylidar/dataset/streams.h>

#include <raylidar/common/io.h>
#include <raylidar/dataset/layout.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace raylidar::dataset
{
namespace
{
namespace fs = std::filesystem;

constexpr std::size_t kMaxFrames = 1000000U;
constexpr std::uintmax_t kPointBytes = 16U;
constexpr std::size_t kBinNameLength = 14U; // 10 digits + ".bin"

/// Fixed reason order. Downstream waterfalls rely on it being stable.
constexpr std::array<std::string_view, 11> kMissingOrder = {
    "image_00",
    "image_01",
    "velodyne",
    "timestamp",
    "next_timestamp",
    "pose",
    "next_pose",
    "cam0_to_world",
    "semantics_00",
    "semantics_01",
    "velodyne_size_not_multiple_of_16",
};

std::string FrameName(int frame)
{
    char buffer[16] = {};
    std::snprintf(buffer, sizeof(buffer), "%010d", frame);
    return std::string(buffer);
}

std::string Trim(std::string value)
{
    const auto notSpace = [](unsigned char ch) { return std::isspace(ch) == 0; };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), notSpace));
    value.erase(std::find_if(value.rbegin(), value.rend(), notSpace).base(), value.end());
    return value;
}

bool IsRegularFile(const fs::path &path)
{
    std::error_code error;
    return fs::is_regular_file(path, error);
}

std::vector<std::string> ReadLines(const fs::path &path, std::string_view what)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        throw IoError("cannot read " + std::string(what) + ": " + path.string(), 4);
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        lines.push_back(std::move(line));
    }
    if (input.bad())
    {
        throw IoError("cannot read " + std::string(what) + ": " + path.string(), 4);
    }
    return lines;
}

std::set<int> ReadFrameIds(const fs::path &path, std::string_view what)
{
    const std::vector<std::string> lines = ReadLines(path, what);
    std::set<int> frames;
    for (std::size_t index = 0; index < lines.size(); ++index)
    {
        const std::string trimmed = Trim(lines[index]);
        if (trimmed.empty() || trimmed.front() == '#')
        {
            continue;
        }

        std::istringstream tokens(trimmed);
        std::string token;
        if (!(tokens >> token))
        {
            continue;
        }
        char *end = nullptr;
        errno = 0;
        const long value = std::strtol(token.c_str(), &end, 10);
        if (end == token.c_str() || *end != '\0' || errno != 0 || value < 0)
        {
            throw IoError(std::string(what) + ": invalid frame id '" + token + "' on line " +
                              std::to_string(index + 1U),
                          4);
        }
        frames.insert(static_cast<int>(value));
    }
    return frames;
}

bool IsBinName(std::string_view name)
{
    if (name.size() != kBinNameLength || name.substr(10U) != ".bin")
    {
        return false;
    }
    for (std::size_t index = 0; index < 10U; ++index)
    {
        if (name[index] < '0' || name[index] > '9')
        {
            return false;
        }
    }
    return true;
}

int BinFrame(std::string_view name)
{
    return static_cast<int>(std::strtol(std::string(name.substr(0U, 10U)).c_str(), nullptr, 10));
}

std::vector<int> EnumerateLidarFrames(const fs::path &lidarRoot)
{
    const fs::path dataDir = lidarRoot / "data";
    std::error_code error;
    if (!fs::is_directory(dataDir, error))
    {
        throw IoError("Velodyne data directory is missing or unreadable: " + dataDir.string(), 4);
    }

    std::set<int> frames;
    fs::directory_iterator iterator(dataDir, fs::directory_options::skip_permission_denied,
                                    error);
    const fs::directory_iterator end;
    if (error)
    {
        throw IoError("cannot enumerate Velodyne data directory: " + dataDir.string(), 4);
    }
    for (; iterator != end; iterator.increment(error))
    {
        if (error)
        {
            error.clear();
            continue;
        }
        const fs::directory_entry &entry = *iterator;
        std::error_code fileError;
        if (!entry.is_regular_file(fileError))
        {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (!IsBinName(name))
        {
            continue;
        }
        frames.insert(BinFrame(name));
        if (frames.size() > kMaxFrames)
        {
            throw IoError("Velodyne frame count exceeds the 1,000,000 cap", 4);
        }
    }
    return std::vector<int>(frames.begin(), frames.end());
}

bool HasTimestamp(const std::vector<std::string> &lines, int frame)
{
    if (frame < 0 || static_cast<std::size_t>(frame) >= lines.size())
    {
        return false;
    }
    return !Trim(lines[static_cast<std::size_t>(frame)]).empty();
}

bool ContainsCamera(const std::vector<std::string> &cameras, std::string_view name)
{
    return std::any_of(cameras.begin(), cameras.end(),
                       [name](const std::string &camera) { return camera == name; });
}

int MissingRank(const std::string &reason)
{
    for (std::size_t index = 0; index < kMissingOrder.size(); ++index)
    {
        if (reason == kMissingOrder[index])
        {
            return static_cast<int>(index);
        }
    }
    return static_cast<int>(kMissingOrder.size());
}

void SortMissing(std::vector<std::string> &missing)
{
    std::stable_sort(missing.begin(), missing.end(), [](const std::string &left,
                                                       const std::string &right) {
        return MissingRank(left) < MissingRank(right);
    });
    missing.erase(std::unique(missing.begin(), missing.end()), missing.end());
}

void AddMissing(FrameRecord &record, std::string reason)
{
    record.missing.push_back(std::move(reason));
}

std::vector<FrameRecord> AllRecords(const StreamIndex &index)
{
    std::vector<const FrameRecord *> pointers;
    pointers.reserve(index.usable.size() + index.dropped.size());
    for (const FrameRecord &record : index.usable)
    {
        pointers.push_back(&record);
    }
    for (const FrameRecord &record : index.dropped)
    {
        pointers.push_back(&record);
    }
    std::sort(pointers.begin(), pointers.end(),
              [](const FrameRecord *left, const FrameRecord *right)
              { return left->frame < right->frame; });

    std::vector<FrameRecord> records;
    records.reserve(pointers.size());
    for (const FrameRecord *record : pointers)
    {
        records.push_back(*record);
    }
    return records;
}
} // namespace

StreamIndex BuildStreamIndex(const std::filesystem::path &datasetRoot,
                             const std::filesystem::path &lidarRoot, std::string_view sequence,
                             const std::vector<std::string> &cameras, bool requireExactPose,
                             bool requireNextPose, bool requireSemantics)
{
    DatasetLayout layout;
    layout.root = datasetRoot;
    layout.sequence = std::string(sequence);
    layout.cameras = cameras;

    const std::vector<int> frames = EnumerateLidarFrames(lidarRoot);

    const fs::path posesDir = layout.PosesDir();
    const std::set<int> poses = ReadFrameIds(posesDir / "poses.txt", "poses.txt");
    const std::set<int> cam0ToWorld =
        ReadFrameIds(posesDir / "cam0_to_world.txt", "cam0_to_world.txt");
    const std::vector<std::string> timestamps =
        ReadLines(lidarRoot / "timestamps.txt", "timestamps.txt");

    const bool hasImage00 = ContainsCamera(cameras, "image_00");
    const bool hasImage01 = ContainsCamera(cameras, "image_01");

    StreamIndex index;
    index.cameras = cameras;
    index.usable.reserve(frames.size());
    index.dropped.reserve(frames.size());

    for (const int frame : frames)
    {
        FrameRecord record;
        record.frame = frame;
        StreamPresence &present = record.present;
        const std::string name = FrameName(frame);

        if (hasImage00)
        {
            present.image_00 =
                IsRegularFile(layout.ImageDir("image_00") / (name + ".png"));
        }
        if (!present.image_00)
        {
            AddMissing(record, "image_00");
        }

        if (hasImage01)
        {
            present.image_01 =
                IsRegularFile(layout.ImageDir("image_01") / (name + ".png"));
        }
        if (!present.image_01)
        {
            AddMissing(record, "image_01");
        }

        const fs::path binPath = lidarRoot / "data" / (name + ".bin");
        present.velodyne = IsRegularFile(binPath);
        if (!present.velodyne)
        {
            AddMissing(record, "velodyne");
        }
        else
        {
            std::error_code error;
            const std::uintmax_t size = fs::file_size(binPath, error);
            if (error)
            {
                throw IoError("cannot stat Velodyne scan " + binPath.string(), 4);
            }
            if (size == 0U || size % kPointBytes != 0U)
            {
                AddMissing(record, "velodyne_size_not_multiple_of_16");
            }
        }

        present.timestamp = HasTimestamp(timestamps, frame);
        if (!present.timestamp)
        {
            AddMissing(record, "timestamp");
        }

        present.pose = poses.find(frame) != poses.end();
        present.cam0_to_world = cam0ToWorld.find(frame) != cam0ToWorld.end();
        if (requireExactPose)
        {
            if (!present.pose)
            {
                AddMissing(record, "pose");
            }
            if (!present.cam0_to_world)
            {
                AddMissing(record, "cam0_to_world");
            }
        }

        if (requireNextPose)
        {
            present.next_pose = poses.find(frame + 1) != poses.end();
            present.next_timestamp = HasTimestamp(timestamps, frame + 1);
            if (!present.next_timestamp)
            {
                AddMissing(record, "next_timestamp");
            }
            if (!present.next_pose)
            {
                AddMissing(record, "next_pose");
            }
        }

        if (hasImage00)
        {
            present.semantics_00 =
                IsRegularFile(layout.SemanticsDir("image_00") / (name + ".png"));
        }
        if (!present.semantics_00 && requireSemantics)
        {
            AddMissing(record, "semantics_00");
        }

        if (hasImage01)
        {
            present.semantics_01 =
                IsRegularFile(layout.SemanticsDir("image_01") / (name + ".png"));
        }
        if (!present.semantics_01 && requireSemantics)
        {
            AddMissing(record, "semantics_01");
        }

        SortMissing(record.missing);
        if (record.missing.empty())
        {
            index.usable.push_back(std::move(record));
        }
        else
        {
            index.dropped.push_back(std::move(record));
        }
    }

    std::sort(index.usable.begin(), index.usable.end(),
              [](const FrameRecord &left, const FrameRecord &right)
              { return left.frame < right.frame; });
    std::sort(index.dropped.begin(), index.dropped.end(),
              [](const FrameRecord &left, const FrameRecord &right)
              { return left.frame < right.frame; });
    return index;
}

nlohmann::json StreamIndexToJson(const StreamIndex &index, std::string_view sequence,
                                 const std::filesystem::path &datasetRoot,
                                 const std::filesystem::path &lidarRoot)
{
    nlohmann::json document;
    document["schema_version"] = 1;
    document["generated_at_utc"] = UtcNowIso8601();
    document["sequence"] = std::string(sequence);
    document["dataset_root"] = datasetRoot.string();
    document["lidar_root"] = lidarRoot.string();
    document["cameras"] = index.cameras;

    nlohmann::json counts;
    counts["total"] = index.usable.size() + index.dropped.size();
    counts["usable"] = index.usable.size();
    counts["dropped"] = index.dropped.size();
    document["counts"] = std::move(counts);

    nlohmann::json usable = nlohmann::json::array();
    for (const FrameRecord &record : index.usable)
    {
        usable.push_back(record.frame);
    }
    document["usable_frames"] = std::move(usable);

    nlohmann::json dropped = nlohmann::json::array();
    for (const FrameRecord &record : index.dropped)
    {
        nlohmann::json entry;
        entry["frame"] = record.frame;
        entry["missing"] = record.missing;
        dropped.push_back(std::move(entry));
    }
    document["dropped"] = std::move(dropped);
    return document;
}

void WriteStreamIndexCsv(const std::filesystem::path &path, const StreamIndex &index)
{
    const fs::path parent = path.parent_path();
    if (!parent.empty())
    {
        EnsureDir(parent);
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
    {
        throw IoError("cannot write frame index CSV: " + path.string(), 4);
    }

    output << "frame,image_00,image_01,velodyne,timestamp,next_timestamp,pose,next_pose,"
              "cam0_to_world,semantics_00,semantics_01\n";
    for (const FrameRecord &record : AllRecords(index))
    {
        const StreamPresence &present = record.present;
        output << record.frame << ',' << static_cast<int>(present.image_00) << ','
               << static_cast<int>(present.image_01) << ','
               << static_cast<int>(present.velodyne) << ','
               << static_cast<int>(present.timestamp) << ','
               << static_cast<int>(present.next_timestamp) << ','
               << static_cast<int>(present.pose) << ',' << static_cast<int>(present.next_pose)
               << ',' << static_cast<int>(present.cam0_to_world) << ','
               << static_cast<int>(present.semantics_00) << ','
               << static_cast<int>(present.semantics_01) << '\n';
    }
    if (!output)
    {
        throw IoError("cannot write frame index CSV: " + path.string(), 4);
    }
}
} // namespace raylidar::dataset
