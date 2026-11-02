#include <raylidar/dataset/layout.h>

#include <stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
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

constexpr const char *kForbiddenCalibrationNames[] = {
    "calib_cam_to_cam.txt",
    "calib_velo_to_cam.txt",
    "calib_imu_to_velo.txt",
};

constexpr int kExpectedImageWidth = 1408;
constexpr int kExpectedImageHeight = 376;
constexpr std::size_t kExpectedPoseTokens = 13U;      // frame + 3x4
constexpr std::size_t kExpectedCam0WorldTokens = 17U; // frame + 4x4
constexpr std::uintmax_t kLidarPointBytes = 16U;      // 4 float32 per point

bool IsRegularFile(const fs::path &path)
{
    std::error_code error;
    return fs::is_regular_file(path, error);
}

bool IsDirectory(const fs::path &path)
{
    std::error_code error;
    return fs::is_directory(path, error);
}

std::uintmax_t FileSize(const fs::path &path)
{
    std::error_code error;
    const std::uintmax_t size = fs::file_size(path, error);
    return error ? 0U : size;
}

std::string RelativeTo(const fs::path &base, const fs::path &path)
{
    std::error_code error;
    const fs::path relative = fs::relative(path, base, error);
    if (error || relative.empty())
    {
        return path.lexically_normal().generic_string();
    }
    return relative.generic_string();
}

std::string JoinNames(const std::vector<std::string> &values)
{
    std::string joined;
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        if (index != 0U)
        {
            joined += ", ";
        }
        joined += values[index];
    }
    return joined;
}

bool IsForbiddenCalibrationName(std::string_view name)
{
    for (const char *forbidden : kForbiddenCalibrationNames)
    {
        if (name == forbidden)
        {
            return true;
        }
    }
    return false;
}

struct DirectoryFiles
{
    std::size_t count = 0;
    std::uintmax_t bytes = 0;
    std::string first; // lexicographically smallest matching filename
};

DirectoryFiles CollectExtension(const fs::path &directory, std::string_view extension)
{
    DirectoryFiles result;
    std::error_code error;
    fs::directory_iterator iterator(directory, fs::directory_options::skip_permission_denied, error);
    const fs::directory_iterator end;
    if (error)
    {
        return result;
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
        const fs::path path = entry.path();
        if (path.extension() != extension)
        {
            continue;
        }
        ++result.count;
        result.bytes += entry.file_size(fileError);
        const std::string name = path.filename().string();
        if (result.first.empty() || name < result.first)
        {
            result.first = name;
        }
    }
    return result;
}

struct PosesFileStats
{
    std::size_t dataLines = 0;
    bool malformed = false;
    std::string detail;
};

PosesFileStats InspectPosesFile(const fs::path &path, std::size_t expectedTokens)
{
    PosesFileStats stats;
    std::ifstream input(path);
    if (!input)
    {
        stats.malformed = true;
        stats.detail = "unreadable";
        return stats;
    }

    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(input, line))
    {
        ++lineNumber;
        const std::size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos)
        {
            continue;
        }
        if (line[first] == '#')
        {
            continue;
        }

        std::istringstream tokens(line);
        std::size_t count = 0;
        std::string token;
        while (tokens >> token)
        {
            ++count;
        }
        if (count != expectedTokens)
        {
            stats.malformed = true;
            if (stats.detail.empty())
            {
                stats.detail = "line " + std::to_string(lineNumber) + " has " + std::to_string(count) +
                               " tokens (expected " + std::to_string(expectedTokens) + ")";
            }
        }
        ++stats.dataLines;
    }

    if (stats.dataLines == 0U)
    {
        stats.malformed = true;
        if (stats.detail.empty())
        {
            stats.detail = "no data lines";
        }
    }
    return stats;
}

} // namespace

std::string_view LayoutKindName(LayoutKind kind)
{
    switch (kind)
    {
    case LayoutKind::Official:
        return "Official";
    case LayoutKind::Repackaged:
        return "Repackaged";
    case LayoutKind::Mixed:
        return "Mixed";
    case LayoutKind::Unknown:
        return "Unknown";
    }
    return "Unknown";
}

bool ComponentStatus::Ok() const noexcept
{
    return missing.empty() && rejected.empty();
}

fs::path DatasetLayout::CalibrationDir() const
{
    return root / "calibration";
}

fs::path DatasetLayout::PosesDir() const
{
    return root / "data_poses" / sequence;
}

fs::path DatasetLayout::ImageDir(std::string_view camera) const
{
    return root / "data_2d_raw" / sequence / std::string(camera) / "data_rect";
}

fs::path DatasetLayout::SemanticsDir(std::string_view camera) const
{
    return root / "data_2d_semantics" / "train" / sequence / std::string(camera) / "semantic";
}

fs::path DatasetLayout::LidarDir() const
{
    return root / "data_3d_raw" / sequence / "velodyne_points" / "data";
}

fs::path DatasetLayout::Path(std::string_view relative) const
{
    return root / std::string(relative);
}

DatasetLayout DetectLayout(const fs::path &root, std::string_view sequence)
{
    DatasetLayout layout;
    layout.root = root;
    layout.sequence = std::string(sequence);

    const fs::path sequencePath(sequence);
    const bool officialCalibration = IsRegularFile(root / "calibration" / "calib_cam_to_pose.txt");
    const bool officialPoses = IsRegularFile(root / "data_poses" / sequencePath / "poses.txt");
    const bool officialImages = IsDirectory(root / "data_2d_raw" / sequencePath);
    const bool repackaged = IsDirectory(root / "raw" / "2013_05_28" / sequencePath / "image_00");

    const int officialMarkers =
        static_cast<int>(officialCalibration) + static_cast<int>(officialPoses) + static_cast<int>(officialImages);

    if (officialMarkers == 3 && !repackaged)
    {
        layout.kind = LayoutKind::Official;
    }
    else if (repackaged && officialMarkers > 0)
    {
        layout.kind = LayoutKind::Mixed;
    }
    else if (repackaged)
    {
        layout.kind = LayoutKind::Repackaged;
    }
    else
    {
        layout.kind = LayoutKind::Unknown;
    }
    return layout;
}

std::vector<std::string> FindRejectedCalibration(const fs::path &root, int maxDepth)
{
    std::vector<std::string> rejected;
    std::error_code error;
    if (!IsDirectory(root))
    {
        return rejected;
    }

    if (maxDepth < 0)
    {
        maxDepth = 0;
    }

    std::set<std::string> seen;

    const auto scan = [&](const fs::path &scanRoot) {
        std::vector<std::pair<fs::path, int>> stack;
        stack.emplace_back(scanRoot, 0);
        while (!stack.empty())
        {
            const auto [directory, depth] = stack.back();
            stack.pop_back();

            std::error_code iterateError;
            fs::directory_iterator iterator(directory, fs::directory_options::skip_permission_denied, iterateError);
            const fs::directory_iterator end;
            if (iterateError)
            {
                continue;
            }

            for (; iterator != end; iterator.increment(iterateError))
            {
                if (iterateError)
                {
                    iterateError.clear();
                    continue;
                }
                const fs::directory_entry &entry = *iterator;
                const std::string name = entry.path().filename().string();

                std::error_code statusError;
                const fs::file_status status = entry.symlink_status(statusError);
                if (statusError)
                {
                    continue;
                }
                // Do not descend through directory symlinks: data_3d_raw and
                // timestamps are legitimate links into /data-20T/kitti.
                const bool directoryEntry = fs::is_directory(status);

                if (!directoryEntry && IsForbiddenCalibrationName(name))
                {
                    std::error_code canonicalError;
                    const fs::path canonical = fs::weakly_canonical(entry.path(), canonicalError);
                    const std::string key =
                        canonicalError ? entry.path().lexically_normal().string() : canonical.string();
                    if (seen.insert(key).second)
                    {
                        rejected.push_back(RelativeTo(root, entry.path()));
                    }
                }

                if (directoryEntry && depth + 1 < maxDepth)
                {
                    stack.emplace_back(entry.path(), depth + 1);
                }
            }
        }
    };

    scan(root);
    const fs::path rawRoot = root / "raw";
    if (IsDirectory(rawRoot))
    {
        scan(rawRoot);
    }

    std::sort(rejected.begin(), rejected.end());
    rejected.erase(std::unique(rejected.begin(), rejected.end()), rejected.end());
    return rejected;
}

std::vector<ComponentStatus> ComponentInventory(const fs::path &root, std::string_view sequence,
                                                const std::vector<std::string> &cameras)
{
    const std::string sequenceName(sequence);
    std::vector<ComponentStatus> components;
    components.reserve(5U);

    // calibration -----------------------------------------------------------
    {
        ComponentStatus status;
        status.name = "calibration";
        const std::array<std::string, 3> required = {
            "calibration/calib_cam_to_pose.txt",
            "calibration/calib_cam_to_velo.txt",
            "calibration/perspective.txt",
        };
        for (const std::string &relative : required)
        {
            status.required.push_back(relative);
            const fs::path path = root / relative;
            if (IsRegularFile(path))
            {
                status.present.push_back(relative);
                ++status.fileCount;
                status.bytes += FileSize(path);
            }
            else
            {
                status.missing.push_back(relative);
            }
        }

        status.rejected = FindRejectedCalibration(root);

        // The official package ships extra files. They are allowed and must not
        // be reported as missing or rejected.
        std::set<std::string> requiredNames;
        for (const std::string &relative : required)
        {
            requiredNames.insert(fs::path(relative).filename().string());
        }

        std::vector<std::string> extras;
        std::error_code error;
        fs::directory_iterator iterator(root / "calibration", fs::directory_options::skip_permission_denied, error);
        const fs::directory_iterator end;
        if (!error)
        {
            for (; iterator != end; iterator.increment(error))
            {
                if (error)
                {
                    error.clear();
                    continue;
                }
                std::error_code fileError;
                if (!iterator->is_regular_file(fileError))
                {
                    continue;
                }
                const std::string name = iterator->path().filename().string();
                if (requiredNames.find(name) == requiredNames.end() && !IsForbiddenCalibrationName(name))
                {
                    extras.push_back(name);
                }
            }
        }
        if (!extras.empty())
        {
            std::sort(extras.begin(), extras.end());
            status.notes = "extra files allowed: " + JoinNames(extras);
        }

        components.push_back(std::move(status));
    }

    // poses ----------------------------------------------------------------
    {
        ComponentStatus status;
        status.name = "poses";
        struct PoseRequirement
        {
            std::string relative;
            std::size_t tokens;
            std::string label;
        };
        const std::array<PoseRequirement, 2> requirements = {{
            {"data_poses/" + sequenceName + "/poses.txt", kExpectedPoseTokens, "poses.txt"},
            {"data_poses/" + sequenceName + "/cam0_to_world.txt", kExpectedCam0WorldTokens, "cam0_to_world.txt"},
        }};

        std::ostringstream notes;
        bool firstNote = true;
        for (const PoseRequirement &requirement : requirements)
        {
            status.required.push_back(requirement.relative);
            const fs::path path = root / requirement.relative;
            if (!IsRegularFile(path))
            {
                status.missing.push_back(requirement.relative);
                continue;
            }

            status.present.push_back(requirement.relative);
            ++status.fileCount;
            status.bytes += FileSize(path);

            const PosesFileStats stats = InspectPosesFile(path, requirement.tokens);
            if (stats.malformed)
            {
                status.rejected.push_back(requirement.relative + " (" + stats.detail + ")");
                continue;
            }

            if (!firstNote)
            {
                notes << "; ";
            }
            firstNote = false;
            notes << requirement.label << ": " << stats.dataLines << " data lines";
        }
        status.notes = notes.str();
        components.push_back(std::move(status));
    }

    // images ---------------------------------------------------------------
    {
        ComponentStatus status;
        status.name = "images";
        std::vector<std::size_t> counts;
        std::ostringstream notes;
        bool firstNote = true;

        for (const std::string &camera : cameras)
        {
            const std::string relative = "data_2d_raw/" + sequenceName + "/" + camera + "/data_rect";
            status.required.push_back(relative);
            const fs::path directory = root / relative;
            if (!IsDirectory(directory))
            {
                status.missing.push_back(relative);
                counts.push_back(0U);
                continue;
            }

            status.present.push_back(relative);
            const DirectoryFiles files = CollectExtension(directory, ".png");
            status.fileCount += files.count;
            status.bytes += files.bytes;
            counts.push_back(files.count);

            if (files.count == 0U)
            {
                status.missing.push_back(relative + " (no *.png)");
                continue;
            }

            const fs::path firstPath = directory / files.first;
            int width = 0;
            int height = 0;
            int channels = 0;
            if (stbi_info(firstPath.string().c_str(), &width, &height, &channels) == 0)
            {
                status.rejected.push_back(RelativeTo(root, firstPath) + " (unreadable PNG header)");
                continue;
            }

            if (width != kExpectedImageWidth || height != kExpectedImageHeight)
            {
                status.rejected.push_back(RelativeTo(root, firstPath) + " (observed " + std::to_string(width) + "x" +
                                          std::to_string(height) + ", expected " + std::to_string(kExpectedImageWidth) +
                                          "x" + std::to_string(kExpectedImageHeight) + ")");
                continue;
            }

            if (!firstNote)
            {
                notes << "; ";
            }
            firstNote = false;
            notes << camera << ": first PNG " << width << "x" << height;
        }

        const bool countsDiffer =
            !counts.empty() && std::any_of(counts.begin(), counts.end(),
                                           [first = counts.front()](std::size_t value) { return value != first; });
        if (countsDiffer)
        {
            if (!notes.str().empty())
            {
                notes << "; ";
            }
            notes << "image counts differ across cameras";
        }
        status.notes = notes.str();
        components.push_back(std::move(status));
    }

    // semantics ------------------------------------------------------------
    {
        ComponentStatus status;
        status.name = "semantics";
        for (const std::string &camera : cameras)
        {
            const std::string relative = "data_2d_semantics/train/" + sequenceName + "/" + camera + "/semantic";
            status.required.push_back(relative);
            const fs::path directory = root / relative;
            if (!IsDirectory(directory))
            {
                status.missing.push_back(relative);
                continue;
            }

            status.present.push_back(relative);
            const DirectoryFiles files = CollectExtension(directory, ".png");
            status.fileCount += files.count;
            status.bytes += files.bytes;
            if (files.count == 0U)
            {
                status.missing.push_back(relative + " (no *.png)");
            }
        }
        components.push_back(std::move(status));
    }

    // lidar ----------------------------------------------------------------
    {
        ComponentStatus status;
        status.name = "lidar";
        const std::string relative = "data_3d_raw/" + sequenceName + "/velodyne_points/data";
        status.required.push_back(relative);
        const fs::path directory = root / relative;
        if (!IsDirectory(directory))
        {
            status.missing.push_back(relative);
            components.push_back(std::move(status));
            return components;
        }

        status.present.push_back(relative);
        const DirectoryFiles files = CollectExtension(directory, ".bin");
        status.fileCount += files.count;
        status.bytes += files.bytes;
        if (files.count == 0U)
        {
            status.missing.push_back(relative + " (no *.bin)");
            components.push_back(std::move(status));
            return components;
        }

        const fs::path firstPath = directory / files.first;
        const std::uintmax_t size = FileSize(firstPath);
        if (size == 0U || size % kLidarPointBytes != 0U)
        {
            status.rejected.push_back(RelativeTo(root, firstPath) + " (size " + std::to_string(size) +
                                      " is not a positive multiple of 16 bytes)");
        }
        components.push_back(std::move(status));
    }

    return components;
}
} // namespace raylidar::dataset
