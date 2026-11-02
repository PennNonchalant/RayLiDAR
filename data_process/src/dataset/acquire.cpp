#include <raylidar/dataset/acquire.h>

#include <raylidar/common/io.h>
#include <raylidar/dataset/layout.h>
#include <raylidar/dataset/preflight.h>

#include <archive.h>
#include <archive_entry.h>
#include <stb_image.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
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

constexpr int kExitLinkSource = 4;
constexpr int kExitComponent = 7;
constexpr int kExitArchive = 8;

constexpr int kExpectedImageWidth = 1408;
constexpr int kExpectedImageHeight = 376;

constexpr const char *kRequiredCalibration[] = {
    "calib_cam_to_pose.txt",
    "calib_cam_to_velo.txt",
    "perspective.txt",
};

constexpr const char *kForbiddenCalibration[] = {
    "calib_cam_to_cam.txt",
    "calib_velo_to_cam.txt",
    "calib_imu_to_velo.txt",
};

std::string Join(const std::vector<std::string> &values, std::string_view separator)
{
    std::string joined;
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        if (index != 0U)
        {
            joined += separator;
        }
        joined += values[index];
    }
    return joined;
}

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

std::string CanonicalString(const fs::path &path)
{
    std::error_code error;
    const fs::path canonical = fs::weakly_canonical(path, error);
    return error ? path.lexically_normal().string() : canonical.string();
}

std::string ArchiveError(struct archive *handle)
{
    const char *message = archive_error_string(handle);
    return message == nullptr ? std::string("unknown libarchive error") : std::string(message);
}

struct ArchiveDeleter
{
    void operator()(struct archive *handle) const
    {
        if (handle != nullptr)
        {
            archive_read_free(handle);
        }
    }
};

using ArchivePtr = std::unique_ptr<struct archive, ArchiveDeleter>;

ArchivePtr OpenArchive(const fs::path &archive)
{
    struct archive *handle = archive_read_new();
    if (handle == nullptr)
    {
        throw AcquisitionError("cannot allocate a libarchive reader", kExitArchive);
    }
    ArchivePtr reader(handle);
    archive_read_support_format_zip(handle);
    archive_read_support_format_tar(handle);
    archive_read_support_filter_gzip(handle);
    if (archive_read_open_filename(handle, archive.string().c_str(), 1U << 20U) != ARCHIVE_OK)
    {
        throw AcquisitionError("cannot open archive " + archive.string() + ": " + ArchiveError(handle), kExitArchive);
    }
    return reader;
}

// ---------------------------------------------------------------- path safety

std::optional<std::string> NormalizeMemberPath(std::string_view raw)
{
    if (raw.empty() || raw.front() == '/')
    {
        return std::nullopt;
    }

    std::vector<std::string> parts;
    std::size_t begin = 0;
    while (begin <= raw.size())
    {
        const std::size_t slash = raw.find('/', begin);
        const std::size_t stop = slash == std::string_view::npos ? raw.size() : slash;
        const std::string_view segment = raw.substr(begin, stop - begin);
        if (!segment.empty() && segment != ".")
        {
            if (segment == "..")
            {
                return std::nullopt;
            }
            parts.emplace_back(segment);
        }
        if (stop == raw.size())
        {
            break;
        }
        begin = stop + 1;
    }

    if (parts.empty())
    {
        return std::nullopt;
    }
    return Join(parts, "/");
}

std::string NormalizePrefix(std::string_view prefix)
{
    std::string value(prefix);
    while (value.rfind("./", 0U) == 0U)
    {
        value.erase(0U, 2U);
    }
    while (!value.empty() && value.back() == '/')
    {
        value.pop_back();
    }
    return value;
}

bool MatchesAllow(const std::string &relative, const std::vector<std::string> &prefixes)
{
    for (const std::string &prefix : prefixes)
    {
        if (prefix.empty())
        {
            continue;
        }
        if (relative == prefix)
        {
            return true;
        }
        if (relative.size() > prefix.size() && relative.compare(0U, prefix.size(), prefix) == 0 &&
            relative[prefix.size()] == '/')
        {
            return true;
        }
    }
    return false;
}

/// Map an archive member onto the destination-relative path, trying the member
/// unchanged first and then with exactly one leading wrapper segment stripped.
std::optional<std::string> MapMember(const std::string &normalized, const std::vector<std::string> &prefixes)
{
    if (MatchesAllow(normalized, prefixes))
    {
        return normalized;
    }
    const std::size_t slash = normalized.find('/');
    if (slash != std::string::npos && slash + 1U < normalized.size())
    {
        const std::string stripped = normalized.substr(slash + 1U);
        if (MatchesAllow(stripped, prefixes))
        {
            return stripped;
        }
    }
    return std::nullopt;
}

void EnsureSafeParents(const fs::path &dest, const fs::path &parent)
{
    std::error_code error;
    fs::create_directories(dest, error);
    if (!fs::is_directory(dest))
    {
        throw AcquisitionError("cannot create destination directory: " + dest.string(), kExitArchive);
    }

    const fs::path relative = parent.lexically_relative(dest);
    if (relative.empty())
    {
        return;
    }
    if (*relative.begin() == "..")
    {
        throw AcquisitionError("refusing to write outside the destination: " + parent.string(), kExitArchive);
    }

    fs::path current = dest;
    for (const fs::path &part : relative)
    {
        current /= part;
        const fs::file_status status = fs::symlink_status(current, error);
        if (!error && fs::is_symlink(status))
        {
            throw AcquisitionError("refusing to follow a symlink inside the destination: " + current.string(),
                                   kExitArchive);
        }
        error.clear();
        if (!fs::exists(status))
        {
            fs::create_directory(current, error);
            if (error)
            {
                throw AcquisitionError("cannot create directory " + current.string() + ": " + error.message(),
                                       kExitArchive);
            }
        }
        else if (!fs::is_directory(status))
        {
            throw AcquisitionError("destination component is not a directory: " + current.string(), kExitArchive);
        }
    }
}

void WriteEntry(struct archive *reader, const fs::path &dest, const std::string &member, const std::string &relative,
                ExtractResult &result)
{
    const fs::path target = dest / fs::path(relative);
    EnsureSafeParents(dest, target.parent_path());

    std::error_code error;
    const fs::file_status existing = fs::symlink_status(target, error);
    if (!error && fs::is_symlink(existing))
    {
        throw AcquisitionError("refusing to overwrite a symlink: " + target.string(), kExitArchive);
    }

    std::ofstream output(target, std::ios::binary | std::ios::trunc);
    if (!output)
    {
        throw AcquisitionError("cannot write extracted file: " + target.string(), kExitArchive);
    }

    std::array<char, 1U << 16U> buffer{};
    la_ssize_t count = 0;
    while ((count = archive_read_data(reader, buffer.data(), buffer.size())) > 0)
    {
        output.write(buffer.data(), static_cast<std::streamsize>(count));
        result.bytes += static_cast<std::uintmax_t>(count);
    }
    if (count < 0)
    {
        throw AcquisitionError("cannot read archive member '" + member + "': " + ArchiveError(reader), kExitArchive);
    }
    output.close();
    if (!output)
    {
        throw AcquisitionError("failed to write extracted file: " + target.string(), kExitArchive);
    }

    fs::permissions(target,
                    fs::perms::owner_read | fs::perms::owner_write | fs::perms::group_read | fs::perms::others_read,
                    fs::perm_options::replace, error);
    result.extracted.push_back(relative);
    result.mapping[member] = relative;
}

std::uintmax_t EntrySize(struct archive_entry *entry)
{
    const la_int64_t size = archive_entry_size(entry);
    return size < 0 ? 0U : static_cast<std::uintmax_t>(size);
}

void RejectUnsafeEntry(struct archive_entry *entry, const char *raw)
{
    if (archive_entry_symlink(entry) != nullptr)
    {
        throw AcquisitionError("archive contains a symlink member: " + std::string(raw), kExitArchive);
    }
    if (archive_entry_hardlink(entry) != nullptr)
    {
        throw AcquisitionError("archive contains a hardlink member: " + std::string(raw), kExitArchive);
    }
    const auto type = archive_entry_filetype(entry);
    if (type != AE_IFREG && type != AE_IFDIR)
    {
        throw AcquisitionError("archive contains an unsupported special member: " + std::string(raw), kExitArchive);
    }
}

struct PassSummary
{
    std::uintmax_t totalBytes = 0;
    std::size_t members = 0;
    std::uintmax_t allowlistedBytes = 0;
    std::size_t skippedMembers = 0;
};

PassSummary PreScan(struct archive *reader, const std::vector<std::string> &prefixes, std::uintmax_t maxBytes,
                    std::size_t maxMembers)
{
    PassSummary summary;
    struct archive_entry *entry = nullptr;
    while (true)
    {
        const int status = archive_read_next_header(reader, &entry);
        if (status == ARCHIVE_EOF)
        {
            break;
        }
        if (status != ARCHIVE_OK)
        {
            throw AcquisitionError("cannot read archive header: " + ArchiveError(reader), kExitArchive);
        }

        if (++summary.members > maxMembers)
        {
            throw AcquisitionError("archive has more than " + std::to_string(maxMembers) + " members", kExitArchive);
        }

        const char *raw = archive_entry_pathname(entry);
        if (raw == nullptr)
        {
            throw AcquisitionError("archive member has no name", kExitArchive);
        }
        const std::optional<std::string> normalized = NormalizeMemberPath(raw);
        if (!normalized)
        {
            throw AcquisitionError("unsafe archive member path: " + std::string(raw), kExitArchive);
        }
        RejectUnsafeEntry(entry, raw);

        summary.totalBytes += EntrySize(entry);
        if (summary.totalBytes > maxBytes)
        {
            throw AcquisitionError("archive uncompressed size exceeds " + std::to_string(maxBytes) + " bytes",
                                   kExitArchive);
        }

        const std::optional<std::string> mapped = MapMember(*normalized, prefixes);
        if (!mapped)
        {
            ++summary.skippedMembers;
            continue;
        }
        if (archive_entry_filetype(entry) == AE_IFDIR)
        {
            continue;
        }
        summary.allowlistedBytes += EntrySize(entry);
    }
    return summary;
}

void CheckDisk(const fs::path &dest, std::uintmax_t extractBytes, double diskFactor,
               const std::optional<std::uintmax_t> &availableBytesOverride)
{
    const double factor = diskFactor < 0.0 ? 0.0 : diskFactor;
    const long double requiredRaw = static_cast<long double>(extractBytes) * static_cast<long double>(factor);
    const std::uintmax_t required = requiredRaw > static_cast<long double>(std::numeric_limits<std::uintmax_t>::max())
                                        ? std::numeric_limits<std::uintmax_t>::max()
                                        : static_cast<std::uintmax_t>(requiredRaw);

    std::uintmax_t available = 0;
    if (availableBytesOverride)
    {
        available = *availableBytesOverride;
    }
    else
    {
        std::error_code error;
        const fs::space_info info = fs::space(dest, error);
        if (error)
        {
            throw AcquisitionError("cannot determine free space for " + dest.string() + ": " + error.message(),
                                   kExitArchive);
        }
        available = info.available;
    }

    if (available < required)
    {
        std::ostringstream message;
        message << "insufficient free space in " << dest.string() << ": required " << required << " bytes (" << factor
                << "x " << extractBytes << " allowlisted bytes), free " << available << " bytes";
        throw AcquisitionError(message.str(), kExitArchive);
    }
}

// ------------------------------------------------------------ component checks

struct CameraCheck
{
    bool ok = false;
    std::size_t count = 0;
    int width = 0;
    int height = 0;
    std::string first;
    std::string error;
};

CameraCheck InspectCamera(const fs::path &root, const std::string &sequence, const std::string &camera)
{
    CameraCheck check;
    const fs::path directory = root / "data_2d_raw" / sequence / camera / "data_rect";
    std::error_code error;
    if (!fs::is_directory(directory, error))
    {
        check.error = camera + ": missing data_rect directory " + directory.string();
        return check;
    }

    std::vector<fs::path> files;
    for (fs::directory_iterator iterator(directory, fs::directory_options::skip_permission_denied, error);
         !error && iterator != fs::directory_iterator(); iterator.increment(error))
    {
        std::error_code fileError;
        if (!iterator->is_regular_file(fileError))
        {
            continue;
        }
        if (iterator->path().extension() == ".png")
        {
            files.push_back(iterator->path());
        }
    }
    std::sort(files.begin(), files.end());
    check.count = files.size();
    if (files.empty())
    {
        check.error = camera + ": no *.png images under " + directory.string();
        return check;
    }

    check.first = files.front().filename().string();
    int width = 0;
    int height = 0;
    int channels = 0;
    if (stbi_info(files.front().string().c_str(), &width, &height, &channels) == 0)
    {
        check.error = camera + ": unreadable PNG header " + files.front().string();
        return check;
    }
    check.width = width;
    check.height = height;
    if (width != kExpectedImageWidth || height != kExpectedImageHeight)
    {
        check.error = camera + ": first PNG " + check.first + " is " + std::to_string(width) + "x" +
                      std::to_string(height) + ", expected " + std::to_string(kExpectedImageWidth) + "x" +
                      std::to_string(kExpectedImageHeight);
        return check;
    }
    check.ok = true;
    return check;
}

/// Validate every camera's official data_rect directory. Returns an empty string
/// on success, otherwise a message naming the camera and the observed mismatch.
std::string ValidateOfficialImages(const fs::path &root, const std::string &sequence,
                                   const std::vector<std::string> &cameras,
                                   std::map<std::string, CameraCheck> *checks = nullptr)
{
    std::vector<CameraCheck> results;
    results.reserve(cameras.size());
    for (const std::string &camera : cameras)
    {
        CameraCheck check = InspectCamera(root, sequence, camera);
        if (!check.ok)
        {
            if (checks != nullptr)
            {
                (*checks)[camera] = check;
            }
            return check.error;
        }
        results.push_back(check);
        if (checks != nullptr)
        {
            (*checks)[camera] = check;
        }
    }

    for (std::size_t index = 1; index < results.size(); ++index)
    {
        if (results[index].count != results.front().count)
        {
            return cameras[index] + ": " + std::to_string(results[index].count) + " PNG files, " + cameras.front() +
                   ": " + std::to_string(results.front().count) + " PNG files (counts must match)";
        }
    }
    return {};
}

bool CalibrationComplete(const fs::path &root)
{
    for (const char *name : kRequiredCalibration)
    {
        if (!IsRegularFile(root / "calibration" / name))
        {
            return false;
        }
    }
    return true;
}

bool PosesComplete(const fs::path &root, const std::string &sequence)
{
    return IsRegularFile(root / "data_poses" / sequence / "poses.txt") &&
           IsRegularFile(root / "data_poses" / sequence / "cam0_to_world.txt");
}

bool SemanticsComplete(const fs::path &root, const std::string &sequence, const std::vector<std::string> &cameras)
{
    for (const std::string &camera : cameras)
    {
        const fs::path directory = root / "data_2d_semantics" / "train" / sequence / camera / "semantic";
        std::error_code error;
        bool found = false;
        for (fs::directory_iterator iterator(directory, fs::directory_options::skip_permission_denied, error);
             !error && iterator != fs::directory_iterator(); iterator.increment(error))
        {
            std::error_code fileError;
            if (iterator->is_regular_file(fileError) && iterator->path().extension() == ".png")
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            return false;
        }
    }
    return true;
}

// ------------------------------------------------------------------- archives

enum class ComponentKind
{
    Calibration,
    Poses,
    Semantics,
    Images,
    Unknown,
};

struct ArchiveSpec
{
    fs::path path;
    std::string name;
    std::string base;
    ComponentKind kind = ComponentKind::Unknown;
    std::string camera;
    fs::path destination;
    std::vector<std::string> allowPrefixes;
    std::string granularity;
};

std::string LowerCase(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
    return value;
}

bool EndsWith(std::string_view value, std::string_view suffix)
{
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool LooksLikeArchive(const std::string &name)
{
    const std::string lower = LowerCase(name);
    return EndsWith(lower, ".zip") || EndsWith(lower, ".tar") || EndsWith(lower, ".tar.gz") || EndsWith(lower, ".tgz");
}

std::string StripArchiveSuffix(std::string name)
{
    const std::string lower = LowerCase(name);
    if (EndsWith(lower, ".tar.gz"))
    {
        return name.substr(0U, name.size() - 7U);
    }
    if (EndsWith(lower, ".zip") || EndsWith(lower, ".tgz"))
    {
        return name.substr(0U, name.size() - 4U);
    }
    if (EndsWith(lower, ".tar"))
    {
        return name.substr(0U, name.size() - 4U);
    }
    return name;
}

ArchiveSpec MakeSpec(const fs::path &staging, const std::string &name, const std::string &sequence)
{
    ArchiveSpec spec;
    spec.path = staging / name;
    spec.name = name;
    spec.base = StripArchiveSuffix(name);

    if (spec.base == "calibration")
    {
        spec.kind = ComponentKind::Calibration;
        spec.destination = fs::path(); // filled by caller (root)
        spec.allowPrefixes = {"calibration"};
        spec.granularity = "all_sequences";
        return spec;
    }
    if (spec.base.rfind("data_poses", 0U) == 0U)
    {
        spec.kind = ComponentKind::Poses;
        spec.destination = fs::path(); // filled by caller (root/data_poses)
        spec.allowPrefixes = {sequence};
        spec.granularity = "all_sequences";
        return spec;
    }
    if (spec.base.rfind("data_2d_semantics", 0U) == 0U)
    {
        spec.kind = ComponentKind::Semantics;
        spec.allowPrefixes = {"data_2d_semantics/train/" + sequence};
        spec.granularity = "all_sequences";
        return spec;
    }
    const std::string imagePrefix = sequence + "_image_";
    if (spec.base.rfind(imagePrefix, 0U) == 0U)
    {
        spec.kind = ComponentKind::Images;
        spec.camera = "image_" + spec.base.substr(imagePrefix.size());
        spec.allowPrefixes = {sequence + "/" + spec.camera};
        spec.granularity = "per_sequence_per_camera";
        return spec;
    }
    return spec;
}

void InstallTree(const fs::path &source, const fs::path &destination)
{
    std::error_code error;
    fs::create_directories(destination, error);
    for (fs::recursive_directory_iterator iterator(source, fs::directory_options::skip_permission_denied, error);
         !error && iterator != fs::recursive_directory_iterator(); iterator.increment(error))
    {
        std::error_code entryError;
        if (iterator->is_directory(entryError))
        {
            fs::create_directories(destination / iterator->path().lexically_relative(source), entryError);
            continue;
        }
        if (!iterator->is_regular_file(entryError))
        {
            continue;
        }
        const fs::path target = destination / iterator->path().lexically_relative(source);
        fs::create_directories(target.parent_path(), entryError);
        fs::copy_file(iterator->path(), target, fs::copy_options::overwrite_existing, entryError);
        if (entryError)
        {
            throw AcquisitionError("cannot install " + iterator->path().string() + " -> " + target.string() + ": " +
                                       entryError.message(),
                                   kExitArchive);
        }
    }
    if (error)
    {
        throw AcquisitionError("cannot walk extracted tree " + source.string() + ": " + error.message(), kExitArchive);
    }
}

// ---------------------------------------------------------------------- links

nlohmann::json MakeLinkRecord(const std::string &key, const fs::path &target, const std::string &source)
{
    return nlohmann::json{{"key", key}, {"target", target.string()}, {"source", source}};
}

nlohmann::json EnsureLink(const std::string &key, const fs::path &target, const std::string &source)
{
    nlohmann::json record = MakeLinkRecord(key, target, source);
    std::error_code error;

    if (source.empty())
    {
        if (key.find("timestamps") != std::string::npos)
        {
            record["pre_existing"] = false;
            record["skipped"] = true;
            record["skip_reason"] = "approved_null_source";
            return record;
        }
        if (key == "lidar" && IsDirectory(target))
        {
            record["pre_existing"] = true;
            record["skipped"] = true;
            record["skip_reason"] = "existing_component";
            return record;
        }
        throw AcquisitionError("missing link source for '" + key + "' (set link_sources." + key +
                                   " or pass --link-source " + key + "=PATH)",
                               kExitLinkSource);
    }
    if (source == "REQUIRED_USER_CONFIRM")
    {
        throw AcquisitionError("unconfirmed link source for '" + key + "': REQUIRED_USER_CONFIRM", kExitLinkSource);
    }

    const fs::path sourcePath(source);
    if (!fs::exists(sourcePath, error))
    {
        throw AcquisitionError("link source for '" + key + "' does not exist: " + source, kExitLinkSource);
    }
    const std::string canonicalSource = CanonicalString(sourcePath);

    const fs::file_status status = fs::symlink_status(target, error);
    error.clear();
    if (fs::exists(status))
    {
        const std::string canonicalTarget = CanonicalString(target);
        if (canonicalTarget == canonicalSource)
        {
            record["pre_existing"] = true;
            record["resolved"] = canonicalTarget;
            return record;
        }
        throw AcquisitionError("conflicting link for '" + key + "': " + target.string() + " resolves to " +
                                   canonicalTarget + ", expected " + canonicalSource,
                               kExitLinkSource);
    }

    fs::create_directories(target.parent_path(), error);
    if (error)
    {
        throw AcquisitionError("cannot create parent for link '" + key + "': " + target.parent_path().string() + ": " +
                                   error.message(),
                               kExitLinkSource);
    }
    if (fs::is_directory(sourcePath, error))
    {
        fs::create_directory_symlink(sourcePath, target, error);
    }
    else
    {
        fs::create_symlink(sourcePath, target, error);
    }
    if (error)
    {
        throw AcquisitionError("cannot create link '" + key + "' -> " + source + ": " + error.message(),
                               kExitLinkSource);
    }
    record["pre_existing"] = false;
    record["resolved"] = canonicalSource;
    return record;
}

// ------------------------------------------------------------------- manifest

std::map<std::string, std::string> ReadSha256Sums(const fs::path &staging)
{
    std::map<std::string, std::string> sums;
    std::ifstream input(staging / "SHA256SUMS");
    if (!input)
    {
        return sums;
    }
    std::string line;
    while (std::getline(input, line))
    {
        const std::size_t firstSpace = line.find_first_of(" \t");
        if (firstSpace == std::string::npos)
        {
            continue;
        }
        std::string hash = line.substr(0U, firstSpace);
        std::string name = line.substr(line.find_first_not_of(" \t", firstSpace));
        if (!name.empty() && name.front() == '*')
        {
            name.erase(0U, 1U);
        }
        if (!hash.empty() && !name.empty())
        {
            sums[name] = LowerCase(hash);
        }
    }
    return sums;
}

nlohmann::json ComponentSummary(const PreflightReport &report)
{
    nlohmann::json components = nlohmann::json::object();
    for (const ComponentStatus &component : report.components)
    {
        components[component.name] = {
            {"files", component.fileCount},
            {"bytes", component.bytes},
            {"ok", component.Ok()},
        };
    }
    return components;
}

nlohmann::json SequenceDirectories(const fs::path &directory, const std::string &sequence)
{
    nlohmann::json extras = nlohmann::json::array();
    std::error_code error;
    fs::directory_iterator iterator(directory, fs::directory_options::skip_permission_denied, error);
    if (error)
    {
        return extras;
    }
    std::vector<std::string> names;
    for (; iterator != fs::directory_iterator(); iterator.increment(error))
    {
        if (error)
        {
            break;
        }
        std::error_code directoryError;
        if (iterator->is_directory(directoryError))
        {
            const std::string name = iterator->path().filename().string();
            if (name != sequence)
            {
                names.push_back(name);
            }
        }
    }
    std::sort(names.begin(), names.end());
    for (const std::string &name : names)
    {
        extras.push_back(name);
    }
    return extras;
}

} // namespace

AcquisitionError::AcquisitionError(const std::string &what, int exitCode)
    : std::runtime_error(what), exitCode_(exitCode)
{
}

int AcquisitionError::exitCode() const noexcept
{
    return exitCode_;
}

ExtractResult SafeExtract(const std::filesystem::path &archive, const std::filesystem::path &dest,
                          const std::vector<std::string> &allowPrefixes, bool stream, std::uintmax_t maxBytes,
                          std::size_t maxMembers, double diskFactor, std::uintmax_t estimatedExtractBytes,
                          std::optional<std::uintmax_t> availableBytesOverride)
{
    if (!IsRegularFile(archive))
    {
        throw AcquisitionError("archive not found: " + archive.string(), kExitArchive);
    }

    std::vector<std::string> prefixes;
    prefixes.reserve(allowPrefixes.size());
    for (const std::string &prefix : allowPrefixes)
    {
        const std::string normalized = NormalizePrefix(prefix);
        if (!normalized.empty())
        {
            prefixes.push_back(normalized);
        }
    }

    {
        std::error_code error;
        fs::create_directories(dest, error);
        if (error || !fs::is_directory(dest))
        {
            throw AcquisitionError("cannot prepare destination " + dest.string(), kExitArchive);
        }
    }

    ExtractResult result;

    if (stream)
    {
        if (estimatedExtractBytes == 0U)
        {
            throw AcquisitionError("stream extraction requires a positive estimated extraction size", kExitArchive);
        }
        CheckDisk(dest, estimatedExtractBytes, diskFactor, availableBytesOverride);

        ArchivePtr reader = OpenArchive(archive);
        struct archive_entry *entry = nullptr;
        std::size_t members = 0;
        std::uintmax_t totalBytes = 0;
        while (true)
        {
            const int status = archive_read_next_header(reader.get(), &entry);
            if (status == ARCHIVE_EOF)
            {
                break;
            }
            if (status != ARCHIVE_OK)
            {
                throw AcquisitionError("cannot read archive header: " + ArchiveError(reader.get()), kExitArchive);
            }
            if (++members > maxMembers)
            {
                throw AcquisitionError("archive has more than " + std::to_string(maxMembers) + " members",
                                       kExitArchive);
            }
            const char *raw = archive_entry_pathname(entry);
            if (raw == nullptr)
            {
                throw AcquisitionError("archive member has no name", kExitArchive);
            }
            const std::optional<std::string> normalized = NormalizeMemberPath(raw);
            if (!normalized)
            {
                throw AcquisitionError("unsafe archive member path: " + std::string(raw), kExitArchive);
            }
            RejectUnsafeEntry(entry, raw);
            totalBytes += EntrySize(entry);
            if (totalBytes > maxBytes)
            {
                throw AcquisitionError("archive uncompressed size exceeds " + std::to_string(maxBytes) + " bytes",
                                       kExitArchive);
            }
            const std::optional<std::string> mapped = MapMember(*normalized, prefixes);
            if (!mapped)
            {
                ++result.skippedMembers;
                continue;
            }
            if (archive_entry_filetype(entry) == AE_IFDIR)
            {
                continue;
            }
            WriteEntry(reader.get(), dest, std::string(raw), *mapped, result);
        }
        std::sort(result.extracted.begin(), result.extracted.end());
        return result;
    }

    PassSummary summary;
    {
        ArchivePtr reader = OpenArchive(archive);
        summary = PreScan(reader.get(), prefixes, maxBytes, maxMembers);
    }

    result.allowlistedBytes = summary.allowlistedBytes;
    result.skippedMembers = summary.skippedMembers;
    CheckDisk(dest, summary.allowlistedBytes, diskFactor, availableBytesOverride);

    {
        ArchivePtr reader = OpenArchive(archive);
        struct archive_entry *entry = nullptr;
        while (true)
        {
            const int status = archive_read_next_header(reader.get(), &entry);
            if (status == ARCHIVE_EOF)
            {
                break;
            }
            if (status != ARCHIVE_OK)
            {
                throw AcquisitionError("cannot read archive header: " + ArchiveError(reader.get()), kExitArchive);
            }
            const char *raw = archive_entry_pathname(entry);
            if (raw == nullptr)
            {
                throw AcquisitionError("archive member has no name", kExitArchive);
            }
            const std::optional<std::string> normalized = NormalizeMemberPath(raw);
            if (!normalized)
            {
                throw AcquisitionError("unsafe archive member path: " + std::string(raw), kExitArchive);
            }
            const std::optional<std::string> mapped = MapMember(*normalized, prefixes);
            if (!mapped)
            {
                continue;
            }
            if (archive_entry_filetype(entry) == AE_IFDIR)
            {
                continue;
            }
            WriteEntry(reader.get(), dest, std::string(raw), *mapped, result);
        }
    }

    std::sort(result.extracted.begin(), result.extracted.end());
    return result;
}

nlohmann::json InstallStaging(const std::filesystem::path &staging, const std::filesystem::path &root,
                              std::string_view sequence, const std::vector<std::string> &cameras,
                              std::string_view policy, const std::map<std::string, std::string> &linkSources,
                              const std::filesystem::path &manifestPath, bool streamExtract,
                              std::uintmax_t estimatedExtractBytes)
{
    const std::string seq(sequence);
    const fs::path stagingDir(staging);
    const fs::path rootDir(root);

    if (seq.empty())
    {
        throw AcquisitionError("dataset sequence is not configured", kExitLinkSource);
    }
    if (!IsDirectory(stagingDir))
    {
        throw AcquisitionError("staging directory is not a directory: " + stagingDir.string(), kExitComponent);
    }

    {
        std::error_code error;
        fs::create_directories(rootDir, error);
        if (error || !IsDirectory(rootDir))
        {
            throw AcquisitionError("cannot create dataset root " + rootDir.string(), kExitComponent);
        }
    }

    std::vector<std::string> cameraList = cameras;
    if (cameraList.empty())
    {
        cameraList = {"image_00", "image_01"};
    }

    // ---- resolve the policy ------------------------------------------------
    std::map<std::string, CameraCheck> cameraChecks;
    std::string imageError = ValidateOfficialImages(rootDir, seq, cameraList, &cameraChecks);

    std::string effectivePolicy(policy);
    if (effectivePolicy == "auto")
    {
        if (imageError.empty())
        {
            effectivePolicy = "official";
        }
        else
        {
            const auto image00 = linkSources.find("image_00");
            const auto image01 = linkSources.find("image_01");
            const bool reuseSources = image00 != linkSources.end() && !image00->second.empty() &&
                                      image01 != linkSources.end() && !image01->second.empty();
            if (reuseSources)
            {
                effectivePolicy = "reuse_local_after_g1";
            }
            else
            {
                throw AcquisitionError("auto policy: official data_rect images are not valid (" + imageError +
                                           ") and reuse after G1 is not accepted (accept_reuse_after_g1)",
                                       kExitComponent);
            }
        }
    }
    if (effectivePolicy != "official" && effectivePolicy != "reuse_local_after_g1")
    {
        throw AcquisitionError("unknown image policy '" + effectivePolicy + "'", kExitLinkSource);
    }
    const bool official = effectivePolicy == "official";

    // ---- enumerate the staged archives ------------------------------------
    std::vector<std::string> names;
    {
        std::error_code error;
        for (fs::directory_iterator iterator(stagingDir, fs::directory_options::skip_permission_denied, error);
             !error && iterator != fs::directory_iterator(); iterator.increment(error))
        {
            std::error_code fileError;
            if (iterator->is_regular_file(fileError) || fs::is_symlink(iterator->symlink_status(fileError)))
            {
                names.push_back(iterator->path().filename().string());
            }
        }
    }
    std::sort(names.begin(), names.end());

    const std::map<std::string, std::string> officialSums = ReadSha256Sums(stagingDir);

    std::vector<ArchiveSpec> specs;
    nlohmann::json archives = nlohmann::json::array();
    nlohmann::json skippedArchives = nlohmann::json::array();

    for (const std::string &name : names)
    {
        if (!LooksLikeArchive(name))
        {
            continue;
        }
        ArchiveSpec spec = MakeSpec(stagingDir, name, seq);
        if (spec.kind == ComponentKind::Unknown)
        {
            skippedArchives.push_back({{"name", name}, {"reason", "unrecognised_archive"}});
            continue;
        }
        switch (spec.kind)
        {
        case ComponentKind::Calibration:
            spec.destination = rootDir;
            break;
        case ComponentKind::Poses:
            spec.destination = rootDir / "data_poses";
            break;
        case ComponentKind::Semantics:
            spec.destination = rootDir;
            break;
        case ComponentKind::Images:
            spec.destination = rootDir / "data_2d_raw";
            break;
        case ComponentKind::Unknown:
            break;
        }

        std::string hash;
        std::string hashSource = "computed_locally";
        try
        {
            hash = Sha256File(spec.path);
        }
        catch (const IoError &error)
        {
            throw AcquisitionError("cannot hash archive " + spec.path.string() + ": " + error.what(), kExitArchive);
        }
        const auto officialHash = officialSums.find(name);
        if (officialHash != officialSums.end())
        {
            if (LowerCase(officialHash->second) != hash)
            {
                throw AcquisitionError("archive hash mismatch for " + name + ": expected " + officialHash->second +
                                           ", computed " + hash,
                                       kExitArchive);
            }
            hashSource = "official_sha256sums";
        }

        archives.push_back({{"name", name},
                            {"bytes", fs::file_size(spec.path)},
                            {"sha256", hash},
                            {"sha256_source", hashSource},
                            {"granularity", spec.granularity}});
        specs.push_back(std::move(spec));
    }

    // ---- install the archives ---------------------------------------------
    std::set<std::string> preExistingCameras;
    std::map<std::string, std::string> installedFrom;
    if (official)
    {
        for (const std::string &camera : cameraList)
        {
            std::error_code error;
            if (fs::exists(rootDir / "data_2d_raw" / seq / camera / "data_rect", error))
            {
                preExistingCameras.insert(camera);
                installedFrom[camera] = "pre_existing";
                continue;
            }
            const std::string expected = seq + "_" + camera + ".zip";
            const bool available = std::any_of(specs.begin(), specs.end(), [&camera](const ArchiveSpec &spec) {
                return spec.kind == ComponentKind::Images && spec.camera == camera;
            });
            if (!available)
            {
                throw AcquisitionError("missing official image archive for " + camera + ": " + expected,
                                       kExitComponent);
            }
        }
    }

    std::uintmax_t skippedMembers = 0;
    const fs::path extractRoot = stagingDir / "_extract";

    for (const ArchiveSpec &spec : specs)
    {
        if (spec.kind == ComponentKind::Images)
        {
            if (!official)
            {
                skippedArchives.push_back({{"name", spec.name}, {"reason", "reuse_policy"}});
                continue;
            }
            const auto check = cameraChecks.find(spec.camera);
            if (check != cameraChecks.end() && check->second.ok)
            {
                skippedArchives.push_back({{"name", spec.name}, {"reason", "already_installed"}});
                continue;
            }
        }

        bool alreadyInstalled = false;
        switch (spec.kind)
        {
        case ComponentKind::Calibration:
            alreadyInstalled = CalibrationComplete(rootDir);
            break;
        case ComponentKind::Poses:
            alreadyInstalled = PosesComplete(rootDir, seq);
            break;
        case ComponentKind::Semantics:
            alreadyInstalled = SemanticsComplete(rootDir, seq, cameraList);
            break;
        case ComponentKind::Images:
        case ComponentKind::Unknown:
            alreadyInstalled = false;
            break;
        }

        if (alreadyInstalled)
        {
            skippedArchives.push_back({{"name", spec.name}, {"reason", "already_installed"}});
            continue;
        }

        const fs::path temporary = extractRoot / spec.base;
        {
            std::error_code error;
            fs::remove_all(temporary, error);
            fs::create_directories(temporary, error);
            if (error)
            {
                throw AcquisitionError(
                    "cannot create extraction directory " + temporary.string() + ": " + error.message(), kExitArchive);
            }
        }

        const ExtractResult result = SafeExtract(spec.path, temporary, spec.allowPrefixes, streamExtract,
                                                 120'000'000'000ULL, 1'000'000U, 2.5, estimatedExtractBytes);
        skippedMembers += result.skippedMembers;
        if (spec.kind == ComponentKind::Images)
        {
            installedFrom[spec.camera] = spec.name;
        }
        InstallTree(temporary, spec.destination);
        {
            std::error_code error;
            fs::remove_all(temporary, error);
        }
    }

    // ---- validate the installed components --------------------------------
    const std::vector<std::string> rejected = FindRejectedCalibration(rootDir);
    if (!rejected.empty())
    {
        throw AcquisitionError("KITTI-raw calibration is not accepted: " + rejected.front(), kExitComponent);
    }

    if (!CalibrationComplete(rootDir))
    {
        std::vector<std::string> missing;
        for (const char *name : kRequiredCalibration)
        {
            if (!IsRegularFile(rootDir / "calibration" / name))
            {
                missing.emplace_back(name);
            }
        }
        throw AcquisitionError("missing official calibration files: " + Join(missing, ", "), kExitComponent);
    }

    if (official)
    {
        cameraChecks.clear();
        const std::string error = ValidateOfficialImages(rootDir, seq, cameraList, &cameraChecks);
        if (!error.empty())
        {
            throw AcquisitionError("official data_rect images are not valid: " + error, kExitComponent);
        }
    }

    // ---- links -------------------------------------------------------------
    const std::vector<std::string> linkOrder =
        official
            ? std::vector<std::string>{"lidar", "image_00_timestamps", "image_01_timestamps"}
            : std::vector<std::string>{"image_00", "image_01", "lidar", "image_00_timestamps", "image_01_timestamps"};

    std::vector<std::string> linkCameras = cameraList;
    if (linkCameras.empty())
    {
        linkCameras = {"image_00", "image_01"};
    }

    nlohmann::json links = nlohmann::json::array();
    for (const std::string &key : linkOrder)
    {
        fs::path target;
        if (key == "lidar")
        {
            target = rootDir / "data_3d_raw" / seq / "velodyne_points";
        }
        else if (key == "image_00" || key == "image_01")
        {
            target = rootDir / "data_2d_raw" / seq / key / "data_rect";
        }
        else if (key == "image_00_timestamps")
        {
            target = rootDir / "data_2d_raw" / seq / "image_00" / "timestamps.txt";
        }
        else if (key == "image_01_timestamps")
        {
            target = rootDir / "data_2d_raw" / seq / "image_01" / "timestamps.txt";
        }
        else
        {
            continue;
        }

        const auto source = linkSources.find(key);
        const std::string sourceValue = source == linkSources.end() ? std::string() : source->second;
        nlohmann::json record = EnsureLink(key, target, sourceValue);
        record["policy"] = effectivePolicy;
        links.push_back(std::move(record));
    }

    // ---- post-install preflight -------------------------------------------
    const std::vector<std::string> forbiddenNames = {
        "calib_cam_to_cam.txt",
        "calib_velo_to_cam.txt",
        "calib_imu_to_velo.txt",
    };
    const PreflightReport report = RunPreflight(rootDir, seq, cameraList, forbiddenNames);
    if (!report.ok)
    {
        throw AcquisitionError("post-install preflight failed:\n" + RenderSummary(report), kExitComponent);
    }

    // ---- readiness manifest ------------------------------------------------
    nlohmann::json images = nlohmann::json::object();
    for (const auto &[camera, check] : cameraChecks)
    {
        const auto origin = installedFrom.find(camera);
        images[camera] = {
            {"path", (rootDir / "data_2d_raw" / seq / camera / "data_rect").string()},
            {"count", check.count},
            {"width", check.width},
            {"height", check.height},
            {"pre_existing", preExistingCameras.count(camera) != 0U},
            {"installed_from", origin == installedFrom.end() ? std::string("pre_existing") : origin->second}};
    }

    nlohmann::json extraSequences = nlohmann::json::object();
    extraSequences["data_poses"] = SequenceDirectories(rootDir / "data_poses", seq);
    extraSequences["data_2d_raw"] = SequenceDirectories(rootDir / "data_2d_raw", seq);
    extraSequences["data_2d_semantics"] = SequenceDirectories(rootDir / "data_2d_semantics" / "train", seq);
    extraSequences["data_3d_raw"] = SequenceDirectories(rootDir / "data_3d_raw", seq);

    nlohmann::json manifest = {
        {"schema_version", 1},
        {"release", "KITTI-360"},
        {"source", "https://www.cvlibs.net/datasets/kitti-360/user_login.php"},
        {"license", "KITTI-360 non-commercial research use"},
        {"sequence", seq},
        {"installed_at_utc", UtcNowIso8601()},
        {"root", rootDir.string()},
        {"policy", effectivePolicy},
        {"requested_policy", std::string(policy)},
        {"image_provenance", official ? "official_data_rect" : "local_repackaged_1408x376"},
        {"archives", std::move(archives)},
        {"components", ComponentSummary(report)},
        {"images", std::move(images)},
        {"links", std::move(links)},
        {"skipped_archives", std::move(skippedArchives)},
        {"skipped_members", skippedMembers},
        {"extra_sequences", std::move(extraSequences)},
    };

    WriteJson(manifestPath, manifest);
    return manifest;
}
} // namespace raylidar::dataset
