#include <raylidar/dataset/preflight.h>

#include <raylidar/common/io.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <iomanip>
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

bool IsBlocking(std::string_view name)
{
    return name == "calibration" || name == "poses" || name == "semantics";
}

bool IsNonBlocking(std::string_view name)
{
    return name == "images" || name == "lidar";
}

nlohmann::json ListSequenceDirectories(const fs::path &directory)
{
    std::vector<std::string> names;
    std::error_code error;
    fs::directory_iterator iterator(directory, fs::directory_options::skip_permission_denied,
                                    error);
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
            std::error_code directoryError;
            if (iterator->is_directory(directoryError))
            {
                names.push_back(iterator->path().filename().string());
            }
        }
    }
    std::sort(names.begin(), names.end());
    return names;
}

nlohmann::json ComponentJson(const ComponentStatus &component)
{
    return nlohmann::json{
        {"name", component.name},
        {"required", component.required},
        {"present", component.present},
        {"missing", component.missing},
        {"rejected", component.rejected},
        {"file_count", component.fileCount},
        {"bytes", component.bytes},
        {"notes", component.notes},
        {"ok", component.Ok()},
    };
}

const ComponentStatus *FindComponent(const std::vector<ComponentStatus> &components,
                                     std::string_view name)
{
    const auto found =
        std::find_if(components.begin(), components.end(),
                     [name](const ComponentStatus &component) { return component.name == name; });
    return found == components.end() ? nullptr : &*found;
}

std::string ComponentStatusLabel(const ComponentStatus &component)
{
    if (!component.rejected.empty())
    {
        return IsBlocking(component.name) ? "BLOCKING REJECTED" : "NON-BLOCKING REJECTED";
    }
    if (!component.missing.empty())
    {
        return IsBlocking(component.name) ? "BLOCKING MISSING" : "NON-BLOCKING MISSING";
    }
    if (component.fileCount == 0U)
    {
        return "EMPTY";
    }
    return "OK";
}

} // namespace

PreflightReport RunPreflight(const std::filesystem::path &root, std::string_view sequence,
                             const std::vector<std::string> &cameras,
                             const std::vector<std::string> &forbiddenNames,
                             const std::vector<std::filesystem::path> &scanRoots)
{
    PreflightReport report;
    report.root = root.string();
    report.sequence = std::string(sequence);

    const DatasetLayout layout = DetectLayout(root, sequence);
    report.layout = std::string(LayoutKindName(layout.kind));
    report.components = ComponentInventory(root, sequence, cameras);

    // Apply any forbidden names supplied by configuration that were not already
    // found by the recursive scan (e.g. a differently-named KITTI-raw file that
    // sits directly in calibration/).
    auto calibrationIt =
        std::find_if(report.components.begin(), report.components.end(),
                     [](const ComponentStatus &component)
                     { return component.name == "calibration"; });
    if (calibrationIt != report.components.end())
    {
        ComponentStatus &calibration = *calibrationIt;
        for (const std::string &name : forbiddenNames)
        {
            const fs::path path = root / "calibration" / name;
            std::error_code error;
            if (!fs::is_regular_file(path, error))
            {
                continue;
            }
            std::error_code relativeError;
            const fs::path relative = fs::relative(path, root, relativeError);
            const std::string entry =
                relativeError ? path.lexically_normal().generic_string()
                              : relative.generic_string();
            if (std::find(calibration.rejected.begin(), calibration.rejected.end(), entry) ==
                calibration.rejected.end())
            {
                calibration.rejected.push_back(entry);
            }
        }
    }

    report.inventory = nlohmann::json{
        {"data_2d_raw", ListSequenceDirectories(root / "data_2d_raw")},
        {"data_poses", ListSequenceDirectories(root / "data_poses")},
        {"data_3d_raw", ListSequenceDirectories(root / "data_3d_raw")},
    };

    report.externalScans = nlohmann::json::object();
    for (const fs::path &scanRoot : scanRoots)
    {
        report.externalScans[scanRoot.string()] = FindRejectedCalibration(scanRoot);
    }

    report.ok = true;
    for (const ComponentStatus &component : report.components)
    {
        if (IsBlocking(component.name))
        {
            report.blockingMissing.insert(report.blockingMissing.end(), component.missing.begin(),
                                          component.missing.end());
        }
        else
        {
            report.nonblockingMissing.insert(report.nonblockingMissing.end(),
                                             component.missing.begin(), component.missing.end());
        }

        if (IsNonBlocking(component.name) && !component.rejected.empty())
        {
            report.nonblockingRejected.insert(report.nonblockingRejected.end(),
                                              component.rejected.begin(),
                                              component.rejected.end());
        }
        report.rejected.insert(report.rejected.end(), component.rejected.begin(),
                               component.rejected.end());

        if (!component.Ok() || component.fileCount == 0U)
        {
            report.ok = false;
        }
    }

    if (report.components.empty())
    {
        report.ok = false;
    }

    report.generatedAtUtc = UtcNowIso8601();
    return report;
}

int PreflightExitCode(const PreflightReport &report, bool strict)
{
    std::error_code error;
    if (!fs::is_directory(fs::path(report.root), error))
    {
        return 4;
    }

    bool blockingBad = report.components.empty();
    bool nonBlockingBad = false;
    for (const ComponentStatus &component : report.components)
    {
        const bool bad = !component.Ok() || component.fileCount == 0U;
        if (IsBlocking(component.name))
        {
            blockingBad = blockingBad || bad;
        }
        else if (IsNonBlocking(component.name))
        {
            nonBlockingBad = nonBlockingBad || bad;
        }
        else
        {
            nonBlockingBad = nonBlockingBad || bad;
        }
    }

    if (blockingBad)
    {
        return 2;
    }
    if (nonBlockingBad)
    {
        return strict ? 2 : 3;
    }
    return 0;
}

std::string RenderSummary(const PreflightReport &report)
{
    std::ostringstream out;
    out << "KITTI-360 dataset preflight\n";
    out << "root: " << report.root << '\n';
    out << "sequence: " << report.sequence << '\n';
    out << "layout: " << report.layout << '\n';
    out << "generated_at_utc: " << report.generatedAtUtc << '\n';
    out << "ready: " << (report.ok ? "yes" : "no") << "\n\n";

    out << std::left << std::setw(12) << "COMPONENT" << std::setw(22) << "STATUS"
        << std::right << std::setw(10) << "FILES" << std::setw(18) << "BYTES"
        << std::setw(9) << "MISSING" << '\n';
    for (const ComponentStatus &component : report.components)
    {
        out << std::left << std::setw(12) << component.name << std::setw(22)
            << ComponentStatusLabel(component) << std::right << std::setw(10)
            << component.fileCount << std::setw(18) << component.bytes << std::setw(9)
            << component.missing.size() << '\n';
    }

    if (!report.blockingMissing.empty())
    {
        out << "\nBLOCKING MISSING:\n";
        for (const std::string &entry : report.blockingMissing)
        {
            out << "  " << entry << '\n';
        }
    }

    const ComponentStatus *calibration = FindComponent(report.components, "calibration");
    const bool hasKittiRaw = calibration != nullptr && !calibration->rejected.empty();
    const bool hasExternal =
        report.externalScans.is_object() && !report.externalScans.empty();
    const bool hasAnyRejected =
        !report.rejected.empty() || !report.nonblockingRejected.empty() || hasKittiRaw ||
        hasExternal;

    if (hasAnyRejected)
    {
        out << "\nREJECTED:\n";
        if (hasKittiRaw)
        {
            out << "  KITTI-raw calibration:\n";
            for (const std::string &entry : calibration->rejected)
            {
                out << "    " << entry << '\n';
            }
        }
        if (!report.nonblockingRejected.empty())
        {
            out << "  NON-BLOCKING REJECTED:\n";
            for (const std::string &entry : report.nonblockingRejected)
            {
                out << "    " << entry << '\n';
            }
        }
        if (hasExternal)
        {
            out << "  EXTERNAL SCAN REJECTIONS:\n";
            for (const auto &[scanRoot, entries] : report.externalScans.items())
            {
                out << "    " << scanRoot << ":\n";
                if (entries.empty())
                {
                    out << "      (none)\n";
                    continue;
                }
                for (const auto &entry : entries)
                {
                    out << "      " << entry.get<std::string>() << '\n';
                }
            }
        }
    }

    return out.str();
}

nlohmann::json PreflightReportJson(const PreflightReport &report)
{
    nlohmann::json components = nlohmann::json::array();
    for (const ComponentStatus &component : report.components)
    {
        components.push_back(ComponentJson(component));
    }

    return nlohmann::json{
        {"schema_version", report.schemaVersion},
        {"root", report.root},
        {"sequence", report.sequence},
        {"layout", report.layout},
        {"components", std::move(components)},
        {"inventory", report.inventory},
        {"external_scans", report.externalScans},
        {"ok", report.ok},
        {"blocking_missing", report.blockingMissing},
        {"nonblocking_missing", report.nonblockingMissing},
        {"nonblocking_rejected", report.nonblockingRejected},
        {"rejected", report.rejected},
        {"generated_at_utc", report.generatedAtUtc},
    };
}
} // namespace raylidar::dataset
