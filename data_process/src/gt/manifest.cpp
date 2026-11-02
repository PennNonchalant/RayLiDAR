#include <raylidar/gt/manifest.h>

#include <raylidar/common/io.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace raylidar::gt
{
namespace fs = std::filesystem;

nlohmann::json BuildGtManifest(const fs::path &phaseDir, const std::string_view sequence,
                               const std::vector<nlohmann::json> &frames, const std::string_view manifestFileName)
{
    std::string configSha256;
    nlohmann::json cleanFrames = nlohmann::json::array();
    std::uint64_t totalPoints = 0;
    std::uint64_t camera00 = 0;
    std::uint64_t camera01 = 0;
    std::uint64_t cameraUnknown = 0;

    for (std::size_t index = 0; index < frames.size(); ++index)
    {
        nlohmann::json entry = frames[index];
        if (index == 0U && entry.contains("config_sha256") && entry["config_sha256"].is_string())
        {
            configSha256 = entry["config_sha256"].get<std::string>();
        }
        entry.erase("config_sha256");

        totalPoints += entry.value("count", std::uint64_t{0});
        camera00 += entry.value("camera_00", std::uint64_t{0});
        camera01 += entry.value("camera_01", std::uint64_t{0});
        cameraUnknown += entry.value("camera_unknown", std::uint64_t{0});
        cleanFrames.push_back(std::move(entry));
    }

    nlohmann::json totals;
    totals["frames"] = cleanFrames.size();
    totals["points"] = totalPoints;
    totals["camera_00"] = camera00;
    totals["camera_01"] = camera01;
    totals["camera_unknown"] = cameraUnknown;

    nlohmann::json document;
    document["schema_version"] = 1;
    document["sequence"] = sequence;
    document["generated_at_utc"] = UtcNowIso8601();
    document["frames"] = std::move(cleanFrames);
    document["totals"] = std::move(totals);
    document["config_sha256"] = configSha256;

    const fs::path directory = phaseDir / "gt";
    EnsureDir(directory);
    WriteJson(directory / std::string(manifestFileName), document);
    return document;
}
} // namespace raylidar::gt
