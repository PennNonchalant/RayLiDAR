#include <raylidar/common/artifacts.h>

#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>

namespace raylidar
{
namespace
{
constexpr const char *kDefaultArtifactsRoot = "/data-20T/hybrid-lidar-sim/artifacts";
} // namespace

std::filesystem::path ArtifactsRoot()
{
    if (const char *environment = std::getenv("RAYLIDAR_ARTIFACTS_ROOT");
        environment != nullptr && *environment != '\0')
    {
        return std::filesystem::path(environment);
    }
    return std::filesystem::path(kDefaultArtifactsRoot);
}

std::filesystem::path PhaseArtifactsDir(int phase)
{
    std::ostringstream name;
    name << std::setw(2) << std::setfill('0') << phase;
    return ArtifactsRoot() / name.str();
}

std::filesystem::path PhaseArtifact(int phase, std::string_view relative)
{
    return PhaseArtifactsDir(phase) / std::filesystem::path(std::string(relative));
}
} // namespace raylidar
