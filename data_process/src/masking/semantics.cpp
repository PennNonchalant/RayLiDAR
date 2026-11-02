#include <raylidar/masking/semantics.h>

#include <raylidar/common/io.h>

#include <stb_image.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace raylidar::masking
{
namespace
{
namespace fs = std::filesystem;

constexpr int kExitInvalidInput = 4;
constexpr std::uint16_t kUnknownLabel = 65535U;

struct ImageDeleter
{
    void operator()(unsigned char *data) const noexcept
    {
        stbi_image_free(data);
    }
    void operator()(stbi_us *data) const noexcept
    {
        stbi_image_free(data);
    }
};

[[nodiscard]] std::vector<std::uint16_t> Widen(const unsigned char *data, const std::size_t count)
{
    std::vector<std::uint16_t> labels(count);
    for (std::size_t index = 0; index < count; ++index)
    {
        labels[index] = static_cast<std::uint16_t>(data[index]);
    }
    return labels;
}
} // namespace

SemanticImage LoadSemantics(const fs::path &pngPath, const std::size_t maxPixels)
{
    if (maxPixels == 0U)
    {
        throw IoError("semantic pixel cap must be positive", kExitInvalidInput);
    }

    int width = 0;
    int height = 0;
    int channels = 0;
    if (stbi_info(pngPath.string().c_str(), &width, &height, &channels) == 0)
    {
        throw IoError("cannot read semantic image " + pngPath.string(), kExitInvalidInput);
    }
    if (width <= 0 || height <= 0)
    {
        throw IoError("semantic image " + pngPath.string() + " has invalid dimensions", kExitInvalidInput);
    }
    if (channels != 1)
    {
        throw IoError("semantic image " + pngPath.string() + " must be one channel, found " + std::to_string(channels),
                      kExitInvalidInput);
    }
    const std::size_t pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (pixels > maxPixels)
    {
        throw IoError("semantic image " + pngPath.string() + " exceeds the pixel cap of " + std::to_string(maxPixels),
                      kExitInvalidInput);
    }

    SemanticImage image;
    image.width = width;
    image.height = height;

    if (stbi_is_16_bit(pngPath.string().c_str()) != 0)
    {
        int decodedWidth = 0;
        int decodedHeight = 0;
        int decodedChannels = 0;
        std::unique_ptr<stbi_us, ImageDeleter> data(
            stbi_load_16(pngPath.string().c_str(), &decodedWidth, &decodedHeight, &decodedChannels, 1));
        if (data == nullptr || decodedWidth != width || decodedHeight != height || decodedChannels != 1)
        {
            throw IoError("cannot decode 16-bit semantic image " + pngPath.string(), kExitInvalidInput);
        }
        image.labels.assign(data.get(), data.get() + pixels);
        return image;
    }

    int decodedWidth = 0;
    int decodedHeight = 0;
    int decodedChannels = 0;
    std::unique_ptr<unsigned char, ImageDeleter> data(
        stbi_load(pngPath.string().c_str(), &decodedWidth, &decodedHeight, &decodedChannels, 1));
    if (data == nullptr || decodedWidth != width || decodedHeight != height || decodedChannels != 1)
    {
        throw IoError("cannot decode semantic image " + pngPath.string(), kExitInvalidInput);
    }
    image.labels = Widen(data.get(), pixels);
    return image;
}

bool IsDynamicLabel(const std::uint16_t labelId)
{
    switch (labelId)
    {
    case 24U: // person
    case 25U: // rider
    case 26U: // car
    case 27U: // truck
    case 28U: // bus
    case 32U: // motorcycle
    case 33U: // bicycle
        return true;
    default:
        return false;
    }
}

std::uint16_t LabelAt(const SemanticImage &image, const double u, const double v)
{
    if (!std::isfinite(u) || !std::isfinite(v) || image.width <= 0 || image.height <= 0)
    {
        return kUnknownLabel;
    }
    const double shiftedU = u + 0.5;
    const double shiftedV = v + 0.5;
    if (shiftedU < 0.0 || shiftedV < 0.0)
    {
        return kUnknownLabel;
    }
    const auto column = static_cast<long long>(std::floor(shiftedU));
    const auto row = static_cast<long long>(std::floor(shiftedV));
    if (column >= image.width || row >= image.height)
    {
        return kUnknownLabel;
    }
    const std::size_t index =
        static_cast<std::size_t>(row) * static_cast<std::size_t>(image.width) + static_cast<std::size_t>(column);
    if (index >= image.labels.size())
    {
        return kUnknownLabel;
    }
    return image.labels[index];
}
} // namespace raylidar::masking
