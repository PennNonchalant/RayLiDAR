#include <raylidar/masking/bbox.h>

#include <raylidar/common/io.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace raylidar::masking
{
namespace
{
namespace fs = std::filesystem;

constexpr int kExitInvalidInput = 4;
constexpr std::uintmax_t kMaxXmlBytes = 512ULL * 1024ULL * 1024ULL;
constexpr double kAabbEpsilon = 1e-9;
constexpr double kCubeEpsilon = 1e-12;

[[nodiscard]] std::string ReadWholeFile(const fs::path &path)
{
    std::error_code error;
    const std::uintmax_t size = fs::file_size(path, error);
    if (error)
    {
        throw IoError("cannot read bbox XML " + path.string() + ": " + error.message(), kExitInvalidInput);
    }
    if (size == 0U || size > kMaxXmlBytes)
    {
        throw IoError("bbox XML " + path.string() + " has an invalid size (" + std::to_string(size) + " bytes)",
                      kExitInvalidInput);
    }
    std::ifstream input(path, std::ios::binary);
    if (!input.good())
    {
        throw IoError("cannot open bbox XML " + path.string(), kExitInvalidInput);
    }
    std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (!input.good() && !input.eof())
    {
        throw IoError("error while reading bbox XML " + path.string(), kExitInvalidInput);
    }
    return text;
}

[[nodiscard]] std::string ToUpperAscii(std::string_view text)
{
    std::string upper(text);
    std::transform(upper.begin(), upper.end(), upper.begin(),
                   [](const unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
    return upper;
}

void RejectUnsafeXml(const std::string &text, const fs::path &path)
{
    const std::string upper = ToUpperAscii(text);
    if (upper.find("<!DOCTYPE") != std::string::npos || upper.find("<!ENTITY") != std::string::npos)
    {
        throw IoError("bbox XML " + path.string() + " contains a document type or entity declaration",
                      kExitInvalidInput);
    }
}

[[nodiscard]] std::string TagValue(const std::string &block, const std::string &tag)
{
    const std::string open = "<" + tag + ">";
    const std::string close = "</" + tag + ">";
    const std::size_t begin = block.find(open);
    if (begin == std::string::npos)
    {
        throw IoError("bbox object is missing <" + tag + ">", kExitInvalidInput);
    }
    const std::size_t end = block.find(close, begin + open.size());
    if (end == std::string::npos)
    {
        throw IoError("bbox object has an unterminated <" + tag + ">", kExitInvalidInput);
    }
    return block.substr(begin + open.size(), end - begin - open.size());
}

[[nodiscard]] int ParseIntField(const std::string &block, const std::string &tag)
{
    const std::string value = TagValue(block, tag);
    std::istringstream stream(value);
    int parsed = 0;
    if (!(stream >> parsed))
    {
        throw IoError("bbox field <" + tag + "> is not an integer", kExitInvalidInput);
    }
    stream >> std::ws;
    if (!stream.eof())
    {
        throw IoError("bbox field <" + tag + "> has trailing characters", kExitInvalidInput);
    }
    return parsed;
}

[[nodiscard]] bool ParseBoolField(const std::string &block, const std::string &tag)
{
    const int value = ParseIntField(block, tag);
    if (value != 0 && value != 1)
    {
        throw IoError("bbox field <" + tag + "> must be 0 or 1", kExitInvalidInput);
    }
    return value == 1;
}

[[nodiscard]] Eigen::Matrix4d ParseTransform(const std::string &block)
{
    const std::size_t transformBegin = block.find("<transform");
    if (transformBegin == std::string::npos)
    {
        throw IoError("bbox object is missing <transform>", kExitInvalidInput);
    }
    const std::size_t transformEnd = block.find("</transform>", transformBegin);
    if (transformEnd == std::string::npos)
    {
        throw IoError("bbox object has an unterminated <transform>", kExitInvalidInput);
    }
    const std::string transform = block.substr(transformBegin, transformEnd - transformBegin);

    const std::size_t dataBegin = transform.find("<data>");
    if (dataBegin == std::string::npos)
    {
        throw IoError("bbox transform is missing <data>", kExitInvalidInput);
    }
    const std::size_t dataEnd = transform.find("</data>", dataBegin);
    if (dataEnd == std::string::npos)
    {
        throw IoError("bbox transform has an unterminated <data>", kExitInvalidInput);
    }
    const std::string data = transform.substr(dataBegin + 6U, dataEnd - dataBegin - 6U);

    std::istringstream stream(data);
    std::vector<double> values;
    values.reserve(16U);
    while (true)
    {
        double value = 0.0;
        if (!(stream >> value))
        {
            break;
        }
        values.push_back(value);
    }
    stream.clear();
    stream >> std::ws;
    if (values.size() != 16U || !stream.eof())
    {
        throw IoError("bbox transform must contain exactly 16 values", kExitInvalidInput);
    }

    Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            matrix(row, col) = values[static_cast<std::size_t>(row * 4 + col)];
        }
    }
    if (!matrix.allFinite())
    {
        throw IoError("bbox transform contains a non-finite value", kExitInvalidInput);
    }
    return matrix;
}

void FillDerived(const Eigen::Matrix4d &worldFromBox, DynamicBBox *box)
{
    Eigen::Vector3d minimum = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
    Eigen::Vector3d maximum = Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());
    for (int xSign = -1; xSign <= 1; xSign += 2)
    {
        for (int ySign = -1; ySign <= 1; ySign += 2)
        {
            for (int zSign = -1; zSign <= 1; zSign += 2)
            {
                const Eigen::Vector4d corner(0.5 * static_cast<double>(xSign), 0.5 * static_cast<double>(ySign),
                                             0.5 * static_cast<double>(zSign), 1.0);
                const Eigen::Vector3d world = (worldFromBox * corner).head<3>();
                minimum = minimum.cwiseMin(world);
                maximum = maximum.cwiseMax(world);
            }
        }
    }
    box->aabbMin = minimum;
    box->aabbMax = maximum;
    box->T_B_W = worldFromBox.inverse();
    if (!box->T_B_W.allFinite() || !box->aabbMin.allFinite() || !box->aabbMax.allFinite())
    {
        throw IoError("bbox transform is not invertible", kExitInvalidInput);
    }
}
} // namespace

std::vector<DynamicBBox> LoadDynamicBBoxes(const fs::path &xmlPath, const std::size_t maxObjects)
{
    if (maxObjects == 0U)
    {
        throw IoError("bbox object cap must be positive", kExitInvalidInput);
    }
    const std::string text = ReadWholeFile(xmlPath);
    RejectUnsafeXml(text, xmlPath);

    std::vector<DynamicBBox> boxes;
    std::size_t position = 0;
    while ((position = text.find("<object", position)) != std::string::npos)
    {
        const std::size_t nameBegin = position + 7U; // skip "<object"
        std::size_t nameEnd = nameBegin;
        while (nameEnd < text.size() && std::isdigit(static_cast<unsigned char>(text[nameEnd])) != 0)
        {
            ++nameEnd;
        }
        if (nameEnd == nameBegin)
        {
            position = nameBegin;
            continue;
        }
        if (nameEnd >= text.size() || text[nameEnd] != '>')
        {
            throw IoError("malformed bbox object tag in " + xmlPath.string(), kExitInvalidInput);
        }

        const std::string tag = text.substr(position + 1U, nameEnd - position - 1U);
        const std::string closeTag = "</" + tag + ">";
        const std::size_t closeBegin = text.find(closeTag, nameEnd);
        if (closeBegin == std::string::npos)
        {
            throw IoError("truncated bbox object <" + tag + "> in " + xmlPath.string(), kExitInvalidInput);
        }

        const std::string block = text.substr(nameEnd + 1U, closeBegin - nameEnd - 1U);
        DynamicBBox box;
        box.label = TagValue(block, "label");
        box.semanticId = ParseIntField(block, "semanticId");
        box.timestamp = ParseIntField(block, "timestamp");
        box.dynamic = ParseBoolField(block, "dynamic");
        box.startFrame = ParseIntField(block, "start_frame");
        box.endFrame = ParseIntField(block, "end_frame");
        box.T_W_B = ParseTransform(block);
        FillDerived(box.T_W_B, &box);

        boxes.push_back(std::move(box));
        if (boxes.size() > maxObjects)
        {
            throw IoError("bbox XML " + xmlPath.string() + " exceeds the object cap of " + std::to_string(maxObjects),
                          kExitInvalidInput);
        }
        position = closeBegin + closeTag.size();
    }
    return boxes;
}

bool PointInBBox(const Eigen::Vector3d &p_W, const DynamicBBox &box)
{
    if (!p_W.allFinite())
    {
        return false;
    }
    if ((p_W.array() < (box.aabbMin.array() - kAabbEpsilon)).any() ||
        (p_W.array() > (box.aabbMax.array() + kAabbEpsilon)).any())
    {
        return false;
    }
    const Eigen::Vector4d local = box.T_B_W * Eigen::Vector4d(p_W.x(), p_W.y(), p_W.z(), 1.0);
    return std::abs(local.x()) <= (0.5 + kCubeEpsilon) && std::abs(local.y()) <= (0.5 + kCubeEpsilon) &&
           std::abs(local.z()) <= (0.5 + kCubeEpsilon);
}

std::vector<const DynamicBBox *> ApplicableBoxes(const std::vector<DynamicBBox> &boxes, const int frame,
                                                 const int maxTimeGapFrames)
{
    std::vector<const DynamicBBox *> applicable;
    for (const DynamicBBox &box : boxes)
    {
        if (!box.dynamic)
        {
            continue;
        }
        if (frame < box.startFrame || frame > box.endFrame)
        {
            continue;
        }
        if (std::abs(box.timestamp - frame) > maxTimeGapFrames)
        {
            continue;
        }
        applicable.push_back(&box);
    }
    return applicable;
}
} // namespace raylidar::masking
