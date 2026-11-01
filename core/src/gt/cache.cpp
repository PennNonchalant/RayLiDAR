#include <raylidar/gt/cache.h>

#include <raylidar/common/io.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace raylidar::gt
{
namespace
{
namespace fs = std::filesystem;

constexpr std::uint16_t kVersion = 1;
constexpr std::uint16_t kRecordSize = 32;
constexpr std::size_t kHeaderSize = 64;
constexpr char kMagic[4] = {'R', 'L', 'G', 'T'};

void PutU16(std::uint8_t *bytes, const std::uint16_t value)
{
    bytes[0] = static_cast<std::uint8_t>(value & 0xFFU);
    bytes[1] = static_cast<std::uint8_t>((value >> 8U) & 0xFFU);
}

void PutU32(std::uint8_t *bytes, const std::uint32_t value)
{
    bytes[0] = static_cast<std::uint8_t>(value & 0xFFU);
    bytes[1] = static_cast<std::uint8_t>((value >> 8U) & 0xFFU);
    bytes[2] = static_cast<std::uint8_t>((value >> 16U) & 0xFFU);
    bytes[3] = static_cast<std::uint8_t>((value >> 24U) & 0xFFU);
}

void PutF32(std::uint8_t *bytes, const float value)
{
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    PutU32(bytes, bits);
}

std::uint16_t GetU16(const std::uint8_t *bytes)
{
    return static_cast<std::uint16_t>(bytes[0]) |
           static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[1]) << 8U);
}

std::uint32_t GetU32(const std::uint8_t *bytes)
{
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8U) |
           (static_cast<std::uint32_t>(bytes[2]) << 16U) |
           (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

float GetF32(const std::uint8_t *bytes)
{
    const std::uint32_t bits = GetU32(bytes);
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

[[noreturn]] void Reject(const std::string &message)
{
    throw IoError(message, 4);
}
} // namespace

void WriteGtFrame(const fs::path &path, const int frame, const std::vector<GtPoint> &points)
{
    if (frame < 0)
    {
        throw std::invalid_argument("GT frame id must be non-negative");
    }
    if (points.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
    {
        throw std::invalid_argument("GT frame holds more points than the header can encode");
    }

    std::array<std::uint8_t, kHeaderSize> header{};
    std::memcpy(header.data(), kMagic, sizeof(kMagic));
    PutU16(header.data() + 4, kVersion);
    PutU16(header.data() + 6, kRecordSize);
    PutU32(header.data() + 8, static_cast<std::uint32_t>(points.size()));
    PutU32(header.data() + 12, static_cast<std::uint32_t>(frame));

    std::vector<std::uint8_t> body(points.size() * kRecordSize, 0U);
    for (std::size_t index = 0; index < points.size(); ++index)
    {
        const GtPoint &point = points[index];
        if (!point.p_L.allFinite() || !std::isfinite(point.r_gt))
        {
            throw std::invalid_argument("refusing to write a non-finite GT point");
        }
        std::uint8_t *record = body.data() + index * kRecordSize;
        PutF32(record + 0, point.p_L.x());
        PutF32(record + 4, point.p_L.y());
        PutF32(record + 8, point.p_L.z());
        PutF32(record + 12, point.r_gt);
        PutU32(record + 16, point.frame);
        PutU16(record + 20, point.semanticId);
        record[22] = point.camera;
        record[23] = point.valid;
        // record[24..31] stays zero padding.
    }

    if (!path.parent_path().empty())
    {
        EnsureDir(path.parent_path());
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output.good())
    {
        Reject("cannot write GT frame " + path.string());
    }
    output.write(reinterpret_cast<const char *>(header.data()),
                 static_cast<std::streamsize>(header.size()));
    if (!body.empty())
    {
        output.write(reinterpret_cast<const char *>(body.data()),
                     static_cast<std::streamsize>(body.size()));
    }
    if (!output.good())
    {
        Reject("error while writing GT frame " + path.string());
    }
}

std::vector<GtPoint> ReadGtFrame(const fs::path &path, int *frame)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input.good())
    {
        Reject("cannot read GT frame " + path.string());
    }
    const std::streamoff end = input.tellg();
    if (end < static_cast<std::streamoff>(kHeaderSize))
    {
        Reject("GT frame is shorter than its header: " + path.string());
    }
    const std::uint64_t fileSize = static_cast<std::uint64_t>(end);
    input.seekg(0, std::ios::beg);

    std::array<std::uint8_t, kHeaderSize> header{};
    input.read(reinterpret_cast<char *>(header.data()),
               static_cast<std::streamsize>(header.size()));
    if (!input.good())
    {
        Reject("cannot read GT frame header " + path.string());
    }
    if (std::memcmp(header.data(), kMagic, sizeof(kMagic)) != 0)
    {
        Reject("GT frame has a wrong magic: " + path.string());
    }
    if (GetU16(header.data() + 4) != kVersion)
    {
        Reject("GT frame has an unsupported version: " + path.string());
    }
    if (GetU16(header.data() + 6) != kRecordSize)
    {
        Reject("GT frame has a wrong record size: " + path.string());
    }

    const std::uint32_t count = GetU32(header.data() + 8);
    const std::uint32_t headerFrame = GetU32(header.data() + 12);
    const std::uint64_t expected =
        static_cast<std::uint64_t>(kHeaderSize) + static_cast<std::uint64_t>(count) * kRecordSize;
    if (fileSize != expected)
    {
        Reject("GT frame length does not match its header count: " + path.string());
    }

    std::vector<std::uint8_t> body(static_cast<std::size_t>(count) * kRecordSize);
    if (!body.empty())
    {
        input.read(reinterpret_cast<char *>(body.data()),
                   static_cast<std::streamsize>(body.size()));
        if (!input.good())
        {
            Reject("cannot read GT frame records " + path.string());
        }
    }

    std::vector<GtPoint> points;
    points.reserve(count);
    for (std::uint32_t index = 0; index < count; ++index)
    {
        const std::uint8_t *record = body.data() + static_cast<std::size_t>(index) * kRecordSize;
        GtPoint point;
        point.p_L = Eigen::Vector3f(GetF32(record + 0), GetF32(record + 4), GetF32(record + 8));
        point.r_gt = GetF32(record + 12);
        point.frame = GetU32(record + 16);
        point.semanticId = GetU16(record + 20);
        point.camera = record[22];
        point.valid = record[23];

        if (!point.p_L.allFinite() || !std::isfinite(point.r_gt))
        {
            Reject("GT frame contains a non-finite record: " + path.string());
        }
        if (point.frame != headerFrame)
        {
            Reject("GT frame record id disagrees with the header: " + path.string());
        }
        points.push_back(point);
    }

    if (frame != nullptr)
    {
        *frame = static_cast<int>(headerFrame);
    }
    return points;
}
} // namespace raylidar::gt
