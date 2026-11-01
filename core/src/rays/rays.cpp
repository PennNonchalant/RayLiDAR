#include <raylidar/rays/rays.h>

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
#include <system_error>
#include <vector>

namespace raylidar::rays
{
namespace
{
namespace fs = std::filesystem;

constexpr char kMagic[4] = {'R', 'L', 'R', 'Y'};
constexpr float kDirectionTolerance = 1.0e-3F;

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

void PutU64(std::uint8_t *bytes, const std::uint64_t value)
{
    for (int index = 0; index < 8; ++index)
    {
        bytes[index] = static_cast<std::uint8_t>((value >> (8U * static_cast<unsigned>(index))) &
                                                 0xFFU);
    }
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

std::uint64_t GetU64(const std::uint8_t *bytes)
{
    std::uint64_t value = 0;
    for (unsigned index = 0; index < 8U; ++index)
    {
        value |= static_cast<std::uint64_t>(bytes[index]) << (8U * index);
    }
    return value;
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

/// Write a byte buffer through a sibling temporary and rename, so a crashed or
/// interrupted build never leaves a partially written artifact (T-02-18).
void WriteBytesAtomic(const fs::path &path, const std::uint8_t *data, const std::size_t size)
{
    if (!path.parent_path().empty())
    {
        EnsureDir(path.parent_path());
    }
    fs::path temporary = path;
    temporary += ".tmp";

    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output.good())
        {
            Reject("cannot write ray artifact " + path.string());
        }
        if (size != 0U)
        {
            output.write(reinterpret_cast<const char *>(data),
                         static_cast<std::streamsize>(size));
        }
        if (!output.good())
        {
            std::error_code ignored;
            fs::remove(temporary, ignored);
            Reject("error while writing ray artifact " + path.string());
        }
    }

    std::error_code error;
    fs::rename(temporary, path, error);
    if (error)
    {
        std::error_code ignored;
        fs::remove(temporary, ignored);
        Reject("cannot replace ray artifact " + path.string() + ": " + error.message());
    }
}
} // namespace

const char *BucketName(const Bucket bucket)
{
    switch (bucket)
    {
    case Bucket::B0_20:
        return "0_20";
    case Bucket::B20_40:
        return "20_40";
    case Bucket::B40_50:
        return "40_50";
    case Bucket::B50_60:
        return "50_60";
    }
    return "unknown";
}

Bucket BucketOf(const double r_gt)
{
    if (!std::isfinite(r_gt) || r_gt < 0.0 || r_gt > kRayMaxRangeM)
    {
        throw std::invalid_argument("ray range outside [0, 60] m cannot be bucketed");
    }
    if (r_gt < 20.0)
    {
        return Bucket::B0_20;
    }
    if (r_gt < 40.0)
    {
        return Bucket::B20_40;
    }
    if (r_gt <= 50.0)
    {
        return Bucket::B40_50;
    }
    return Bucket::B50_60;
}


void WriteRayFrame(const fs::path &path, const int frame, const std::vector<Ray> &rays)
{
    if (frame < 0)
    {
        throw std::invalid_argument("ray frame id must be non-negative");
    }
    if (rays.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
    {
        throw std::invalid_argument("a frame cannot hold more than 2^32-1 rays");
    }

    std::array<std::uint8_t, kRayHeaderSize> header{};
    std::memcpy(header.data(), kMagic, sizeof(kMagic));
    PutU16(header.data() + 4, kRayVersion);
    PutU16(header.data() + 6, kRayRecordSize);
    PutU32(header.data() + 8, static_cast<std::uint32_t>(rays.size()));
    PutU32(header.data() + 12, static_cast<std::uint32_t>(frame));

    std::vector<std::uint8_t> body(rays.size() * kRayRecordSize, 0U);
    for (std::size_t index = 0; index < rays.size(); ++index)
    {
        const Ray &ray = rays[index];
        if (!ray.p_L.allFinite() || !ray.p_C.allFinite() || !ray.dir_C.allFinite() ||
            !std::isfinite(ray.t_gt) || ray.t_gt <= 0.0F)
        {
            throw std::invalid_argument("refusing to write a non-finite ray");
        }
        if (ray.bucket > static_cast<std::uint8_t>(Bucket::B50_60))
        {
            throw std::invalid_argument("refusing to write an out-of-range ray bucket");
        }
        if (ray.camera > 1U)
        {
            throw std::invalid_argument("refusing to write an unknown ray camera");
        }

        std::uint8_t *record = body.data() + index * kRayRecordSize;
        PutU64(record + 0, ray.ray_id);
        PutF32(record + 8, ray.dir_C.x());
        PutF32(record + 12, ray.dir_C.y());
        PutF32(record + 16, ray.dir_C.z());
        PutF32(record + 20, ray.t_gt);
        PutF32(record + 24, ray.p_L.x());
        PutF32(record + 28, ray.p_L.y());
        PutF32(record + 32, ray.p_L.z());
        PutF32(record + 36, ray.p_C.x());
        PutF32(record + 40, ray.p_C.y());
        PutF32(record + 44, ray.p_C.z());
        record[48] = ray.bucket;
        record[49] = ray.occluded;
        record[50] = ray.camera;
        PutU16(record + 51, ray.semanticId);
        // record[53..63] stays zero padding.
    }

    std::vector<std::uint8_t> bytes;
    bytes.reserve(header.size() + body.size());
    bytes.insert(bytes.end(), header.begin(), header.end());
    bytes.insert(bytes.end(), body.begin(), body.end());
    WriteBytesAtomic(path, bytes.data(), bytes.size());
}

std::vector<Ray> ReadRayFrame(const fs::path &path, int *frame)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input.good())
    {
        Reject("cannot read ray frame " + path.string());
    }
    const std::streamoff end = input.tellg();
    if (end < static_cast<std::streamoff>(kRayHeaderSize))
    {
        Reject("ray frame is shorter than its header: " + path.string());
    }
    const std::uint64_t fileSize = static_cast<std::uint64_t>(end);
    input.seekg(0, std::ios::beg);

    std::array<std::uint8_t, kRayHeaderSize> header{};
    input.read(reinterpret_cast<char *>(header.data()),
               static_cast<std::streamsize>(header.size()));
    if (!input.good())
    {
        Reject("cannot read ray frame header " + path.string());
    }
    if (std::memcmp(header.data(), kMagic, sizeof(kMagic)) != 0)
    {
        Reject("ray frame has a wrong magic: " + path.string());
    }
    if (GetU16(header.data() + 4) != kRayVersion)
    {
        Reject("ray frame has an unsupported version: " + path.string());
    }
    if (GetU16(header.data() + 6) != kRayRecordSize)
    {
        Reject("ray frame has a wrong record size: " + path.string());
    }

    const std::uint32_t count = GetU32(header.data() + 8);
    const std::uint32_t headerFrame = GetU32(header.data() + 12);
    const std::uint64_t expected = static_cast<std::uint64_t>(kRayHeaderSize) +
                                   static_cast<std::uint64_t>(count) * kRayRecordSize;
    if (fileSize != expected)
    {
        Reject("ray frame length does not match its header count: " + path.string());
    }

    std::vector<std::uint8_t> body(static_cast<std::size_t>(count) * kRayRecordSize);
    if (!body.empty())
    {
        input.read(reinterpret_cast<char *>(body.data()),
                   static_cast<std::streamsize>(body.size()));
        if (!input.good())
        {
            Reject("cannot read ray frame records " + path.string());
        }
    }

    std::vector<Ray> rays;
    rays.reserve(count);
    for (std::uint32_t index = 0; index < count; ++index)
    {
        const std::uint8_t *record = body.data() + static_cast<std::size_t>(index) * kRayRecordSize;
        Ray ray;
        ray.ray_id = GetU64(record + 0);
        ray.dir_C = Eigen::Vector3f(GetF32(record + 8), GetF32(record + 12), GetF32(record + 16));
        ray.t_gt = GetF32(record + 20);
        ray.p_L = Eigen::Vector3f(GetF32(record + 24), GetF32(record + 28), GetF32(record + 32));
        ray.p_C = Eigen::Vector3f(GetF32(record + 36), GetF32(record + 40), GetF32(record + 44));
        ray.bucket = record[48];
        ray.occluded = record[49];
        ray.camera = record[50];
        ray.semanticId = GetU16(record + 51);

        if (!ray.p_L.allFinite() || !ray.p_C.allFinite() || !ray.dir_C.allFinite() ||
            !std::isfinite(ray.t_gt) || ray.t_gt <= 0.0F)
        {
            Reject("ray frame contains a non-finite record: " + path.string());
        }
        if (ray.bucket > static_cast<std::uint8_t>(Bucket::B50_60))
        {
            Reject("ray frame contains an out-of-range bucket: " + path.string());
        }
        if (ray.camera > 1U)
        {
            Reject("ray frame contains an unknown camera id: " + path.string());
        }
        if (std::abs(ray.dir_C.norm() - 1.0F) > kDirectionTolerance)
        {
            Reject("ray frame contains a non-unit direction: " + path.string());
        }
        Bucket expectedBucket = Bucket::B0_20;
        try
        {
            expectedBucket = BucketOf(static_cast<double>(ray.t_gt));
        }
        catch (const std::invalid_argument &)
        {
            Reject("ray frame contains a t_gt outside the frozen domain: " + path.string());
        }
        if (static_cast<std::uint8_t>(expectedBucket) != ray.bucket)
        {
            Reject("ray frame bucket disagrees with t_gt: " + path.string());
        }
        rays.push_back(ray);
    }

    if (frame != nullptr)
    {
        *frame = static_cast<int>(headerFrame);
    }
    return rays;
}

void WriteMaskFile(const fs::path &path, const std::vector<std::uint8_t> &mask)
{
    for (const std::uint8_t value : mask)
    {
        if (value > 1U)
        {
            throw std::invalid_argument("ray mask values must be 0 or 1");
        }
    }
    WriteBytesAtomic(path, mask.data(), mask.size());
}
} // namespace raylidar::rays
