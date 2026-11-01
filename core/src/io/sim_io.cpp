#include <raylidar/io/sim_io.h>

#include <raylidar/common/io.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>
#include <unordered_set>
#include <vector>

namespace raylidar::io
{
namespace
{
namespace fs = std::filesystem;

constexpr char kGtRaysMagic[] = "RLGR";
constexpr char kHitsMagic[] = "RLHT";
constexpr double kDirectionTolerance = 1.0e-6;

[[noreturn]] void Reject(const std::string &message)
{
    throw IoError(message, 4);
}

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
        bytes[index] =
            static_cast<std::uint8_t>((value >> (8U * static_cast<unsigned>(index))) & 0xFFU);
    }
}

void PutF64(std::uint8_t *bytes, const double value)
{
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    PutU64(bytes, bits);
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

double GetF64(const std::uint8_t *bytes)
{
    const std::uint64_t bits = GetU64(bytes);
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

/// Write a byte buffer through a sibling temporary and rename, so an
/// interrupted build never leaves a partially written artifact.
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
            Reject("cannot write simulation I/O file " + path.string());
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
            Reject("error while writing simulation I/O file " + path.string());
        }
    }

    std::error_code error;
    fs::rename(temporary, path, error);
    if (error)
    {
        std::error_code ignored;
        fs::remove(temporary, ignored);
        Reject("cannot replace simulation I/O file " + path.string() + ": " + error.message());
    }
}

void ValidateGtRayValues(const GtRay &ray, const std::string &path)
{
    if (!ray.origin_W.allFinite() || !ray.dir_W.allFinite() || !std::isfinite(ray.t_gt) ||
        !std::isfinite(ray.t_min) || !std::isfinite(ray.t_max))
    {
        Reject("gt_rays.bin contains a non-finite value: " + path);
    }
    if (!(ray.t_min > 0.0))
    {
        Reject("gt_rays.bin contains t_min <= 0: " + path);
    }
    if (!(ray.t_min <= ray.t_gt && ray.t_gt <= ray.t_max))
    {
        Reject("gt_rays.bin contains t_gt outside [t_min, t_max]: " + path);
    }
    if (std::abs(ray.dir_W.norm() - 1.0) > kDirectionTolerance)
    {
        Reject("gt_rays.bin contains a non-unit dir_W: " + path);
    }
    if (ray.camera > 1U)
    {
        Reject("gt_rays.bin contains an unknown camera: " + path);
    }
}

void ValidateHitValues(const Hit &hit, const std::string &path)
{
    if (hit.hit)
    {
        if (!std::isfinite(hit.t_pred) || hit.t_pred <= 0.0)
        {
            Reject("hits.bin contains a hit with an invalid t_pred: " + path);
        }
        return;
    }
    if (!std::isinf(hit.t_pred) || hit.t_pred <= 0.0)
    {
        Reject("hits.bin contains a miss with a finite t_pred: " + path);
    }
    if (hit.prim_id != kPrimIdUnavailable)
    {
        Reject("hits.bin contains a miss without the 0xFFFFFFFF prim_id sentinel: " + path);
    }
}
} // namespace

void WriteGtRays(const fs::path &path, const std::span<const GtRay> rays)
{
    if (rays.size() > static_cast<std::size_t>(kMaxRecordCount))
    {
        Reject("gt_rays.bin count exceeds the record cap: " + path.string());
    }

    std::array<std::uint8_t, kSimIoHeaderSize> header{};
    std::memcpy(header.data(), kGtRaysMagic, sizeof(kGtRaysMagic) - 1U);
    PutU16(header.data() + 4, kGtRaysVersion);
    PutU16(header.data() + 6, kGtRaysRecordSize);
    PutU32(header.data() + 8, static_cast<std::uint32_t>(rays.size()));

    std::unordered_set<std::uint64_t> ids;
    std::vector<std::uint8_t> body(rays.size() * kGtRaysRecordSize, 0U);
    for (std::size_t index = 0; index < rays.size(); ++index)
    {
        const GtRay &ray = rays[index];
        ValidateGtRayValues(ray, path.string());
        if (!ids.insert(ray.ray_id).second)
        {
            Reject("gt_rays.bin contains a duplicate ray_id: " + path.string());
        }

        std::uint8_t *record = body.data() + index * kGtRaysRecordSize;
        PutU64(record + 0, ray.ray_id);
        PutF64(record + 8, ray.origin_W.x());
        PutF64(record + 16, ray.origin_W.y());
        PutF64(record + 24, ray.origin_W.z());
        PutF64(record + 32, ray.dir_W.x());
        PutF64(record + 40, ray.dir_W.y());
        PutF64(record + 48, ray.dir_W.z());
        PutF64(record + 56, ray.t_gt);
        PutF64(record + 64, ray.t_min);
        PutF64(record + 72, ray.t_max);
        PutU32(record + 80, ray.frame);
        PutU16(record + 84, ray.camera);
        // record[86..95] stays zero reserved padding.
    }

    std::vector<std::uint8_t> bytes;
    bytes.reserve(header.size() + body.size());
    bytes.insert(bytes.end(), header.begin(), header.end());
    bytes.insert(bytes.end(), body.begin(), body.end());
    WriteBytesAtomic(path, bytes.data(), bytes.size());
}

std::vector<GtRay> ReadGtRays(const fs::path &path, GtRaysInfo *info)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input.good())
    {
        Reject("cannot read gt_rays.bin " + path.string());
    }
    const std::streamoff end = input.tellg();
    if (end < static_cast<std::streamoff>(kSimIoHeaderSize))
    {
        Reject("gt_rays.bin is shorter than its header: " + path.string());
    }
    const std::uint64_t fileSize = static_cast<std::uint64_t>(end);
    input.seekg(0, std::ios::beg);

    std::array<std::uint8_t, kSimIoHeaderSize> header{};
    input.read(reinterpret_cast<char *>(header.data()),
               static_cast<std::streamsize>(header.size()));
    if (!input.good())
    {
        Reject("cannot read gt_rays.bin header " + path.string());
    }
    if (std::memcmp(header.data(), kGtRaysMagic, sizeof(kGtRaysMagic) - 1U) != 0)
    {
        Reject("gt_rays.bin has a wrong magic: " + path.string());
    }
    const std::uint16_t version = GetU16(header.data() + 4);
    if (version != kGtRaysVersion)
    {
        Reject("gt_rays.bin has an unsupported version: " + path.string());
    }
    const std::uint16_t recordSize = GetU16(header.data() + 6);
    if (recordSize != kGtRaysRecordSize)
    {
        Reject("gt_rays.bin has a wrong record size: " + path.string());
    }
    const std::uint32_t count = GetU32(header.data() + 8);
    if (count > kMaxRecordCount)
    {
        Reject("gt_rays.bin count exceeds the record cap: " + path.string());
    }
    const std::uint64_t expected =
        static_cast<std::uint64_t>(kSimIoHeaderSize) +
        static_cast<std::uint64_t>(count) * static_cast<std::uint64_t>(kGtRaysRecordSize);
    if (fileSize != expected)
    {
        Reject("gt_rays.bin length does not match its header count: " + path.string());
    }

    std::vector<std::uint8_t> body(static_cast<std::size_t>(count) * kGtRaysRecordSize);
    if (!body.empty())
    {
        input.read(reinterpret_cast<char *>(body.data()),
                   static_cast<std::streamsize>(body.size()));
        if (!input.good())
        {
            Reject("cannot read gt_rays.bin records " + path.string());
        }
    }

    std::vector<GtRay> rays;
    rays.reserve(count);
    std::unordered_set<std::uint64_t> ids;
    for (std::uint32_t index = 0; index < count; ++index)
    {
        const std::uint8_t *record =
            body.data() + static_cast<std::size_t>(index) * kGtRaysRecordSize;
        GtRay ray;
        ray.ray_id = GetU64(record + 0);
        ray.origin_W = Eigen::Vector3d(GetF64(record + 8), GetF64(record + 16),
                                       GetF64(record + 24));
        ray.dir_W = Eigen::Vector3d(GetF64(record + 32), GetF64(record + 40),
                                    GetF64(record + 48));
        ray.t_gt = GetF64(record + 56);
        ray.t_min = GetF64(record + 64);
        ray.t_max = GetF64(record + 72);
        ray.frame = GetU32(record + 80);
        ray.camera = GetU16(record + 84);

        ValidateGtRayValues(ray, path.string());
        if (!ids.insert(ray.ray_id).second)
        {
            Reject("gt_rays.bin contains a duplicate ray_id: " + path.string());
        }
        rays.push_back(ray);
    }

    if (info != nullptr)
    {
        info->count = count;
        info->version = version;
        info->record_size = recordSize;
        info->sha256 = Sha256File(path);
    }
    return rays;
}

void WriteHits(const fs::path &path, const std::span<const Hit> hits)
{
    if (hits.size() > static_cast<std::size_t>(kMaxRecordCount))
    {
        Reject("hits.bin count exceeds the record cap: " + path.string());
    }

    std::array<std::uint8_t, kSimIoHeaderSize> header{};
    std::memcpy(header.data(), kHitsMagic, sizeof(kHitsMagic) - 1U);
    PutU16(header.data() + 4, kHitsVersion);
    PutU16(header.data() + 6, kHitsRecordSize);
    PutU32(header.data() + 8, static_cast<std::uint32_t>(hits.size()));

    std::vector<std::uint8_t> body(hits.size() * kHitsRecordSize, 0U);
    for (std::size_t index = 0; index < hits.size(); ++index)
    {
        const Hit &hit = hits[index];
        ValidateHitValues(hit, path.string());

        std::uint8_t *record = body.data() + index * kHitsRecordSize;
        PutU64(record + 0, hit.ray_id);
        PutF64(record + 8, hit.t_pred);
        record[16] = hit.hit ? 1U : 0U;
        // record[17..19] stays zero reserved padding.
        PutU32(record + 20, hit.prim_id);
    }

    std::vector<std::uint8_t> bytes;
    bytes.reserve(header.size() + body.size());
    bytes.insert(bytes.end(), header.begin(), header.end());
    bytes.insert(bytes.end(), body.begin(), body.end());
    WriteBytesAtomic(path, bytes.data(), bytes.size());
}

std::vector<Hit> ReadHits(const fs::path &path, HitsInfo *info)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input.good())
    {
        Reject("cannot read hits.bin " + path.string());
    }
    const std::streamoff end = input.tellg();
    if (end < static_cast<std::streamoff>(kSimIoHeaderSize))
    {
        Reject("hits.bin is shorter than its header: " + path.string());
    }
    const std::uint64_t fileSize = static_cast<std::uint64_t>(end);
    input.seekg(0, std::ios::beg);

    std::array<std::uint8_t, kSimIoHeaderSize> header{};
    input.read(reinterpret_cast<char *>(header.data()),
               static_cast<std::streamsize>(header.size()));
    if (!input.good())
    {
        Reject("cannot read hits.bin header " + path.string());
    }
    if (std::memcmp(header.data(), kHitsMagic, sizeof(kHitsMagic) - 1U) != 0)
    {
        Reject("hits.bin has a wrong magic: " + path.string());
    }
    const std::uint16_t version = GetU16(header.data() + 4);
    if (version != kHitsVersion)
    {
        Reject("hits.bin has an unsupported version: " + path.string());
    }
    const std::uint16_t recordSize = GetU16(header.data() + 6);
    if (recordSize != kHitsRecordSize)
    {
        Reject("hits.bin has a wrong record size: " + path.string());
    }
    const std::uint32_t count = GetU32(header.data() + 8);
    if (count > kMaxRecordCount)
    {
        Reject("hits.bin count exceeds the record cap: " + path.string());
    }
    const std::uint64_t expected =
        static_cast<std::uint64_t>(kSimIoHeaderSize) +
        static_cast<std::uint64_t>(count) * static_cast<std::uint64_t>(kHitsRecordSize);
    if (fileSize != expected)
    {
        Reject("hits.bin length does not match its header count: " + path.string());
    }

    std::vector<std::uint8_t> body(static_cast<std::size_t>(count) * kHitsRecordSize);
    if (!body.empty())
    {
        input.read(reinterpret_cast<char *>(body.data()),
                   static_cast<std::streamsize>(body.size()));
        if (!input.good())
        {
            Reject("cannot read hits.bin records " + path.string());
        }
    }

    std::vector<Hit> hits;
    hits.reserve(count);
    for (std::uint32_t index = 0; index < count; ++index)
    {
        const std::uint8_t *record = body.data() + static_cast<std::size_t>(index) * kHitsRecordSize;
        const std::uint8_t hitByte = record[16];
        if (hitByte > 1U)
        {
            Reject("hits.bin contains an invalid hit value: " + path.string());
        }
        Hit hit;
        hit.ray_id = GetU64(record + 0);
        hit.t_pred = GetF64(record + 8);
        hit.hit = hitByte == 1U;
        hit.prim_id = GetU32(record + 20);

        ValidateHitValues(hit, path.string());
        hits.push_back(hit);
    }

    if (info != nullptr)
    {
        info->count = count;
        info->version = version;
        info->record_size = recordSize;
        info->sha256 = Sha256File(path);
    }
    return hits;
}

void ValidateHitsAgainstRays(const std::span<const GtRay> rays, const std::span<const Hit> hits)
{
    if (rays.size() != hits.size())
    {
        Reject("hits.bin count does not match gt_rays.bin count");
    }
    for (std::size_t index = 0; index < rays.size(); ++index)
    {
        const GtRay &ray = rays[index];
        const Hit &hit = hits[index];
        if (hit.ray_id != ray.ray_id)
        {
            Reject("hits.bin ray_id does not match the ordered gt_rays.bin ray_id");
        }
        if (hit.hit)
        {
            if (!std::isfinite(hit.t_pred) || hit.t_pred <= 0.0)
            {
                Reject("hits.bin contains a hit with an invalid t_pred");
            }
            if (!(ray.t_min <= hit.t_pred && hit.t_pred <= ray.t_max))
            {
                Reject("hits.bin contains a hit with t_pred outside [t_min, t_max]");
            }
        }
        else
        {
            if (!std::isinf(hit.t_pred) || hit.t_pred <= 0.0)
            {
                Reject("hits.bin contains a miss with a finite t_pred");
            }
            if (hit.prim_id != kPrimIdUnavailable)
            {
                Reject("hits.bin contains a miss without the 0xFFFFFFFF prim_id sentinel");
            }
        }
    }
}
} // namespace raylidar::io
