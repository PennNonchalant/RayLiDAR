#include <raylidar/io/sim_io.h>

#include <raylidar/common/io.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
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

namespace
{
namespace fs = std::filesystem;

constexpr std::size_t kMaxPlyHeaderLine = 1U << 16U;
constexpr std::size_t kMaxObjLine = 1U << 20U;
constexpr std::uint32_t kGlbJsonChunk = 0x4E4F534AU;
constexpr std::uint32_t kGlbBinChunk = 0x004E4942U;

std::string LowerExtension(const fs::path &path)
{
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return extension;
}

void StripCarriageReturn(std::string &line)
{
    if (!line.empty() && line.back() == '\r')
    {
        line.pop_back();
    }
}

std::string Trim(std::string_view text)
{
    std::size_t begin = 0;
    std::size_t end = text.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(text[begin])) != 0)
    {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0)
    {
        --end;
    }
    return std::string(text.substr(begin, end - begin));
}

std::vector<std::string> SplitWhitespace(std::string_view text)
{
    std::vector<std::string> tokens;
    std::size_t index = 0;
    while (index < text.size())
    {
        while (index < text.size() &&
               std::isspace(static_cast<unsigned char>(text[index])) != 0)
        {
            ++index;
        }
        const std::size_t start = index;
        while (index < text.size() &&
               std::isspace(static_cast<unsigned char>(text[index])) == 0)
        {
            ++index;
        }
        if (index > start)
        {
            tokens.emplace_back(text.substr(start, index - start));
        }
    }
    return tokens;
}

int PlyScalarSize(const std::string &type)
{
    if (type == "char" || type == "uchar" || type == "int8" || type == "uint8")
    {
        return 1;
    }
    if (type == "short" || type == "ushort" || type == "int16" || type == "uint16")
    {
        return 2;
    }
    if (type == "int" || type == "uint" || type == "int32" || type == "uint32" ||
        type == "float" || type == "float32")
    {
        return 4;
    }
    if (type == "double" || type == "float64")
    {
        return 8;
    }
    return -1;
}

bool PlyIsFloat(const std::string &type)
{
    return type == "float" || type == "float32" || type == "double" || type == "float64";
}

bool PlyIsDouble(const std::string &type)
{
    return type == "double" || type == "float64";
}

double DecodePlyScalar(const std::uint8_t *bytes, const std::string &type)
{
    if (PlyIsDouble(type))
    {
        std::uint64_t bits = 0;
        for (unsigned index = 0; index < 8U; ++index)
        {
            bits |= static_cast<std::uint64_t>(bytes[index]) << (8U * index);
        }
        double value = 0.0;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
    std::uint32_t bits = 0;
    for (unsigned index = 0; index < 4U; ++index)
    {
        bits |= static_cast<std::uint32_t>(bytes[index]) << (8U * index);
    }
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return static_cast<double>(value);
}

std::int64_t DecodePlyInt(const std::uint8_t *bytes, const std::string &type)
{
    if (type == "char" || type == "int8")
    {
        std::int8_t value = 0;
        std::memcpy(&value, bytes, 1U);
        return value;
    }
    if (type == "uchar" || type == "uint8")
    {
        return static_cast<std::int64_t>(bytes[0]);
    }
    if (type == "short" || type == "int16")
    {
        const std::uint16_t raw = static_cast<std::uint16_t>(bytes[0]) |
                                  static_cast<std::uint16_t>(
                                      static_cast<std::uint16_t>(bytes[1]) << 8U);
        std::int16_t value = 0;
        std::memcpy(&value, &raw, 2U);
        return value;
    }
    if (type == "ushort" || type == "uint16")
    {
        return static_cast<std::int64_t>(static_cast<std::uint16_t>(bytes[0]) |
                                         static_cast<std::uint16_t>(
                                             static_cast<std::uint16_t>(bytes[1]) << 8U));
    }
    if (type == "uint" || type == "uint32")
    {
        return static_cast<std::int64_t>(GetU32(bytes));
    }
    std::int32_t value = 0;
    const std::uint32_t raw = GetU32(bytes);
    std::memcpy(&value, &raw, sizeof(value));
    return value;
}

struct PlyVertexProperty
{
    std::string name;
    std::string type;
    std::size_t offset = 0;
};

MeshStats ValidatePly(const fs::path &path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input.good())
    {
        Reject("cannot read PLY mesh " + path.string());
    }

    enum class Element
    {
        None,
        Vertex,
        Face
    };

    std::string line;
    std::size_t lineNumber = 0;
    bool sawPly = false;
    bool sawFormat = false;
    bool sawFrameComment = false;
    bool sawUnitsComment = false;
    bool sawEndHeader = false;
    bool sawVertexElement = false;
    bool sawFaceElement = false;
    bool hasFaceList = false;
    std::string faceCountType;
    std::string faceIndexType;
    std::uint64_t vertexCount = 0;
    std::uint64_t faceCount = 0;
    std::size_t vertexStride = 0;
    Element current = Element::None;
    std::vector<PlyVertexProperty> properties;

    while (std::getline(input, line))
    {
        ++lineNumber;
        StripCarriageReturn(line);
        if (line.size() > kMaxPlyHeaderLine)
        {
            Reject("PLY header line is too long: " + path.string());
        }
        const std::string trimmed = Trim(line);
        if (lineNumber == 1U)
        {
            if (trimmed != "ply")
            {
                Reject("PLY file does not start with 'ply': " + path.string());
            }
            sawPly = true;
            continue;
        }
        if (trimmed.empty())
        {
            continue;
        }
        const std::vector<std::string> tokens = SplitWhitespace(trimmed);
        if (tokens.empty())
        {
            continue;
        }
        const std::string &keyword = tokens[0];
        if (!sawFormat && keyword == "format")
        {
            if (trimmed.find("binary_little_endian") == std::string::npos)
            {
                if (trimmed.find("binary_big_endian") != std::string::npos)
                {
                    Reject("PLY must be little-endian, not binary_big_endian: " + path.string());
                }
                Reject("PLY must be binary_little_endian: " + path.string());
            }
            sawFormat = true;
            continue;
        }
        if (keyword == "comment")
        {
            if (trimmed == "comment frame W")
            {
                sawFrameComment = true;
            }
            else if (trimmed == "comment units m")
            {
                sawUnitsComment = true;
            }
            continue;
        }
        if (keyword == "element")
        {
            if (tokens.size() < 3U)
            {
                Reject("PLY element line is malformed: " + path.string());
            }
            std::uint64_t count = 0;
            try
            {
                count = std::stoull(tokens[2]);
            }
            catch (const std::exception &)
            {
                Reject("PLY element count is malformed: " + path.string());
            }
            if (tokens[1] == "vertex")
            {
                sawVertexElement = true;
                vertexCount = count;
                current = Element::Vertex;
            }
            else if (tokens[1] == "face")
            {
                sawFaceElement = true;
                faceCount = count;
                current = Element::Face;
            }
            else
            {
                current = Element::None;
            }
            continue;
        }
        if (keyword == "property")
        {
            if (current == Element::Vertex)
            {
                if (tokens.size() >= 2U && tokens[1] == "list")
                {
                    Reject("PLY vertex list properties are not supported: " + path.string());
                }
                if (tokens.size() < 3U)
                {
                    Reject("PLY property line is malformed: " + path.string());
                }
                const int size = PlyScalarSize(tokens[1]);
                if (size < 0)
                {
                    Reject("PLY property has an unknown type: " + tokens[1]);
                }
                properties.push_back(PlyVertexProperty{tokens[2], tokens[1], vertexStride});
                vertexStride += static_cast<std::size_t>(size);
            }
            else if (current == Element::Face && tokens.size() >= 5U && tokens[1] == "list")
            {
                faceCountType = tokens[2];
                faceIndexType = tokens[3];
                hasFaceList = true;
            }
            continue;
        }
        if (keyword == "end_header")
        {
            sawEndHeader = true;
            break;
        }
    }

    if (!sawPly || !sawFormat || !sawEndHeader)
    {
        Reject("PLY header is incomplete: " + path.string());
    }
    if (!sawFrameComment)
    {
        Reject("PLY is missing the 'comment frame W' line: " + path.string());
    }
    if (!sawUnitsComment)
    {
        Reject("PLY is missing the 'comment units m' line: " + path.string());
    }
    if (!sawVertexElement || vertexCount == 0U)
    {
        Reject("PLY mesh has no vertex records: " + path.string());
    }
    if (!sawFaceElement || faceCount == 0U)
    {
        Reject("PLY mesh has no triangles: " + path.string());
    }
    if (!hasFaceList)
    {
        Reject("PLY faces must be triangle index lists: " + path.string());
    }

    std::size_t xOffset = 0;
    std::size_t yOffset = 0;
    std::size_t zOffset = 0;
    std::string xType;
    std::string yType;
    std::string zType;
    bool hasX = false;
    bool hasY = false;
    bool hasZ = false;
    bool hasNormals = false;
    bool hasColors = false;
    for (const PlyVertexProperty &property : properties)
    {
        if (property.name == "x")
        {
            hasX = PlyIsFloat(property.type);
            xOffset = property.offset;
            xType = property.type;
        }
        else if (property.name == "y")
        {
            hasY = PlyIsFloat(property.type);
            yOffset = property.offset;
            yType = property.type;
        }
        else if (property.name == "z")
        {
            hasZ = PlyIsFloat(property.type);
            zOffset = property.offset;
            zType = property.type;
        }
        else if (property.name == "nx" || property.name == "ny" || property.name == "nz")
        {
            hasNormals = true;
        }
        else if (property.name == "red" || property.name == "green" || property.name == "blue")
        {
            hasColors = true;
        }
    }
    if (!hasX || !hasY || !hasZ || vertexStride == 0U)
    {
        Reject("PLY is missing the float x/y/z vertex properties: " + path.string());
    }
    const int faceCountSize = PlyScalarSize(faceCountType);
    const int faceIndexSize = PlyScalarSize(faceIndexType);
    if (faceCountSize < 0 || faceIndexSize < 0)
    {
        Reject("PLY face list has an unknown index type: " + path.string());
    }

    const std::streamoff headerBytes = input.tellg();
    if (headerBytes < 0)
    {
        Reject("cannot locate the PLY body: " + path.string());
    }
    std::error_code sizeError;
    const std::uint64_t fileSize = fs::file_size(path, sizeError);
    if (sizeError || fileSize < static_cast<std::uint64_t>(headerBytes))
    {
        Reject("cannot read the PLY body size: " + path.string());
    }
    const std::uint64_t remaining = fileSize - static_cast<std::uint64_t>(headerBytes);
    if (vertexCount > remaining / static_cast<std::uint64_t>(vertexStride))
    {
        Reject("PLY vertex count exceeds the file size: " + path.string());
    }
    if (faceCount > remaining)
    {
        Reject("PLY face count exceeds the file size: " + path.string());
    }

    std::vector<std::uint8_t> record(vertexStride);
    Eigen::Vector3d bboxMin = Eigen::Vector3d::Zero();
    Eigen::Vector3d bboxMax = Eigen::Vector3d::Zero();
    bool haveBbox = false;
    for (std::uint64_t index = 0; index < vertexCount; ++index)
    {
        input.read(reinterpret_cast<char *>(record.data()),
                   static_cast<std::streamsize>(record.size()));
        if (!input.good())
        {
            Reject("PLY vertex block is truncated: " + path.string());
        }
        const double x = DecodePlyScalar(record.data() + xOffset, xType);
        const double y = DecodePlyScalar(record.data() + yOffset, yType);
        const double z = DecodePlyScalar(record.data() + zOffset, zType);
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
        {
            Reject("PLY contains a non-finite coordinate: " + path.string());
        }
        if (!haveBbox)
        {
            bboxMin = Eigen::Vector3d(x, y, z);
            bboxMax = bboxMin;
            haveBbox = true;
        }
        else
        {
            bboxMin = bboxMin.cwiseMin(Eigen::Vector3d(x, y, z));
            bboxMax = bboxMax.cwiseMax(Eigen::Vector3d(x, y, z));
        }
    }

    std::array<std::uint8_t, 8> scalar{};
    for (std::uint64_t face = 0; face < faceCount; ++face)
    {
        input.read(reinterpret_cast<char *>(scalar.data()), faceCountSize);
        if (!input.good())
        {
            Reject("PLY face block is truncated: " + path.string());
        }
        const std::int64_t indexCount = DecodePlyInt(scalar.data(), faceCountType);
        if (indexCount != 3)
        {
            Reject("PLY face is not a triangle: " + path.string());
        }
        for (int corner = 0; corner < 3; ++corner)
        {
            input.read(reinterpret_cast<char *>(scalar.data()), faceIndexSize);
            if (!input.good())
            {
                Reject("PLY face block is truncated: " + path.string());
            }
            const std::int64_t vertexIndex = DecodePlyInt(scalar.data(), faceIndexType);
            if (vertexIndex < 0 ||
                static_cast<std::uint64_t>(vertexIndex) >= vertexCount)
            {
                Reject("PLY face index is out of range: " + path.string());
            }
        }
    }

    MeshStats stats;
    stats.format = MeshFormat::Ply;
    stats.vertex_count = vertexCount;
    stats.triangle_count = faceCount;
    stats.bbox_min = bboxMin;
    stats.bbox_max = bboxMax;
    stats.has_normals = hasNormals;
    stats.has_colors = hasColors;
    stats.float64_positions = PlyIsDouble(xType) && PlyIsDouble(yType) && PlyIsDouble(zType);
    stats.frame = "W";
    stats.units = "m";
    return stats;
}

MeshStats ValidateGlb(const fs::path &path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input.good())
    {
        Reject("cannot read GLB mesh " + path.string());
    }
    const std::streamoff end = input.tellg();
    if (end < 12)
    {
        Reject("GLB file is shorter than its header: " + path.string());
    }
    const std::size_t fileSize = static_cast<std::size_t>(end);
    input.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(fileSize);
    input.read(reinterpret_cast<char *>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    if (!input.good())
    {
        Reject("cannot read GLB mesh " + path.string());
    }

    if (std::memcmp(bytes.data(), "glTF", 4U) != 0)
    {
        Reject("GLB has a wrong magic: " + path.string());
    }
    if (GetU32(bytes.data() + 4) != 2U)
    {
        Reject("GLB has an unsupported container version: " + path.string());
    }
    if (GetU32(bytes.data() + 8) != fileSize)
    {
        Reject("GLB length does not match the file size: " + path.string());
    }

    std::size_t offset = 12U;
    bool sawJson = false;
    std::string jsonText;
    while (offset + 8U <= bytes.size())
    {
        const std::uint32_t chunkLength = GetU32(bytes.data() + offset);
        const std::uint32_t chunkType = GetU32(bytes.data() + offset + 4U);
        offset += 8U;
        if (chunkLength > bytes.size() - offset)
        {
            Reject("GLB chunk length exceeds the file: " + path.string());
        }
        if (chunkType == kGlbJsonChunk && !sawJson)
        {
            jsonText.assign(reinterpret_cast<const char *>(bytes.data() + offset), chunkLength);
            sawJson = true;
        }
        offset += chunkLength;
    }
    if (offset != bytes.size())
    {
        Reject("GLB chunk lengths do not sum to the file length: " + path.string());
    }
    if (!sawJson)
    {
        Reject("GLB has no JSON chunk: " + path.string());
    }

    MeshStats stats;
    stats.format = MeshFormat::Glb;
    stats.frame = "W";
    stats.units = "m";
    stats.float64_positions = false;

    try
    {
        const nlohmann::json document = nlohmann::json::parse(jsonText);
        if (!document.contains("asset") || !document["asset"].is_object())
        {
            Reject("GLB has no asset metadata: " + path.string());
        }
        const nlohmann::json &asset = document["asset"];
        if (!asset.contains("version") || asset["version"] != "2.0")
        {
            Reject("GLB asset version must be 2.0: " + path.string());
        }
        if (!asset.contains("extras") || !asset["extras"].is_object())
        {
            Reject("GLB asset extras.frame is missing: " + path.string());
        }
        const nlohmann::json &extras = asset["extras"];
        if (!extras.contains("frame") || extras["frame"] != "W")
        {
            Reject("GLB asset extras.frame must be W: " + path.string());
        }
        if (!extras.contains("units") || extras["units"] != "m")
        {
            Reject("GLB asset extras.units must be m: " + path.string());
        }
        if (document.contains("animations"))
        {
            Reject("GLB animations are not supported: " + path.string());
        }
        if (document.contains("skins"))
        {
            Reject("GLB skins are not supported: " + path.string());
        }
        if (document.contains("nodes"))
        {
            if (!document["nodes"].is_array())
            {
                Reject("GLB nodes must be an array: " + path.string());
            }
            for (const nlohmann::json &node : document["nodes"])
            {
                if (!node.is_object())
                {
                    continue;
                }
                for (const char *key : {"matrix", "translation", "rotation", "scale"})
                {
                    if (node.contains(key))
                    {
                        Reject("GLB node has a transform '" + std::string(key) + "': " +
                               path.string());
                    }
                }
            }
        }
        if (!document.contains("accessors") || !document["accessors"].is_array() ||
            document["accessors"].empty())
        {
            Reject("GLB has no accessors: " + path.string());
        }
        const nlohmann::json &accessors = document["accessors"];
        for (const nlohmann::json &accessor : accessors)
        {
            if (!accessor.is_object())
            {
                Reject("GLB accessor is not an object: " + path.string());
            }
            if (accessor.contains("sparse"))
            {
                Reject("GLB sparse accessors are not supported: " + path.string());
            }
        }
        if (!document.contains("meshes") || !document["meshes"].is_array() ||
            document["meshes"].empty())
        {
            Reject("GLB has no mesh: " + path.string());
        }

        std::uint64_t vertexCount = 0;
        std::uint64_t triangleCount = 0;
        bool haveBbox = false;
        Eigen::Vector3d bboxMin = Eigen::Vector3d::Zero();
        Eigen::Vector3d bboxMax = Eigen::Vector3d::Zero();
        for (const nlohmann::json &mesh : document["meshes"])
        {
            if (!mesh.is_object() || !mesh.contains("primitives") ||
                !mesh["primitives"].is_array())
            {
                Reject("GLB mesh has no primitives: " + path.string());
            }
            for (const nlohmann::json &primitive : mesh["primitives"])
            {
                if (!primitive.is_object())
                {
                    Reject("GLB primitive is not an object: " + path.string());
                }
                if (primitive.contains("targets"))
                {
                    Reject("GLB morph targets are not supported: " + path.string());
                }
                if (!primitive.contains("attributes") ||
                    !primitive["attributes"].is_object() ||
                    !primitive["attributes"].contains("POSITION"))
                {
                    Reject("GLB primitive has no POSITION accessor: " + path.string());
                }
                const nlohmann::json &attributes = primitive["attributes"];
                const std::size_t positionIndex =
                    attributes["POSITION"].get<std::size_t>();
                if (positionIndex >= accessors.size())
                {
                    Reject("GLB POSITION accessor index is out of range: " + path.string());
                }
                const nlohmann::json &position = accessors[positionIndex];
                if (position.value("componentType", 0) != 5126)
                {
                    Reject("GLB POSITION accessor must have componentType 5126: " +
                           path.string());
                }
                const std::uint64_t count =
                    position.value("count", static_cast<std::uint64_t>(0));
                if (count == 0U)
                {
                    Reject("GLB POSITION accessor count is zero: " + path.string());
                }
                if (!position.contains("min") || !position["min"].is_array() ||
                    position["min"].size() != 3U)
                {
                    Reject("GLB POSITION accessor min is missing: " + path.string());
                }
                if (!position.contains("max") || !position["max"].is_array() ||
                    position["max"].size() != 3U)
                {
                    Reject("GLB POSITION accessor max is missing: " + path.string());
                }
                const nlohmann::json &minValue = position["min"];
                const nlohmann::json &maxValue = position["max"];
                Eigen::Vector3d primitiveMin = Eigen::Vector3d::Zero();
                Eigen::Vector3d primitiveMax = Eigen::Vector3d::Zero();
                for (int axis = 0; axis < 3; ++axis)
                {
                    if (!minValue[axis].is_number() || !maxValue[axis].is_number())
                    {
                        Reject("GLB POSITION min/max must be numeric: " + path.string());
                    }
                    const double low = minValue[axis].get<double>();
                    const double high = maxValue[axis].get<double>();
                    if (!std::isfinite(low) || !std::isfinite(high))
                    {
                        Reject("GLB POSITION min/max is non-finite: " + path.string());
                    }
                    if (low > high)
                    {
                        Reject("GLB POSITION min exceeds max: " + path.string());
                    }
                    primitiveMin[axis] = low;
                    primitiveMax[axis] = high;
                }
                if (!haveBbox)
                {
                    bboxMin = primitiveMin;
                    bboxMax = primitiveMax;
                    haveBbox = true;
                }
                else
                {
                    bboxMin = bboxMin.cwiseMin(primitiveMin);
                    bboxMax = bboxMax.cwiseMax(primitiveMax);
                }
                vertexCount += count;
                if (primitive.contains("indices"))
                {
                    const std::size_t indexAccessor = primitive["indices"].get<std::size_t>();
                    if (indexAccessor >= accessors.size())
                    {
                        Reject("GLB index accessor index is out of range: " + path.string());
                    }
                    const std::uint64_t indexCount =
                        accessors[indexAccessor].value("count",
                                                       static_cast<std::uint64_t>(0));
                    if (indexCount == 0U || (indexCount % 3U) != 0U)
                    {
                        Reject("GLB index count is not a multiple of three: " +
                               path.string());
                    }
                    triangleCount += indexCount / 3U;
                }
                else
                {
                    if ((count % 3U) != 0U)
                    {
                        Reject("GLB POSITION count is not a multiple of three: " +
                               path.string());
                    }
                    triangleCount += count / 3U;
                }
                if (attributes.contains("NORMAL"))
                {
                    stats.has_normals = true;
                }
                if (attributes.contains("COLOR_0"))
                {
                    stats.has_colors = true;
                }
            }
        }
        if (vertexCount == 0U)
        {
            Reject("GLB mesh has no vertex records: " + path.string());
        }
        if (triangleCount == 0U)
        {
            Reject("GLB mesh has no triangles: " + path.string());
        }
        stats.vertex_count = vertexCount;
        stats.triangle_count = triangleCount;
        stats.bbox_min = bboxMin;
        stats.bbox_max = bboxMax;
    }
    catch (const nlohmann::json::exception &error)
    {
        Reject("GLB JSON is malformed: " + std::string(error.what()));
    }
    return stats;
}

MeshStats ValidateObj(const fs::path &path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input.good())
    {
        Reject("cannot read OBJ mesh " + path.string());
    }

    std::string line;
    bool sawFrameComment = false;
    bool sawUnitsComment = false;
    std::uint64_t vertexCount = 0;
    std::uint64_t triangleCount = 0;
    bool haveBbox = false;
    Eigen::Vector3d bboxMin = Eigen::Vector3d::Zero();
    Eigen::Vector3d bboxMax = Eigen::Vector3d::Zero();

    while (std::getline(input, line))
    {
        if (line.size() > kMaxObjLine)
        {
            Reject("OBJ line exceeds 1 MiB: " + path.string());
        }
        StripCarriageReturn(line);
        const std::string trimmed = Trim(line);
        if (trimmed.empty())
        {
            continue;
        }
        if (trimmed[0] == '#')
        {
            if (trimmed == "# frame W")
            {
                sawFrameComment = true;
            }
            else if (trimmed == "# units m")
            {
                sawUnitsComment = true;
            }
            continue;
        }
        const std::vector<std::string> tokens = SplitWhitespace(trimmed);
        if (tokens.empty())
        {
            continue;
        }
        const std::string &keyword = tokens[0];
        if (keyword == "v")
        {
            if (!sawFrameComment || !sawUnitsComment)
            {
                Reject("OBJ is missing the '# frame W'/'# units m' comments before "
                       "vertices: " +
                       path.string());
            }
            if (tokens.size() < 4U)
            {
                Reject("OBJ vertex line is malformed: " + path.string());
            }
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            try
            {
                x = std::stod(tokens[1]);
                y = std::stod(tokens[2]);
                z = std::stod(tokens[3]);
            }
            catch (const std::exception &)
            {
                Reject("OBJ vertex line contains a malformed number: " + path.string());
            }
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
            {
                Reject("OBJ contains a non-finite coordinate: " + path.string());
            }
            if (!haveBbox)
            {
                bboxMin = Eigen::Vector3d(x, y, z);
                bboxMax = bboxMin;
                haveBbox = true;
            }
            else
            {
                bboxMin = bboxMin.cwiseMin(Eigen::Vector3d(x, y, z));
                bboxMax = bboxMax.cwiseMax(Eigen::Vector3d(x, y, z));
            }
            ++vertexCount;
            continue;
        }
        if (keyword == "f")
        {
            if (tokens.size() != 4U)
            {
                Reject("OBJ face is not a triangle: " + path.string());
            }
            for (std::size_t corner = 1U; corner < tokens.size(); ++corner)
            {
                const std::string &token = tokens[corner];
                const std::size_t slash = token.find('/');
                const std::string index = token.substr(0, slash);
                if (index.empty() || index[0] == '-')
                {
                    Reject("OBJ face index is out of range: " + path.string());
                }
                long long value = 0;
                try
                {
                    value = std::stoll(index);
                }
                catch (const std::exception &)
                {
                    Reject("OBJ face index is malformed: " + path.string());
                }
                if (value <= 0 || static_cast<std::uint64_t>(value) > vertexCount)
                {
                    Reject("OBJ face index is out of range: " + path.string());
                }
            }
            ++triangleCount;
            continue;
        }
        if (keyword == "o" || keyword == "g" || keyword == "vp" || keyword == "l" ||
            keyword == "p")
        {
            Reject("OBJ transform directive '" + keyword + "' is not supported: " +
                   path.string());
        }
        // vn, vt, s, usemtl, and mtllib carry no geometry and are ignored.
    }

    if (vertexCount == 0U)
    {
        Reject("OBJ mesh has no vertex records: " + path.string());
    }
    if (triangleCount == 0U)
    {
        Reject("OBJ mesh has no triangles: " + path.string());
    }
    if (!sawFrameComment)
    {
        Reject("OBJ is missing '# frame W': " + path.string());
    }
    if (!sawUnitsComment)
    {
        Reject("OBJ is missing '# units m': " + path.string());
    }

    MeshStats stats;
    stats.format = MeshFormat::Obj;
    stats.vertex_count = vertexCount;
    stats.triangle_count = triangleCount;
    stats.bbox_min = bboxMin;
    stats.bbox_max = bboxMax;
    stats.float64_positions = false;
    stats.frame = "W";
    stats.units = "m";
    return stats;
}
} // namespace

MeshFormat FormatFromExtension(const fs::path &path)
{
    const std::string extension = LowerExtension(path);
    if (extension == ".ply")
    {
        return MeshFormat::Ply;
    }
    if (extension == ".glb")
    {
        return MeshFormat::Glb;
    }
    if (extension == ".obj")
    {
        return MeshFormat::Obj;
    }
    Reject("mesh file has an unknown extension: " + path.string());
}

MeshStats ValidateMeshFile(const fs::path &path)
{
    std::error_code error;
    if (!fs::exists(path, error) || !fs::is_regular_file(path, error))
    {
        Reject("mesh file is missing: " + path.string());
    }
    switch (FormatFromExtension(path))
    {
    case MeshFormat::Ply:
        return ValidatePly(path);
    case MeshFormat::Glb:
        return ValidateGlb(path);
    case MeshFormat::Obj:
        return ValidateObj(path);
    }
    Reject("mesh file has an unknown extension: " + path.string());
}
} // namespace raylidar::io
