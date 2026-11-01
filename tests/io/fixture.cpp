#include "fixture.h"

#include <atomic>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace raylidar::test
{
namespace
{
namespace fs = std::filesystem;

std::atomic<unsigned long> gCounter{0U};

void AppendBytes(std::vector<std::uint8_t> &out, const void *data, const std::size_t size)
{
    const auto *bytes = static_cast<const std::uint8_t *>(data);
    out.insert(out.end(), bytes, bytes + size);
}
} // namespace

fs::path MakeIoTempRoot(const std::string_view name)
{
    std::error_code error;
    const fs::path base = fs::temp_directory_path(error) / "raylidar_io_tests";
    fs::create_directories(base, error);
    const unsigned long counter = gCounter.fetch_add(1U) + 1U;
    const fs::path root = base / (std::string(name) + "_" + std::to_string(counter));
    fs::remove_all(root, error);
    fs::create_directories(root, error);
    return root;
}

void WriteAllBytes(const fs::path &path, const std::span<const std::uint8_t> bytes)
{
    std::error_code error;
    if (!path.parent_path().empty())
    {
        fs::create_directories(path.parent_path(), error);
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output.good())
    {
        throw std::runtime_error("cannot write " + path.string());
    }
    if (!bytes.empty())
    {
        output.write(reinterpret_cast<const char *>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
    }
    if (!output.good())
    {
        throw std::runtime_error("error while writing " + path.string());
    }
}

std::vector<std::uint8_t> ReadAllBytes(const fs::path &path)
{
    std::ifstream input(path, std::ios::binary);
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(input),
                                     std::istreambuf_iterator<char>());
}

fs::path WriteTempFile(const fs::path &root, const std::string_view name,
                       const std::span<const std::uint8_t> bytes)
{
    const fs::path path = root / std::string(name);
    WriteAllBytes(path, bytes);
    return path;
}

fs::path WriteTempText(const fs::path &root, const std::string_view name,
                       const std::string_view text)
{
    const fs::path path = root / std::string(name);
    WriteAllBytes(path,
                  std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t *>(text.data()),
                                                text.size()));
    return path;
}

void AppendU16(std::vector<std::uint8_t> &out, const std::uint16_t value)
{
    const std::uint8_t bytes[2] = {static_cast<std::uint8_t>(value & 0xFFU),
                                   static_cast<std::uint8_t>((value >> 8U) & 0xFFU)};
    AppendBytes(out, bytes, sizeof(bytes));
}

void AppendU32(std::vector<std::uint8_t> &out, const std::uint32_t value)
{
    const std::uint8_t bytes[4] = {static_cast<std::uint8_t>(value & 0xFFU),
                                   static_cast<std::uint8_t>((value >> 8U) & 0xFFU),
                                   static_cast<std::uint8_t>((value >> 16U) & 0xFFU),
                                   static_cast<std::uint8_t>((value >> 24U) & 0xFFU)};
    AppendBytes(out, bytes, sizeof(bytes));
}

void AppendU64(std::vector<std::uint8_t> &out, const std::uint64_t value)
{
    std::uint8_t bytes[8] = {};
    for (unsigned index = 0; index < 8U; ++index)
    {
        bytes[index] = static_cast<std::uint8_t>((value >> (8U * index)) & 0xFFU);
    }
    AppendBytes(out, bytes, sizeof(bytes));
}

void AppendF32(std::vector<std::uint8_t> &out, const float value)
{
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    AppendU32(out, bits);
}

void AppendF64(std::vector<std::uint8_t> &out, const double value)
{
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    AppendU64(out, bits);
}

std::vector<std::uint8_t> EncodeGtRays(const std::span<const GtRayBytes> rays)
{
    std::vector<std::uint8_t> bytes;
    bytes.reserve(32U + rays.size() * 96U);
    // Header: magic[4], version u16, record_size u16, count u32, reserved[20].
    const char magic[4] = {'R', 'L', 'G', 'R'};
    AppendBytes(bytes, magic, sizeof(magic));
    AppendU16(bytes, 1U);
    AppendU16(bytes, 96U);
    AppendU32(bytes, static_cast<std::uint32_t>(rays.size()));
    for (int index = 0; index < 20; ++index)
    {
        bytes.push_back(0U);
    }
    // Records at the documented offsets.
    for (const GtRayBytes &ray : rays)
    {
        AppendU64(bytes, ray.ray_id);              // 0
        AppendF64(bytes, ray.origin_W[0]);         // 8
        AppendF64(bytes, ray.origin_W[1]);         // 16
        AppendF64(bytes, ray.origin_W[2]);         // 24
        AppendF64(bytes, ray.dir_W[0]);            // 32
        AppendF64(bytes, ray.dir_W[1]);            // 40
        AppendF64(bytes, ray.dir_W[2]);            // 48
        AppendF64(bytes, ray.t_gt);                // 56
        AppendF64(bytes, ray.t_min);               // 64
        AppendF64(bytes, ray.t_max);               // 72
        AppendU32(bytes, ray.frame);               // 80
        AppendU16(bytes, ray.camera);              // 84
        for (int index = 0; index < 10; ++index)   // 86 reserved[10]
        {
            bytes.push_back(0U);
        }
    }
    return bytes;
}

std::vector<std::uint8_t> EncodeHits(const std::span<const HitBytes> hits)
{
    std::vector<std::uint8_t> bytes;
    bytes.reserve(32U + hits.size() * 24U);
    const char magic[4] = {'R', 'L', 'H', 'T'};
    AppendBytes(bytes, magic, sizeof(magic));
    AppendU16(bytes, 1U);
    AppendU16(bytes, 24U);
    AppendU32(bytes, static_cast<std::uint32_t>(hits.size()));
    for (int index = 0; index < 20; ++index)
    {
        bytes.push_back(0U);
    }
    for (const HitBytes &hit : hits)
    {
        AppendU64(bytes, hit.ray_id);   // 0
        AppendF64(bytes, hit.t_pred);   // 8
        bytes.push_back(hit.hit);       // 16
        for (int index = 0; index < 3; ++index) // 17 reserved[3]
        {
            bytes.push_back(0U);
        }
        AppendU32(bytes, hit.prim_id);  // 20
    }
    return bytes;
}

std::vector<std::uint8_t> BuildPly(const std::string_view header,
                                   const std::span<const std::uint8_t> body)
{
    std::vector<std::uint8_t> bytes;
    bytes.reserve(header.size() + body.size());
    AppendBytes(bytes, header.data(), header.size());
    AppendBytes(bytes, body.data(), body.size());
    return bytes;
}

std::vector<std::uint8_t> BuildGlb(const std::string_view json,
                                   const std::span<const std::uint8_t> bin)
{
    // Pad each chunk to a 4-byte boundary: JSON with spaces, BIN with zeros.
    std::vector<std::uint8_t> jsonChunk(json.begin(), json.end());
    while ((jsonChunk.size() % 4U) != 0U)
    {
        jsonChunk.push_back(static_cast<std::uint8_t>(' '));
    }
    std::vector<std::uint8_t> binChunk(bin.begin(), bin.end());
    while ((binChunk.size() % 4U) != 0U)
    {
        binChunk.push_back(0U);
    }

    const std::uint32_t total = static_cast<std::uint32_t>(12U + 8U + jsonChunk.size() + 8U +
                                                           binChunk.size());
    std::vector<std::uint8_t> bytes;
    bytes.reserve(total);
    const char magic[4] = {'g', 'l', 'T', 'F'};
    AppendBytes(bytes, magic, sizeof(magic));
    AppendU32(bytes, 2U);        // container version
    AppendU32(bytes, total);     // total length

    AppendU32(bytes, static_cast<std::uint32_t>(jsonChunk.size()));
    AppendU32(bytes, 0x4E4F534AU); // 'JSON'
    AppendBytes(bytes, jsonChunk.data(), jsonChunk.size());

    AppendU32(bytes, static_cast<std::uint32_t>(binChunk.size()));
    AppendU32(bytes, 0x004E4942U); // 'BIN\0'
    AppendBytes(bytes, binChunk.data(), binChunk.size());
    return bytes;
}
} // namespace raylidar::test
