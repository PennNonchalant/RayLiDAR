#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace raylidar::test
{
/// Fresh empty directory under the system temp directory.
std::filesystem::path MakeIoTempRoot(std::string_view name);

/// Write raw bytes, creating parent directories.
void WriteAllBytes(const std::filesystem::path &path, std::span<const std::uint8_t> bytes);

/// Read a whole file as bytes.
std::vector<std::uint8_t> ReadAllBytes(const std::filesystem::path &path);

/// Write `bytes` to `root/name` and return the path.
std::filesystem::path WriteTempFile(const std::filesystem::path &root, std::string_view name,
                                    std::span<const std::uint8_t> bytes);

/// Write `text` verbatim to `root/name` and return the path.
std::filesystem::path WriteTempText(const std::filesystem::path &root, std::string_view name,
                                    std::string_view text);

// Little-endian append helpers used to build independent byte fixtures.
void AppendU16(std::vector<std::uint8_t> &out, std::uint16_t value);
void AppendU32(std::vector<std::uint8_t> &out, std::uint32_t value);
void AppendU64(std::vector<std::uint8_t> &out, std::uint64_t value);
void AppendF32(std::vector<std::uint8_t> &out, float value);
void AppendF64(std::vector<std::uint8_t> &out, double value);

/// One `gt_rays.bin` record, described independently of the reader under test.
struct GtRayBytes
{
    std::uint64_t ray_id = 0;
    double origin_W[3] = {0.0, 0.0, 0.0};
    double dir_W[3] = {0.0, 0.0, 1.0};
    double t_gt = 1.0;
    double t_min = 1.0e-4;
    double t_max = 60.0;
    std::uint32_t frame = 0;
    std::uint16_t camera = 0;
};

/// Encode a complete `gt_rays.bin` (header + records) with hand-computed offsets.
std::vector<std::uint8_t> EncodeGtRays(std::span<const GtRayBytes> rays);

/// One `hits.bin` record, described independently of the reader under test.
struct HitBytes
{
    std::uint64_t ray_id = 0;
    double t_pred = 1.0;
    std::uint8_t hit = 1;
    std::uint32_t prim_id = 0;
};

/// Encode a complete `hits.bin` (header + records) with hand-computed offsets.
std::vector<std::uint8_t> EncodeHits(std::span<const HitBytes> hits);

/// Concatenate an ASCII PLY header (must end with `end_header\n`) and a binary body.
std::vector<std::uint8_t> BuildPly(std::string_view header, std::span<const std::uint8_t> body);

/// Build a GLB 2.0 container from a JSON document string and a BIN chunk.
std::vector<std::uint8_t> BuildGlb(std::string_view json, std::span<const std::uint8_t> bin);
} // namespace raylidar::test
