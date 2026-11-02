#include "fixture.h"

#include <raylidar/common/io.h>
#include <raylidar/io/sim_io.h>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::IoError;
using raylidar::io::GtRay;
using raylidar::io::GtRaysInfo;
using raylidar::io::ReadGtRays;
using raylidar::io::WriteGtRays;
using raylidar::test::EncodeGtRays;
using raylidar::test::GtRayBytes;
using raylidar::test::MakeIoTempRoot;
using raylidar::test::ReadAllBytes;
using raylidar::test::WriteTempFile;

std::uint16_t GetU16(const std::uint8_t *bytes)
{
    return static_cast<std::uint16_t>(bytes[0]) |
           static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[1]) << 8U);
}

std::uint32_t GetU32(const std::uint8_t *bytes)
{
    return static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8U) |
           (static_cast<std::uint32_t>(bytes[2]) << 16U) | (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

void SetU16(std::vector<std::uint8_t> &bytes, const std::size_t offset, const std::uint16_t value)
{
    bytes[offset] = static_cast<std::uint8_t>(value & 0xFFU);
    bytes[offset + 1] = static_cast<std::uint8_t>((value >> 8U) & 0xFFU);
}

void SetU32(std::vector<std::uint8_t> &bytes, const std::size_t offset, const std::uint32_t value)
{
    bytes[offset] = static_cast<std::uint8_t>(value & 0xFFU);
    bytes[offset + 1] = static_cast<std::uint8_t>((value >> 8U) & 0xFFU);
    bytes[offset + 2] = static_cast<std::uint8_t>((value >> 16U) & 0xFFU);
    bytes[offset + 3] = static_cast<std::uint8_t>((value >> 24U) & 0xFFU);
}

GtRay MakeRay(const std::uint64_t ray_id, const Eigen::Vector3d &origin, const Eigen::Vector3d &dir, const double t_gt,
              const std::uint32_t frame, const std::uint16_t camera)
{
    GtRay ray;
    ray.ray_id = ray_id;
    ray.origin_W = origin;
    ray.dir_W = dir;
    ray.t_gt = t_gt;
    ray.t_min = 1.0e-4;
    ray.t_max = 60.0;
    ray.frame = frame;
    ray.camera = camera;
    return ray;
}

void ExpectIoError(const std::function<void()> &call, const std::string &needle)
{
    try
    {
        call();
    }
    catch (const IoError &error)
    {
        CHECK(error.exitCode() == 4);
        INFO(error.what());
        CHECK(std::string(error.what()).find(needle) != std::string::npos);
        return;
    }
    FAIL("expected raylidar::IoError");
}

/// Write raw fixture bytes to a `.gtrays` file and read it.
void ExpectBytesRejected(const fs::path &root, const std::vector<std::uint8_t> &bytes, const std::string &needle)
{
    const fs::path path = WriteTempFile(root, "bad.gtrays", bytes);
    ExpectIoError([&path] { (void)ReadGtRays(path); }, needle);
}
} // namespace

TEST_CASE("gt_rays round-trip preserves every field", "[io][gt_rays][sim_io]")
{
    const fs::path root = MakeIoTempRoot("gt_rays_roundtrip");
    const std::vector<GtRay> rays = {
        MakeRay(1U, Eigen::Vector3d(1.5, -2.25, 3.75), Eigen::Vector3d(0.0, 0.0, 1.0), 12.5, 0U, 0U),
        MakeRay(2U, Eigen::Vector3d(-0.5, 0.125, 2.0), Eigen::Vector3d(1.0, 0.0, 0.0), 3.25, 0U, 1U),
        MakeRay(3U, Eigen::Vector3d(10.0, 20.0, 30.0), Eigen::Vector3d(0.0, 1.0, 0.0), 59.75, 7U, 0U)};

    const fs::path path = root / "gt_rays.bin";
    WriteGtRays(path, rays);
    REQUIRE(fs::exists(path));

    const std::vector<std::uint8_t> bytes = ReadAllBytes(path);
    REQUIRE(bytes.size() == 32U + rays.size() * 96U);
    CHECK(std::memcmp(bytes.data(), "RLGR", 4) == 0);
    CHECK(GetU16(bytes.data() + 4) == 1U);
    CHECK(GetU16(bytes.data() + 6) == 96U);
    CHECK(GetU32(bytes.data() + 8) == rays.size());

    GtRaysInfo info;
    const std::vector<GtRay> decoded = ReadGtRays(path, &info);
    REQUIRE(decoded.size() == rays.size());
    for (std::size_t index = 0; index < rays.size(); ++index)
    {
        const GtRay &expected = rays[index];
        const GtRay &actual = decoded[index];
        CHECK(actual.ray_id == expected.ray_id);
        CHECK((actual.origin_W - expected.origin_W).cwiseAbs().maxCoeff() == 0.0);
        CHECK((actual.dir_W - expected.dir_W).cwiseAbs().maxCoeff() == 0.0);
        CHECK(actual.t_gt == expected.t_gt);
        CHECK(actual.t_min == expected.t_min);
        CHECK(actual.t_max == expected.t_max);
        CHECK(actual.frame == expected.frame);
        CHECK(actual.camera == expected.camera);
    }
    CHECK(info.count == rays.size());
    CHECK(info.version == 1U);
    CHECK(info.record_size == 96U);
    CHECK(info.sha256 == raylidar::Sha256File(path));
}

TEST_CASE("gt_rays independent byte fixture decodes the documented offsets", "[io][gt_rays][sim_io]")
{
    const fs::path root = MakeIoTempRoot("gt_rays_offsets");
    GtRayBytes fixture;
    fixture.ray_id = 0x1122334455667788ULL;
    fixture.origin_W[0] = 101.25;
    fixture.origin_W[1] = -7.5;
    fixture.origin_W[2] = 4.5;
    fixture.dir_W[0] = 0.0;
    fixture.dir_W[1] = 0.0;
    fixture.dir_W[2] = 1.0;
    fixture.t_gt = 42.125;
    fixture.t_min = 0.0001;
    fixture.t_max = 60.0;
    fixture.frame = 1234U;
    fixture.camera = 1U;

    const std::vector<std::uint8_t> bytes = EncodeGtRays(std::span<const GtRayBytes>(&fixture, 1));
    const fs::path path = WriteTempFile(root, "fixture.gtrays", bytes);

    const std::vector<GtRay> rays = ReadGtRays(path);
    REQUIRE(rays.size() == 1U);
    CHECK(rays[0].ray_id == fixture.ray_id);
    CHECK(rays[0].origin_W.x() == fixture.origin_W[0]);
    CHECK(rays[0].origin_W.y() == fixture.origin_W[1]);
    CHECK(rays[0].origin_W.z() == fixture.origin_W[2]);
    CHECK(rays[0].dir_W.x() == fixture.dir_W[0]);
    CHECK(rays[0].dir_W.y() == fixture.dir_W[1]);
    CHECK(rays[0].dir_W.z() == fixture.dir_W[2]);
    CHECK(rays[0].t_gt == fixture.t_gt);
    CHECK(rays[0].t_min == fixture.t_min);
    CHECK(rays[0].t_max == fixture.t_max);
    CHECK(rays[0].frame == fixture.frame);
    CHECK(rays[0].camera == fixture.camera);
}

TEST_CASE("gt_rays rejects a malformed header", "[io][gt_rays][sim_io]")
{
    const fs::path root = MakeIoTempRoot("gt_rays_header_errors");
    GtRayBytes fixture;
    fixture.ray_id = 1U;
    const std::vector<std::uint8_t> valid = EncodeGtRays(std::span<const GtRayBytes>(&fixture, 1));

    {
        auto bytes = valid;
        bytes[0] = static_cast<std::uint8_t>('X');
        ExpectBytesRejected(root, bytes, "magic");
    }
    {
        auto bytes = valid;
        SetU16(bytes, 4, 2U);
        ExpectBytesRejected(root, bytes, "version");
    }
    {
        auto bytes = valid;
        SetU16(bytes, 6, 64U);
        ExpectBytesRejected(root, bytes, "record size");
    }
    {
        auto bytes = valid;
        bytes.resize(16U);
        ExpectBytesRejected(root, bytes, "header");
    }
    {
        auto bytes = valid;
        bytes.resize(bytes.size() - 4U);
        ExpectBytesRejected(root, bytes, "length");
    }
    {
        auto bytes = valid;
        bytes.push_back(0U);
        ExpectBytesRejected(root, bytes, "length");
    }
}

TEST_CASE("gt_rays rejects invalid records", "[io][gt_rays][sim_io]")
{
    const fs::path root = MakeIoTempRoot("gt_rays_record_errors");

    {
        GtRayBytes a;
        a.ray_id = 7U;
        GtRayBytes b;
        b.ray_id = 7U;
        const GtRayBytes rays[2] = {a, b};
        ExpectBytesRejected(root, EncodeGtRays(rays), "ray_id");
    }
    {
        GtRayBytes bad;
        bad.ray_id = 1U;
        bad.camera = 2U;
        ExpectBytesRejected(root, EncodeGtRays(std::span<const GtRayBytes>(&bad, 1)), "camera");
    }
    {
        GtRayBytes bad;
        bad.ray_id = 1U;
        bad.origin_W[0] = std::numeric_limits<double>::quiet_NaN();
        ExpectBytesRejected(root, EncodeGtRays(std::span<const GtRayBytes>(&bad, 1)), "non-finite");
    }
    {
        GtRayBytes bad;
        bad.ray_id = 1U;
        bad.dir_W[2] = std::numeric_limits<double>::infinity();
        ExpectBytesRejected(root, EncodeGtRays(std::span<const GtRayBytes>(&bad, 1)), "non-finite");
    }
    {
        GtRayBytes bad;
        bad.ray_id = 1U;
        bad.t_gt = std::numeric_limits<double>::quiet_NaN();
        ExpectBytesRejected(root, EncodeGtRays(std::span<const GtRayBytes>(&bad, 1)), "non-finite");
    }
    {
        GtRayBytes bad;
        bad.ray_id = 1U;
        bad.dir_W[2] = 2.0;
        ExpectBytesRejected(root, EncodeGtRays(std::span<const GtRayBytes>(&bad, 1)), "dir_W");
    }
    {
        GtRayBytes bad;
        bad.ray_id = 1U;
        bad.t_min = 0.0;
        bad.t_gt = 1.0;
        ExpectBytesRejected(root, EncodeGtRays(std::span<const GtRayBytes>(&bad, 1)), "t_min");
    }
    {
        GtRayBytes bad;
        bad.ray_id = 1U;
        bad.t_gt = 1.0e-5;
        ExpectBytesRejected(root, EncodeGtRays(std::span<const GtRayBytes>(&bad, 1)), "t_gt");
    }
    {
        GtRayBytes bad;
        bad.ray_id = 1U;
        bad.t_gt = 60.5;
        ExpectBytesRejected(root, EncodeGtRays(std::span<const GtRayBytes>(&bad, 1)), "t_gt");
    }
}

TEST_CASE("gt_rays rejects a count above the cap or the file size", "[io][gt_rays][sim_io]")
{
    const fs::path root = MakeIoTempRoot("gt_rays_count_errors");
    GtRayBytes fixture;
    fixture.ray_id = 1U;
    const std::vector<std::uint8_t> valid = EncodeGtRays(std::span<const GtRayBytes>(&fixture, 1));

    {
        // Above kMaxRecordCount: rejected from the header alone.
        auto bytes = valid;
        SetU32(bytes, 8, 0xFFFFFFFFU);
        ExpectBytesRejected(root, bytes, "count");
    }
    {
        auto bytes = valid;
        SetU32(bytes, 8, 300000000U);
        ExpectBytesRejected(root, bytes, "count");
    }
    {
        // Within the cap but inconsistent with the actual file length.
        auto bytes = valid;
        SetU32(bytes, 8, 2U);
        ExpectBytesRejected(root, bytes, "length");
    }
}
