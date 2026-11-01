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
using raylidar::io::Hit;
using raylidar::io::HitsInfo;
using raylidar::io::kPrimIdUnavailable;
using raylidar::io::ReadHits;
using raylidar::io::ValidateHitsAgainstRays;
using raylidar::io::WriteHits;
using raylidar::test::EncodeHits;
using raylidar::test::HitBytes;
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
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8U) |
           (static_cast<std::uint32_t>(bytes[2]) << 16U) |
           (static_cast<std::uint32_t>(bytes[3]) << 24U);
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

GtRay MakeRay(const std::uint64_t ray_id, const double t_min, const double t_max)
{
    GtRay ray;
    ray.ray_id = ray_id;
    ray.origin_W = Eigen::Vector3d::Zero();
    ray.dir_W = Eigen::Vector3d(0.0, 0.0, 1.0);
    ray.t_gt = t_min;
    ray.t_min = t_min;
    ray.t_max = t_max;
    return ray;
}

Hit MakeHit(const std::uint64_t ray_id, const bool hit, const double t_pred,
            const std::uint32_t prim_id)
{
    Hit value;
    value.ray_id = ray_id;
    value.hit = hit;
    value.t_pred = t_pred;
    value.prim_id = prim_id;
    return value;
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
        CHECK_MESSAGE(std::string(error.what()).find(needle) != std::string::npos, error.what());
        return;
    }
    FAIL("expected raylidar::IoError");
}

void ExpectBytesRejected(const fs::path &root, const std::vector<std::uint8_t> &bytes,
                         const std::string &needle)
{
    const fs::path path = WriteTempFile(root, "bad.hits", bytes);
    ExpectIoError([&path] { (void)ReadHits(path); }, needle);
}
} // namespace

TEST_CASE("hits round-trip preserves every field", "[io][hits][sim_io]")
{
    const fs::path root = MakeIoTempRoot("hits_roundtrip");
    const std::vector<Hit> hits = {MakeHit(1U, true, 12.5, 17U),
                                   MakeHit(2U, false, std::numeric_limits<double>::infinity(),
                                           kPrimIdUnavailable),
                                   MakeHit(3U, true, 0.5, kPrimIdUnavailable)};

    const fs::path path = root / "hits.bin";
    WriteHits(path, hits);
    REQUIRE(fs::exists(path));

    const std::vector<std::uint8_t> bytes = ReadAllBytes(path);
    REQUIRE(bytes.size() == 32U + hits.size() * 24U);
    CHECK(std::memcmp(bytes.data(), "RLHT", 4) == 0);
    CHECK(GetU16(bytes.data() + 4) == 1U);
    CHECK(GetU16(bytes.data() + 6) == 24U);
    CHECK(GetU32(bytes.data() + 8) == hits.size());

    HitsInfo info;
    const std::vector<Hit> decoded = ReadHits(path, &info);
    REQUIRE(decoded.size() == hits.size());
    for (std::size_t index = 0; index < hits.size(); ++index)
    {
        CHECK(decoded[index].ray_id == hits[index].ray_id);
        CHECK(decoded[index].hit == hits[index].hit);
        CHECK(decoded[index].t_pred == hits[index].t_pred);
        CHECK(decoded[index].prim_id == hits[index].prim_id);
    }
    CHECK(info.count == hits.size());
    CHECK(info.version == 1U);
    CHECK(info.record_size == 24U);
    CHECK(info.sha256 == raylidar::Sha256File(path));
}

TEST_CASE("hits independent byte fixture decodes the documented offsets", "[io][hits][sim_io]")
{
    const fs::path root = MakeIoTempRoot("hits_offsets");
    HitBytes fixture;
    fixture.ray_id = 0xAABBCCDDEEFF0011ULL;
    fixture.t_pred = 33.75;
    fixture.hit = 1U;
    fixture.prim_id = 98765U;

    const std::vector<std::uint8_t> bytes = EncodeHits(std::span<const HitBytes>(&fixture, 1));
    const fs::path path = WriteTempFile(root, "fixture.hits", bytes);

    const std::vector<Hit> hits = ReadHits(path);
    REQUIRE(hits.size() == 1U);
    CHECK(hits[0].ray_id == fixture.ray_id);
    CHECK(hits[0].t_pred == fixture.t_pred);
    CHECK(hits[0].hit == true);
    CHECK(hits[0].prim_id == fixture.prim_id);
}

TEST_CASE("hits rejects a malformed header", "[io][hits][sim_io]")
{
    const fs::path root = MakeIoTempRoot("hits_header_errors");
    HitBytes fixture;
    fixture.ray_id = 1U;
    const std::vector<std::uint8_t> valid = EncodeHits(std::span<const HitBytes>(&fixture, 1));

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
        SetU16(bytes, 6, 32U);
        ExpectBytesRejected(root, bytes, "record size");
    }
    {
        auto bytes = valid;
        bytes.resize(16U);
        ExpectBytesRejected(root, bytes, "header");
    }
    {
        auto bytes = valid;
        bytes.resize(bytes.size() - 2U);
        ExpectBytesRejected(root, bytes, "length");
    }
    {
        auto bytes = valid;
        bytes.push_back(0U);
        ExpectBytesRejected(root, bytes, "length");
    }
}

TEST_CASE("hits rejects invalid records", "[io][hits][sim_io]")
{
    const fs::path root = MakeIoTempRoot("hits_record_errors");

    {
        HitBytes bad;
        bad.ray_id = 1U;
        bad.hit = 2U;
        ExpectBytesRejected(root, EncodeHits(std::span<const HitBytes>(&bad, 1)), "hit");
    }
    {
        HitBytes bad;
        bad.ray_id = 1U;
        bad.hit = 1U;
        bad.t_pred = std::numeric_limits<double>::infinity();
        ExpectBytesRejected(root, EncodeHits(std::span<const HitBytes>(&bad, 1)), "t_pred");
    }
    {
        HitBytes bad;
        bad.ray_id = 1U;
        bad.hit = 1U;
        bad.t_pred = std::numeric_limits<double>::quiet_NaN();
        ExpectBytesRejected(root, EncodeHits(std::span<const HitBytes>(&bad, 1)), "t_pred");
    }
    {
        HitBytes bad;
        bad.ray_id = 1U;
        bad.hit = 1U;
        bad.t_pred = 0.0;
        ExpectBytesRejected(root, EncodeHits(std::span<const HitBytes>(&bad, 1)), "t_pred");
    }
    {
        HitBytes bad;
        bad.ray_id = 1U;
        bad.hit = 0U;
        bad.t_pred = 5.0;
        const std::vector<std::uint8_t> bytes =
            EncodeHits(std::span<const HitBytes>(&bad, 1));
        ExpectBytesRejected(root, bytes, "miss");
    }
    {
        HitBytes bad;
        bad.ray_id = 1U;
        bad.hit = 0U;
        bad.t_pred = std::numeric_limits<double>::infinity();
        bad.prim_id = 5U;
        ExpectBytesRejected(root, EncodeHits(std::span<const HitBytes>(&bad, 1)), "prim_id");
    }
}

TEST_CASE("hits rejects a count above the cap or the file size", "[io][hits][sim_io]")
{
    const fs::path root = MakeIoTempRoot("hits_count_errors");
    HitBytes fixture;
    fixture.ray_id = 1U;
    const std::vector<std::uint8_t> valid = EncodeHits(std::span<const HitBytes>(&fixture, 1));

    {
        auto bytes = valid;
        SetU32(bytes, 8, 0xFFFFFFFFU);
        const fs::path path = WriteTempFile(root, "huge.hits", bytes);
        // 0xFFFFFFFF * 24 + 32 fits in uint64 but exceeds the file size.
        ExpectIoError([&path] { (void)ReadHits(path); }, "length");
    }
    {
        auto bytes = valid;
        SetU32(bytes, 8, 300000000U);
        const fs::path path = WriteTempFile(root, "cap.hits", bytes);
        ExpectIoError([&path] { (void)ReadHits(path); }, "count");
    }
}

TEST_CASE("cross-validation accepts one-to-one ordered hits", "[io][hits][sim_io]")
{
    const std::vector<GtRay> rays = {MakeRay(10U, 1.0, 60.0), MakeRay(20U, 2.0, 50.0)};
    const std::vector<Hit> hits = {
        MakeHit(10U, true, 10.0, 3U),
        MakeHit(20U, false, std::numeric_limits<double>::infinity(), kPrimIdUnavailable)};

    CHECK_NOTHROW(ValidateHitsAgainstRays(rays, hits));
}

TEST_CASE("cross-validation rejects inconsistent hits", "[io][hits][sim_io]")
{
    const std::vector<GtRay> rays = {MakeRay(10U, 1.0, 60.0), MakeRay(20U, 2.0, 50.0)};
    const double inf = std::numeric_limits<double>::infinity();

    {
        const std::vector<Hit> hits = {MakeHit(10U, true, 10.0, 3U)};
        ExpectIoError([&] { ValidateHitsAgainstRays(rays, hits); }, "count");
    }
    {
        const std::vector<Hit> hits = {MakeHit(20U, true, 10.0, 3U),
                                       MakeHit(10U, true, 10.0, 3U)};
        ExpectIoError([&] { ValidateHitsAgainstRays(rays, hits); }, "ray_id");
    }
    {
        const std::vector<Hit> hits = {MakeHit(10U, true, inf, 3U),
                                       MakeHit(20U, false, inf, kPrimIdUnavailable)};
        ExpectIoError([&] { ValidateHitsAgainstRays(rays, hits); }, "t_pred");
    }
    {
        const std::vector<Hit> hits = {MakeHit(10U, true, 10.0, 3U),
                                       MakeHit(20U, false, 5.0, kPrimIdUnavailable)};
        ExpectIoError([&] { ValidateHitsAgainstRays(rays, hits); }, "miss");
    }
    {
        const std::vector<Hit> hits = {MakeHit(10U, true, 0.5, 3U),
                                       MakeHit(20U, false, inf, kPrimIdUnavailable)};
        ExpectIoError([&] { ValidateHitsAgainstRays(rays, hits); }, "t_pred");
    }
    {
        const std::vector<Hit> hits = {MakeHit(10U, true, 61.0, 3U),
                                       MakeHit(20U, false, inf, kPrimIdUnavailable)};
        ExpectIoError([&] { ValidateHitsAgainstRays(rays, hits); }, "t_pred");
    }
    {
        const std::vector<Hit> hits = {MakeHit(10U, true, 10.0, 3U),
                                       MakeHit(20U, false, inf, 12U)};
        ExpectIoError([&] { ValidateHitsAgainstRays(rays, hits); }, "prim_id");
    }
}
