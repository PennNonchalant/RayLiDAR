#include "fixture.h"

#include <raylidar/common/io.h>
#include <raylidar/mesh/manifest.h>

#include <catch2/catch_test_macros.hpp>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::IoError;
using raylidar::Sha256File;
using raylidar::WriteJson;
using raylidar::mesh::MeshManifestValidation;
using raylidar::mesh::ValidateMeshManifest;
using raylidar::test::AppendF32;
using raylidar::test::AppendF64;
using raylidar::test::AppendU32;
using raylidar::test::BuildGlb;
using raylidar::test::BuildPly;
using raylidar::test::MakeIoTempRoot;
using raylidar::test::WriteTempFile;
using raylidar::test::WriteTempText;
using nlohmann::json;

constexpr const char *kSha = "0000000000000000000000000000000000000000000000000000000000000000";

fs::path WritePly(const fs::path &root, const std::string &name)
{
    std::string header = "ply\n"
                         "format binary_little_endian 1.0\n"
                         "comment frame W\n"
                         "comment units m\n"
                         "element vertex 3\n"
                         "property double x\n"
                         "property double y\n"
                         "property double z\n"
                         "element face 1\n"
                         "property list uchar int vertex_indices\n"
                         "end_header\n";
    std::vector<std::uint8_t> body;
    AppendF64(body, 0.0);
    AppendF64(body, 0.0);
    AppendF64(body, 0.0);
    AppendF64(body, 1.0);
    AppendF64(body, 0.0);
    AppendF64(body, 0.0);
    AppendF64(body, 0.0);
    AppendF64(body, 1.0);
    AppendF64(body, 0.5);
    body.push_back(3U);
    AppendU32(body, 0U);
    AppendU32(body, 1U);
    AppendU32(body, 2U);
    return WriteTempFile(root, name, BuildPly(header, body));
}

fs::path WriteGlb(const fs::path &root, const std::string &name)
{
    const json document = json::parse(R"JSON(
{
  "asset": {"version": "2.0", "extras": {"frame": "W", "units": "m"}},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0}],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
  "accessors": [
    {"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": 3,
     "type": "VEC3", "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.5]},
    {"bufferView": 1, "byteOffset": 0, "componentType": 5125, "count": 3,
     "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 12}
  ],
  "buffers": [{"byteLength": 48}]
}
)JSON");
    std::vector<std::uint8_t> bin;
    for (int index = 0; index < 3; ++index)
    {
        AppendF32(bin, 0.0F);
        AppendF32(bin, 0.0F);
        AppendF32(bin, 0.0F);
    }
    AppendU32(bin, 0U);
    AppendU32(bin, 1U);
    AppendU32(bin, 2U);
    return WriteTempFile(root, name, BuildGlb(document.dump(), bin));
}

fs::path WriteObj(const fs::path &root, const std::string &name)
{
    const std::string text = "# frame W\n"
                             "# units m\n"
                             "v 0 0 0\n"
                             "v 1 0 0\n"
                             "v 0 1 0.5\n"
                             "f 1 2 3\n";
    return WriteTempText(root, name, text);
}

json FileEntry(const fs::path &root, const fs::path &path, const char *format,
               const std::uint64_t vertexCount, const std::uint64_t triangleCount)
{
    return json{{"path", path.filename().string()},
                {"sha256", Sha256File(path)},
                {"bytes", fs::file_size(path)},
                {"format", format},
                {"vertex_count", vertexCount},
                {"triangle_count", triangleCount}};
}

struct Fixture
{
    fs::path root;
    json manifest;

    fs::path manifestPath() const
    {
        return root / "mesh_manifest.json";
    }

    void write() const
    {
        WriteJson(manifestPath(), manifest);
    }
};

Fixture MakeValidFixture(const std::string &name)
{
    Fixture fixture;
    fixture.root = MakeIoTempRoot(name);
    const fs::path ply = WritePly(fixture.root, "mesh.ply");
    const fs::path glb = WriteGlb(fixture.root, "mesh.glb");
    const fs::path obj = WriteObj(fixture.root, "mesh.obj");
    const fs::path viewer = WriteGlb(fixture.root, "mesh_viewer.glb");

    fixture.manifest = json{
        {"schema_version", 1},
        {"kind", "mesh"},
        {"created_at_utc", "2026-11-01T00:00:00Z"},
        {"route", "colmap_openmvs"},
        {"config_path", "tools/mesh/reconstruct/configs/canonical_colmap_openmvs.yaml"},
        {"config_sha256", kSha},
        {"frame", "W"},
        {"units", "m"},
        {"files", {{"ply", FileEntry(fixture.root, ply, "ply", 3U, 1U)},
                   {"glb", FileEntry(fixture.root, glb, "glb", 3U, 1U)},
                   {"obj", FileEntry(fixture.root, obj, "obj", 3U, 1U)}}},
        {"vertex_count", 3U},
        {"triangle_count", 1U},
        {"bbox_min_m", {0.0, 0.0, 0.0}},
        {"bbox_max_m", {1.0, 1.0, 0.5}},
        {"components", {{"count", 1U}, {"largest_fraction", 1.0}}},
        {"degenerate", {{"count", 0U}, {"fraction", 0.0}}},
        {"decimation", {{"canonical_decimated", false}, {"method", "none"}}},
        {"viewer", {{"enabled", true},
                    {"path", viewer.filename().string()},
                    {"vertex_count", 3U},
                    {"triangle_count", 1U},
                    {"sha256", Sha256File(viewer)}}},
        {"inputs", {{"bundle_root", "/data/bundle"},
                    {"frames", {4615}},
                    {"image_camera", "image_00"},
                    {"source_mesh_sha256", kSha},
                    {"dense_cloud_sha256", kSha}}},
        {"dense_backend", "colmap_cuda"},
        {"tool_versions", {{"colmap", "4.2.1"}, {"openmvs", "2.4.0"}}},
        {"git_sha", kSha},
        {"g3", {{"pass", false}, {"checks", json::object()}}}};
    return fixture;
}

void ExpectFailure(const std::function<void()> &call, const int exitCode,
                   const std::string &needle = {})
{
    try
    {
        call();
    }
    catch (const IoError &error)
    {
        INFO(error.what());
        CHECK(error.exitCode() == exitCode);
        if (!needle.empty())
        {
            CHECK(std::string(error.what()).find(needle) != std::string::npos);
        }
        return;
    }
    FAIL("expected raylidar::IoError");
}
} // namespace

TEST_CASE("a complete manifest and its files validate", "[mesh][manifest]")
{
    Fixture fixture = MakeValidFixture("mesh_manifest_ok");
    fixture.write();

    const MeshManifestValidation result =
        ValidateMeshManifest(fixture.manifestPath(), fixture.root, false);
    CHECK(result.ok);
    CHECK(result.vertex_count == 3U);
    CHECK(result.triangle_count == 1U);
    CHECK(result.bbox_min.isApprox(Eigen::Vector3d(0.0, 0.0, 0.0)));
    CHECK(result.bbox_max.isApprox(Eigen::Vector3d(1.0, 1.0, 0.5)));
}

TEST_CASE("hash verification accepts matching files and rejects drift",
          "[mesh][manifest]")
{
    Fixture fixture = MakeValidFixture("mesh_manifest_hash");
    fixture.write();

    CHECK_NOTHROW(ValidateMeshManifest(fixture.manifestPath(), fixture.root, true));

    // Declared hashes are trusted without --verify-hashes; a mismatch is exit 6
    // once verification is requested.
    fixture.manifest["files"]["ply"]["sha256"] = std::string(64U, '0');
    fixture.write();
    CHECK_NOTHROW(ValidateMeshManifest(fixture.manifestPath(), fixture.root, false));
    ExpectFailure(
        [&fixture] { (void)ValidateMeshManifest(fixture.manifestPath(), fixture.root, true); },
        6, "SHA-256");
}

TEST_CASE("manifest schema violations are invalid input", "[mesh][manifest]")
{
    SECTION("missing components")
    {
        Fixture fixture = MakeValidFixture("mesh_manifest_missing");
        fixture.manifest.erase("components");
        fixture.write();
        ExpectFailure(
            [&fixture] { (void)ValidateMeshManifest(fixture.manifestPath(), fixture.root, false); },
            4, "components");
    }
    SECTION("wrong frame")
    {
        Fixture fixture = MakeValidFixture("mesh_manifest_frame");
        fixture.manifest["frame"] = "C";
        fixture.write();
        ExpectFailure(
            [&fixture] { (void)ValidateMeshManifest(fixture.manifestPath(), fixture.root, false); },
            4, "frame");
    }
    SECTION("absolute path")
    {
        Fixture fixture = MakeValidFixture("mesh_manifest_abs");
        fixture.manifest["files"]["ply"]["path"] = "/etc/mesh.ply";
        fixture.write();
        ExpectFailure(
            [&fixture] { (void)ValidateMeshManifest(fixture.manifestPath(), fixture.root, false); },
            4, "relative");
    }
    SECTION("parent traversal")
    {
        Fixture fixture = MakeValidFixture("mesh_manifest_dotdot");
        fixture.manifest["files"]["obj"]["path"] = "../mesh.obj";
        fixture.write();
        ExpectFailure(
            [&fixture] { (void)ValidateMeshManifest(fixture.manifestPath(), fixture.root, false); },
            4, "..");
    }
    SECTION("bad sha256 encoding")
    {
        Fixture fixture = MakeValidFixture("mesh_manifest_badsha");
        fixture.manifest["files"]["glb"]["sha256"] = "not-a-sha";
        fixture.write();
        ExpectFailure(
            [&fixture] { (void)ValidateMeshManifest(fixture.manifestPath(), fixture.root, false); },
            4, "SHA-256");
    }
    SECTION("largest_fraction above one")
    {
        Fixture fixture = MakeValidFixture("mesh_manifest_fraction");
        fixture.manifest["components"]["largest_fraction"] = 1.5;
        fixture.write();
        ExpectFailure(
            [&fixture] { (void)ValidateMeshManifest(fixture.manifestPath(), fixture.root, false); },
            4, "largest_fraction");
    }
    SECTION("invalid JSON")
    {
        const fs::path root = MakeIoTempRoot("mesh_manifest_json");
        const fs::path path = WriteTempText(root, "mesh_manifest.json", "{not json");
        ExpectFailure([&path, &root] { (void)ValidateMeshManifest(path, root, false); }, 4,
                      "invalid JSON");
    }
}

TEST_CASE("declared counts and bbox must match the files", "[mesh][manifest]")
{
    SECTION("canonical vertex count drift")
    {
        Fixture fixture = MakeValidFixture("mesh_manifest_count");
        fixture.manifest["vertex_count"] = 4U;
        fixture.write();
        ExpectFailure(
            [&fixture] { (void)ValidateMeshManifest(fixture.manifestPath(), fixture.root, false); },
            6, "canonical counts");
    }
    SECTION("file entry count drift")
    {
        Fixture fixture = MakeValidFixture("mesh_manifest_file_count");
        fixture.manifest["files"]["glb"]["triangle_count"] = 2U;
        fixture.write();
        ExpectFailure(
            [&fixture] { (void)ValidateMeshManifest(fixture.manifestPath(), fixture.root, false); },
            6, "declares");
    }
    SECTION("bbox drift")
    {
        Fixture fixture = MakeValidFixture("mesh_manifest_bbox");
        fixture.manifest["bbox_max_m"] = {2.0, 1.0, 0.5};
        fixture.write();
        ExpectFailure(
            [&fixture] { (void)ValidateMeshManifest(fixture.manifestPath(), fixture.root, false); },
            6, "bbox");
    }
    SECTION("viewer count drift")
    {
        Fixture fixture = MakeValidFixture("mesh_manifest_viewer");
        fixture.manifest["viewer"]["triangle_count"] = 2U;
        fixture.write();
        ExpectFailure(
            [&fixture] { (void)ValidateMeshManifest(fixture.manifestPath(), fixture.root, false); },
            6, "declares");
    }
}
