#include "fixture.h"

#include <raylidar/cli/commands.h>
#include <raylidar/common/io.h>
#include <raylidar/dataset/acquire.h>

#include <archive.h>
#include <archive_entry.h>
#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include <stb_image_write.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::ReadJson;
using raylidar::dataset::AcquisitionError;
using raylidar::dataset::ExtractResult;
using raylidar::dataset::SafeExtract;
using raylidar::test::DatasetTreeOptions;
using raylidar::test::MakeDatasetTree;
using raylidar::test::MakeTempRoot;

constexpr const char *kSequence = "2013_05_28_drive_0000_sync";
const std::vector<std::string> kCameras = {"image_00", "image_01"};

struct Member
{
    std::string path;
    std::string data;
    std::string symlink;
    bool directory = false;
};

fs::path WriteArchive(const fs::path &path, const std::vector<Member> &members, bool zip)
{
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
    struct archive *writer = archive_write_new();
    REQUIRE(writer != nullptr);
    if (zip)
    {
        archive_write_set_format_zip(writer);
    }
    else
    {
        archive_write_set_format_pax_restricted(writer);
    }
    REQUIRE(archive_write_open_filename(writer, path.string().c_str()) == ARCHIVE_OK);
    for (const Member &member : members)
    {
        archive_entry *entry = archive_entry_new();
        archive_entry_set_pathname(entry, member.path.c_str());
        if (!member.symlink.empty())
        {
            archive_entry_set_filetype(entry, AE_IFLNK);
            archive_entry_set_perm(entry, 0777);
            archive_entry_set_symlink(entry, member.symlink.c_str());
            archive_entry_set_size(entry, 0);
        }
        else if (member.directory)
        {
            archive_entry_set_filetype(entry, AE_IFDIR);
            archive_entry_set_perm(entry, 0755);
            archive_entry_set_size(entry, 0);
        }
        else
        {
            archive_entry_set_filetype(entry, AE_IFREG);
            archive_entry_set_perm(entry, 0644);
            archive_entry_set_size(entry,
                                   static_cast<la_int64_t>(member.data.size()));
        }
        REQUIRE(archive_write_header(writer, entry) == ARCHIVE_OK);
        if (!member.data.empty())
        {
            REQUIRE(archive_write_data(writer, member.data.data(), member.data.size()) ==
                    static_cast<la_ssize_t>(member.data.size()));
        }
        archive_entry_free(entry);
    }
    REQUIRE(archive_write_close(writer) == ARCHIVE_OK);
    archive_write_free(writer);
    return path;
}

void PngSink(void *context, void *data, int size)
{
    auto *output = static_cast<std::vector<unsigned char> *>(context);
    const auto *bytes = static_cast<const unsigned char *>(data);
    output->insert(output->end(), bytes, bytes + size);
}

std::vector<unsigned char> MakePng(int width, int height)
{
    std::vector<unsigned char> pixels(static_cast<std::size_t>(width) *
                                          static_cast<std::size_t>(height),
                                      128U);
    std::vector<unsigned char> bytes;
    REQUIRE(stbi_write_png_to_func(PngSink, &bytes, width, height, 1, pixels.data(), width) !=
            0);
    REQUIRE_FALSE(bytes.empty());
    return bytes;
}

void WriteFile(const fs::path &path, std::string_view text)
{
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    REQUIRE(output.good());
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
}

void WritePngFile(const fs::path &path, int width, int height)
{
    const std::vector<unsigned char> bytes = MakePng(width, height);
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    REQUIRE(output.good());
    output.write(reinterpret_cast<const char *>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
}

int RunCli(const std::vector<std::string> &arguments, std::string *out = nullptr,
           std::string *err = nullptr)
{
    std::vector<std::string_view> views;
    views.reserve(arguments.size());
    for (const std::string &argument : arguments)
    {
        views.emplace_back(argument);
    }

    std::ostringstream outBuffer;
    std::ostringstream errBuffer;
    std::streambuf *oldOut = std::cout.rdbuf(outBuffer.rdbuf());
    std::streambuf *oldErr = std::cerr.rdbuf(errBuffer.rdbuf());
    const int code = RunDataset(views);
    std::cout.rdbuf(oldOut);
    std::cerr.rdbuf(oldErr);
    if (out != nullptr)
    {
        *out = outBuffer.str();
    }
    if (err != nullptr)
    {
        *err = errBuffer.str();
    }
    return code;
}

fs::path WriteConfig(const fs::path &path, const fs::path &root, const fs::path &staging,
                     bool acceptReuse, const std::map<std::string, std::string> &links)
{
    std::ostringstream yaml;
    yaml << "dataset:\n"
         << "  root: " << root.string() << "\n"
         << "  staging_dir: " << staging.string() << "\n"
         << "  sequence: " << kSequence << "\n"
         << "  cameras: [image_00, image_01]\n"
         << "  accept_reuse_after_g1: " << (acceptReuse ? "true" : "false") << "\n"
         << "  link_sources:\n";
    for (const char *key :
         {"image_00", "image_01", "lidar", "image_00_timestamps", "image_01_timestamps"})
    {
        const auto found = links.find(key);
        yaml << "    " << key << ": \""
             << (found == links.end() ? std::string() : found->second) << "\"\n";
    }
    WriteFile(path, yaml.str());
    return path;
}

std::vector<std::string> InstallArgs(const fs::path &config, const fs::path &manifest,
                                     const std::string &policy)
{
    return {"install", "--config", config.string(), "--manifest", manifest.string(),
            "--policy", policy};
}

std::string ReadFileText(const fs::path &path)
{
    std::ifstream input(path, std::ios::binary);
    REQUIRE(input.good());
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

/// Root that already has the official subset installed and whose LiDAR/timestamp
/// links resolve into an external source tree (mirrors the real /data-20T layout).
struct OfficialSetup
{
    fs::path base;
    fs::path root;
    fs::path staging;
    fs::path config;
    std::map<std::string, std::string> links;
};

OfficialSetup MakeOfficialSetup(std::string_view name)
{
    OfficialSetup setup;
    setup.base = MakeTempRoot(name);
    setup.root = setup.base / "root";
    setup.staging = setup.base / "staging";
    fs::create_directories(setup.staging);
    MakeDatasetTree(setup.root, {});

    const fs::path sources = setup.base / "sources";
    WriteFile(sources / "ts_00.txt", "0 0.0\n");
    WriteFile(sources / "ts_01.txt", "0 0.0\n");

    setup.links["image_00"] = "";
    setup.links["image_01"] = "";
    setup.links["lidar"] = (setup.root / "data_3d_raw" / kSequence / "velodyne_points").string();
    setup.links["image_00_timestamps"] = (sources / "ts_00.txt").string();
    setup.links["image_01_timestamps"] = (sources / "ts_01.txt").string();

    std::error_code error;
    fs::create_directory_symlink(sources / "ts_00.txt",
                                 setup.root / "data_2d_raw" / kSequence / "image_00" /
                                     "timestamps.txt",
                                 error);
    fs::create_directory_symlink(sources / "ts_01.txt",
                                 setup.root / "data_2d_raw" / kSequence / "image_01" /
                                     "timestamps.txt",
                                 error);

    setup.config = WriteConfig(setup.base / "dataset.yaml", setup.root, setup.staging,
                               false, setup.links);
    return setup;
}

/// Official tree with the images and LiDAR moved to an external source tree.
struct ReuseSetup
{
    fs::path base;
    fs::path root;
    fs::path staging;
    fs::path config;
    std::map<std::string, std::string> links;
};

ReuseSetup MakeReuseSetup(std::string_view name, bool imagesPresent)
{
    ReuseSetup setup;
    setup.base = MakeTempRoot(name);
    setup.root = setup.base / "root";
    setup.staging = setup.base / "staging";
    fs::create_directories(setup.staging);

    DatasetTreeOptions options;
    if (!imagesPresent)
    {
        options.missing.insert("images");
    }
    MakeDatasetTree(setup.root, options);

    const fs::path sources = setup.base / "sources";
    if (imagesPresent)
    {
        fs::create_directories(sources);
        fs::rename(setup.root / "data_2d_raw" / kSequence / "image_00" / "data_rect",
                   sources / "image_00");
        fs::rename(setup.root / "data_2d_raw" / kSequence / "image_01" / "data_rect",
                   sources / "image_01");
    }
    else
    {
        for (const std::string &camera : kCameras)
        {
            for (int frame = 0; frame < 2; ++frame)
            {
                char name[16];
                std::snprintf(name, sizeof(name), "%010d.png", frame);
                WritePngFile(sources / camera / name, 1408, 376);
            }
        }
    }

    WriteFile(sources / "ts_00.txt", "0 0.0\n");
    WriteFile(sources / "ts_01.txt", "0 0.0\n");

    setup.links["image_00"] = (sources / "image_00").string();
    setup.links["image_01"] = (sources / "image_01").string();
    setup.links["lidar"] = (setup.root / "data_3d_raw" / kSequence / "velodyne_points").string();
    setup.links["image_00_timestamps"] = (sources / "ts_00.txt").string();
    setup.links["image_01_timestamps"] = (sources / "ts_01.txt").string();

    setup.config = WriteConfig(setup.base / "dataset.yaml", setup.root, setup.staging,
                               false, setup.links);
    return setup;
}
} // namespace

TEST_CASE("SafeExtract rejects zip-slip members before writing", "[dataset][acquire]")
{
    const fs::path base = MakeTempRoot("acquire_slip");
    const fs::path archive =
        WriteArchive(base / "evil.tar", {{"../evil.txt", "owned"}}, false);
    const fs::path dest = base / "dest";

    bool threw = false;
    try
    {
        SafeExtract(archive, dest, {"calibration"});
    }
    catch (const AcquisitionError &error)
    {
        threw = true;
        CHECK(error.exitCode() == 8);
    }
    CHECK(threw);
    CHECK_FALSE(fs::exists(base / "evil.txt"));
    CHECK_FALSE(fs::exists(dest / "evil.txt"));
}

TEST_CASE("SafeExtract rejects absolute and symlink members", "[dataset][acquire]")
{
    const fs::path base = MakeTempRoot("acquire_absolute");

    const fs::path absolute =
        WriteArchive(base / "absolute.tar", {{"/tmp/raylidar_evil.txt", "owned"}}, false);
    CHECK_THROWS_AS(SafeExtract(absolute, base / "dest_a", {"calibration"}),
                    AcquisitionError);

    const fs::path symlinked =
        WriteArchive(base / "symlink.tar", {{"link", "", "/etc/passwd"}}, false);
    bool threw = false;
    try
    {
        SafeExtract(symlinked, base / "dest_b", {"link"});
    }
    catch (const AcquisitionError &error)
    {
        threw = true;
        CHECK(error.exitCode() == 8);
    }
    CHECK(threw);
    CHECK_FALSE(fs::exists(base / "dest_b" / "link"));
}

TEST_CASE("SafeExtract installs only the allowlisted sequence", "[dataset][acquire]")
{
    const fs::path base = MakeTempRoot("acquire_allowlist");
    const fs::path archive = WriteArchive(
        base / "data_poses.zip",
        {{std::string(kSequence) + "/poses.txt", "selected"},
         {"2013_05_28_drive_0001_sync/poses.txt", "other"}},
        true);
    const fs::path dest = base / "data_poses";

    const ExtractResult result = SafeExtract(archive, dest, {kSequence});
    CHECK(result.skippedMembers == 1U);
    CHECK(result.extracted.size() == 1U);
    CHECK(ReadFileText(dest / kSequence / "poses.txt") == "selected");
    CHECK_FALSE(fs::exists(dest / "2013_05_28_drive_0001_sync"));
}

TEST_CASE("SafeExtract keeps members under a data_2d_semantics root",
          "[dataset][acquire]")
{
    const fs::path base = MakeTempRoot("acquire_semantics_root");
    const std::string prefix = "data_2d_semantics/train/" + std::string(kSequence);
    const fs::path archive = WriteArchive(
        base / "data_2d_semantics.zip",
        {{prefix + "/image_00/semantic/0000000000.png", "png"}}, true);
    const fs::path dest = base / "root";

    const ExtractResult result = SafeExtract(archive, dest, {prefix});
    CHECK(result.skippedMembers == 0U);
    CHECK(result.extracted.size() == 1U);
    CHECK(fs::exists(dest / prefix / "image_00" / "semantic" / "0000000000.png"));
}

TEST_CASE("SafeExtract strips a single KITTI-360 wrapper segment", "[dataset][acquire]")
{
    const fs::path base = MakeTempRoot("acquire_wrapper");
    const fs::path archive = WriteArchive(
        base / "calibration.zip",
        {{"KITTI-360/calibration/calib_cam_to_pose.txt", "pose"},
         {"calibration/calib_cam_to_velo.txt", "velo"}},
        true);
    const fs::path dest = base / "root";

    SafeExtract(archive, dest, {"calibration"});
    CHECK(ReadFileText(dest / "calibration" / "calib_cam_to_pose.txt") == "pose");
    CHECK(ReadFileText(dest / "calibration" / "calib_cam_to_velo.txt") == "velo");
    CHECK_FALSE(fs::exists(dest / "KITTI-360"));
}

TEST_CASE("SafeExtract guards on allowlisted bytes, not archive size",
          "[dataset][acquire]")
{
    const fs::path base = MakeTempRoot("acquire_disk_guard");
    const std::string decoy(1024U * 1024U, 'x');
    const fs::path archive = WriteArchive(
        base / "calibration.zip",
        {{"decoy/huge.bin", decoy}, {"calibration/small.txt", "0123456789"}}, true);
    const fs::path dest = base / "root";

    // 10 allowlisted bytes needs 25 bytes at 2.5x; 30 free passes even though the
    // archive holds roughly 1 MiB.
    CHECK_NOTHROW(SafeExtract(archive, dest, {"calibration"}, false, 120'000'000'000ULL,
                              1'000'000U, 2.5, 0U, std::uintmax_t{30}));
    CHECK(fs::exists(dest / "calibration" / "small.txt"));
    CHECK_FALSE(fs::exists(dest / "decoy"));

    bool threw = false;
    try
    {
        SafeExtract(archive, base / "root2", {"calibration"}, false, 120'000'000'000ULL,
                    1'000'000U, 2.5, 0U, std::uintmax_t{0});
    }
    catch (const AcquisitionError &error)
    {
        threw = true;
        CHECK(error.exitCode() == 8);
    }
    CHECK(threw);
}

TEST_CASE("SafeExtract stream mode writes only allowlisted members", "[dataset][acquire]")
{
    const fs::path base = MakeTempRoot("acquire_stream");
    const fs::path archive = WriteArchive(
        base / "data_poses.tar",
        {{std::string(kSequence) + "/poses.txt", "selected"},
         {"2013_05_28_drive_0001_sync/poses.txt", "other"}},
        false);
    const fs::path dest = base / "root";

    const ExtractResult result =
        SafeExtract(archive, dest, {kSequence}, true, 120'000'000'000ULL, 1'000'000U, 2.5,
                    10'000U);
    CHECK(result.extracted.size() == 1U);
    CHECK(result.skippedMembers == 1U);
    CHECK(fs::exists(dest / kSequence / "poses.txt"));

    CHECK_THROWS_AS(SafeExtract(archive, base / "root2", {kSequence}, true),
                    AcquisitionError);
}

TEST_CASE("Official mode validates installed data_rect and never creates links",
          "[dataset][acquire]")
{
    OfficialSetup setup = MakeOfficialSetup("acquire_official");
    const std::string archiveName = std::string(kSequence) + "_image_00.zip";
    const std::vector<unsigned char> png = MakePng(1408, 376);
    WriteArchive(setup.staging / archiveName,
                 {{std::string(kSequence) + "/image_00/data_rect/9999999999.png",
                   std::string(png.begin(), png.end())}},
                 true);

    const fs::path manifestPath = setup.base / "readiness.json";
    std::string error;
    CHECK(RunCli(InstallArgs(setup.config, manifestPath, "official"), nullptr, &error) == 0);

    CHECK_FALSE(fs::is_symlink(setup.root / "data_2d_raw" / kSequence / "image_00" /
                               "data_rect"));
    const nlohmann::json manifest = ReadJson(manifestPath);
    CHECK(manifest["image_provenance"] == "official_data_rect");
    CHECK(manifest["sequence"] == kSequence);
    CHECK(manifest["images"]["image_00"]["pre_existing"] == true);
    CHECK(manifest["images"]["image_00"]["width"] == 1408);
    CHECK(manifest["images"]["image_00"]["height"] == 376);
    CHECK(manifest["release"] == "KITTI-360");
    CHECK(manifest.contains("license"));

    bool skippedArchive = false;
    for (const auto &entry : manifest["skipped_archives"])
    {
        if (entry["name"] == archiveName && entry["reason"] == "already_installed")
        {
            skippedArchive = true;
        }
    }
    CHECK(skippedArchive);

    // A second run must be idempotent except for the timestamp.
    const fs::path secondPath = setup.base / "readiness2.json";
    CHECK(RunCli(InstallArgs(setup.config, secondPath, "official")) == 0);
    nlohmann::json first = ReadJson(manifestPath);
    nlohmann::json second = ReadJson(secondPath);
    first.erase("installed_at_utc");
    second.erase("installed_at_utc");
    CHECK(first == second);
}

TEST_CASE("Official mode rejects a wrong first-image size", "[dataset][acquire]")
{
    const fs::path base = MakeTempRoot("acquire_wrong_size");
    const fs::path root = base / "root";
    DatasetTreeOptions options;
    options.imageSize = {1242, 375};
    MakeDatasetTree(root, options);

    const fs::path staging = base / "staging";
    fs::create_directories(staging);
    const fs::path config = WriteConfig(base / "dataset.yaml", root, staging, false,
                                        {{"lidar", (root / "data_3d_raw" / kSequence /
                                                    "velodyne_points")
                                                       .string()}});

    std::string error;
    const int code = RunCli(InstallArgs(config, base / "readiness.json", "official"), nullptr,
                            &error);
    CHECK(code == 7);
    CHECK(error.find("image_00") != std::string::npos);
    CHECK(error.find("1242x375") != std::string::npos);
}

TEST_CASE("Official mode rejects unequal camera counts", "[dataset][acquire]")
{
    const fs::path base = MakeTempRoot("acquire_unequal");
    const fs::path root = base / "root";
    DatasetTreeOptions options;
    options.imageCounts = {2, 3};
    MakeDatasetTree(root, options);

    const fs::path staging = base / "staging";
    fs::create_directories(staging);
    const fs::path config =
        WriteConfig(base / "dataset.yaml", root, staging, false,
                    {{"lidar", (root / "data_3d_raw" / kSequence / "velodyne_points").string()}});

    std::string error;
    const int code =
        RunCli(InstallArgs(config, base / "readiness.json", "official"), nullptr, &error);
    CHECK(code == 7);
    CHECK(error.find("counts must match") != std::string::npos);
}

TEST_CASE("Reuse mode creates all five links and records the provenance",
          "[dataset][acquire]")
{
    ReuseSetup setup = MakeReuseSetup("acquire_reuse", true);
    const fs::path manifestPath = setup.base / "readiness.json";
    CHECK(RunCli(InstallArgs(setup.config, manifestPath, "reuse_local_after_g1")) == 0);

    for (const std::string &camera : kCameras)
    {
        CHECK(fs::is_symlink(setup.root / "data_2d_raw" / kSequence / camera / "data_rect"));
    }
    CHECK(fs::is_symlink(setup.root / "data_2d_raw" / kSequence / "image_00" /
                         "timestamps.txt"));
    CHECK(fs::is_symlink(setup.root / "data_2d_raw" / kSequence / "image_01" /
                         "timestamps.txt"));
    CHECK(fs::exists(setup.root / "data_3d_raw" / kSequence / "velodyne_points" / "data"));

    const nlohmann::json manifest = ReadJson(manifestPath);
    CHECK(manifest["image_provenance"] == "local_repackaged_1408x376");
    CHECK(manifest["links"].size() == 5U);
    CHECK(manifest["policy"] == "reuse_local_after_g1");
}

TEST_CASE("Reuse mode rejects a missing or unconfirmed link source",
          "[dataset][acquire]")
{
    ReuseSetup setup = MakeReuseSetup("acquire_reuse_missing", true);
    std::vector<std::string> arguments = InstallArgs(
        setup.config, setup.base / "readiness.json", "reuse_local_after_g1");
    arguments.push_back("--link-source");
    arguments.push_back("image_00_timestamps=REQUIRED_USER_CONFIRM");
    std::string error;
    CHECK(RunCli(arguments, nullptr, &error) == 4);
    CHECK(error.find("image_00_timestamps") != std::string::npos);

    ReuseSetup missing = MakeReuseSetup("acquire_reuse_missing2", true);
    std::vector<std::string> bad = InstallArgs(missing.config, missing.base / "readiness.json",
                                               "reuse_local_after_g1");
    bad.push_back("--link-source");
    bad.push_back("lidar=" + (missing.base / "does-not-exist").string());
    CHECK(RunCli(bad) == 4);
}

TEST_CASE("Reuse mode records an approved null timestamp source", "[dataset][acquire]")
{
    ReuseSetup setup = MakeReuseSetup("acquire_reuse_null", true);
    std::vector<std::string> arguments =
        InstallArgs(setup.config, setup.base / "readiness.json", "reuse_local_after_g1");
    arguments.push_back("--link-source");
    arguments.push_back("image_00_timestamps=");
    arguments.push_back("--link-source");
    arguments.push_back("image_01_timestamps=");
    CHECK(RunCli(arguments) == 0);

    const nlohmann::json manifest = ReadJson(setup.base / "readiness.json");
    bool skipped = false;
    for (const auto &link : manifest["links"])
    {
        if (link["key"] == "image_00_timestamps" && link["skipped"] == true &&
            link["skip_reason"] == "approved_null_source")
        {
            skipped = true;
        }
    }
    CHECK(skipped);
    CHECK_FALSE(fs::exists(setup.root / "data_2d_raw" / kSequence / "image_00" /
                           "timestamps.txt"));
}

TEST_CASE("A conflicting existing timestamp link is refused", "[dataset][acquire]")
{
    ReuseSetup setup = MakeReuseSetup("acquire_conflict", true);
    const fs::path other = setup.base / "other-ts.txt";
    WriteFile(other, "0 0.0\n");
    std::error_code error;
    fs::create_symlink(other,
                       setup.root / "data_2d_raw" / kSequence / "image_00" / "timestamps.txt",
                       error);
    REQUIRE_FALSE(error);

    std::string err;
    const int code = RunCli(
        InstallArgs(setup.config, setup.base / "readiness.json", "reuse_local_after_g1"),
        nullptr, &err);
    CHECK(code == 4);
    CHECK(err.find("image_00_timestamps") != std::string::npos);
}

TEST_CASE("Auto policy resolves to official when images validate", "[dataset][acquire]")
{
    OfficialSetup setup = MakeOfficialSetup("acquire_auto_official");
    CHECK(RunCli(InstallArgs(setup.config, setup.base / "readiness.json", "auto")) == 0);
    const nlohmann::json manifest = ReadJson(setup.base / "readiness.json");
    CHECK(manifest["policy"] == "official");
    CHECK(manifest["image_provenance"] == "official_data_rect");
}

TEST_CASE("Auto policy falls back only when reuse after G1 is accepted",
          "[dataset][acquire]")
{
    ReuseSetup setup = MakeReuseSetup("acquire_auto_reuse", false);
    setup.config = WriteConfig(setup.base / "dataset.yaml", setup.root, setup.staging, true,
                               setup.links);
    CHECK(RunCli(InstallArgs(setup.config, setup.base / "readiness.json", "auto")) == 0);
    const nlohmann::json manifest = ReadJson(setup.base / "readiness.json");
    CHECK(manifest["policy"] == "reuse_local_after_g1");

    ReuseSetup refused = MakeReuseSetup("acquire_auto_refused", false);
    std::string error;
    CHECK(RunCli(InstallArgs(refused.config, refused.base / "readiness.json", "auto"), nullptr,
                 &error) == 7);
    CHECK(error.find("official data_rect") != std::string::npos);
}

TEST_CASE("Missing calibration and KITTI-raw calibration both exit 7",
          "[dataset][acquire]")
{
    {
        const fs::path base = MakeTempRoot("acquire_missing_calib");
        const fs::path root = base / "root";
        DatasetTreeOptions options;
        options.missing.insert("calibration");
        MakeDatasetTree(root, options);
        const fs::path staging = base / "staging";
        fs::create_directories(staging);
        const fs::path config = WriteConfig(
            base / "dataset.yaml", root, staging, false,
            {{"lidar", (root / "data_3d_raw" / kSequence / "velodyne_points").string()}});

        std::string error;
        CHECK(RunCli(InstallArgs(config, base / "readiness.json", "official"), nullptr,
                     &error) == 7);
        CHECK(error.find("calib_cam_to_pose.txt") != std::string::npos);
    }
    {
        const fs::path base = MakeTempRoot("acquire_kitti_raw");
        const fs::path root = base / "root";
        DatasetTreeOptions options;
        options.missing.insert("calibration");
        options.kittiRawCalib = true;
        MakeDatasetTree(root, options);
        const fs::path staging = base / "staging";
        fs::create_directories(staging);
        const fs::path config = WriteConfig(
            base / "dataset.yaml", root, staging, false,
            {{"lidar", (root / "data_3d_raw" / kSequence / "velodyne_points").string()}});

        std::string error;
        CHECK(RunCli(InstallArgs(config, base / "readiness.json", "official"), nullptr,
                     &error) == 7);
        CHECK(error.find("KITTI-raw calibration") != std::string::npos);
    }
}

TEST_CASE("Install extracts allowlisted archives and writes the manifest",
          "[dataset][acquire]")
{
    const fs::path base = MakeTempRoot("acquire_end_to_end");
    const fs::path root = base / "root";
    const fs::path staging = base / "staging";
    const fs::path sources = base / "sources";
    fs::create_directories(root);
    fs::create_directories(staging);
    fs::create_directories(sources / "lidar" / "data");
    WriteFile(sources / "lidar" / "data" / "0000000000.bin", std::string(64U, '\0'));

    const std::vector<unsigned char> png = MakePng(1408, 376);
    const std::string pngText(png.begin(), png.end());

    WriteArchive(staging / "calibration.zip",
                 {{"calibration/calib_cam_to_pose.txt", "image_00: 1 0 0 0\n"},
                  {"calibration/calib_cam_to_velo.txt", "1 0 0 0\n"},
                  {"calibration/perspective.txt", "P_rect_00: 1 0 0 0\n"},
                  {"calibration/calib_sick_to_velo.txt", "1 0 0 0\n"}},
                 true);
    WriteArchive(staging / "data_poses.zip",
                 {{std::string(kSequence) + "/poses.txt", "0 1 0 0 0 0 1 0 0 0 0 1 0\n"},
                  {std::string(kSequence) + "/cam0_to_world.txt",
                   "0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n"},
                  {"2013_05_28_drive_0001_sync/poses.txt", "0 1 0 0 0 0 1 0 0 0 0 1 0\n"}},
                 true);
    WriteArchive(staging / "data_2d_semantics.zip",
                 {{"data_2d_semantics/train/" + std::string(kSequence) +
                       "/image_00/semantic/0000000000.png",
                   pngText},
                  {"data_2d_semantics/train/other/image_00/semantic/0000000000.png", pngText}},
                 true);
    WriteArchive(staging / "data_2d_semantics_image_01.zip",
                 {{"data_2d_semantics/train/" + std::string(kSequence) +
                       "/image_01/semantic/0000000000.png",
                   pngText}},
                 true);
    WriteArchive(staging / (std::string(kSequence) + "_image_00.zip"),
                 {{std::string(kSequence) + "/image_00/data_rect/0000000000.png", pngText}},
                 true);
    WriteArchive(staging / (std::string(kSequence) + "_image_01.zip"),
                 {{std::string(kSequence) + "/image_01/data_rect/0000000000.png", pngText}},
                 true);

    WriteFile(sources / "ts_00.txt", "0 0.0\n");
    WriteFile(sources / "ts_01.txt", "0 0.0\n");
    const std::map<std::string, std::string> links = {
        {"lidar", (sources / "lidar").string()},
        {"image_00_timestamps", (sources / "ts_00.txt").string()},
        {"image_01_timestamps", (sources / "ts_01.txt").string()},
    };
    const fs::path config = WriteConfig(base / "dataset.yaml", root, staging, false, links);
    const fs::path manifestPath = base / "readiness.json";

    CHECK(RunCli(InstallArgs(config, manifestPath, "official")) == 0);

    CHECK(fs::exists(root / "calibration" / "calib_cam_to_pose.txt"));
    CHECK(fs::exists(root / "calibration" / "calib_sick_to_velo.txt"));
    CHECK(fs::exists(root / "data_poses" / kSequence / "poses.txt"));
    CHECK(fs::exists(root / "data_2d_semantics" / "train" / kSequence / "image_00" /
                     "semantic"));
    CHECK(fs::exists(root / "data_2d_raw" / kSequence / "image_01" / "data_rect" /
                     "0000000000.png"));
    CHECK_FALSE(fs::exists(root / "data_poses" / "2013_05_28_drive_0001_sync"));
    CHECK_FALSE(fs::exists(root / "data_2d_semantics" / "train" / "other"));

    const nlohmann::json manifest = ReadJson(manifestPath);
    CHECK(manifest["schema_version"] == 1);
    CHECK(manifest["release"] == "KITTI-360");
    CHECK(manifest["license"] == "KITTI-360 non-commercial research use");
    CHECK(manifest["sequence"] == kSequence);
    CHECK(manifest["policy"] == "official");
    CHECK(manifest["image_provenance"] == "official_data_rect");
    CHECK(manifest["archives"].size() == 6U);
    CHECK(manifest["archives"][0].contains("sha256"));
    CHECK(manifest["archives"][0].contains("granularity"));
    CHECK(manifest["archives"][0].contains("sha256_source"));
    CHECK(manifest["components"].contains("calibration"));
    CHECK(manifest["links"].size() == 3U);
    CHECK(manifest["skipped_members"].get<std::size_t>() >= 2U);
    CHECK(manifest["images"]["image_00"]["pre_existing"] == false);
}

TEST_CASE("dataset install dry-run reports the resolved plan", "[dataset][acquire]")
{
    OfficialSetup setup = MakeOfficialSetup("acquire_dry_run");
    std::string out;
    std::vector<std::string> arguments = InstallArgs(setup.config, setup.base / "readiness.json",
                                                     "auto");
    arguments.push_back("--dry-run");
    CHECK(RunCli(arguments, &out, nullptr) == 0);
    CHECK(out.find("dataset install plan") != std::string::npos);
    CHECK(out.find("sequence: " + std::string(kSequence)) != std::string::npos);
    CHECK_FALSE(fs::exists(setup.base / "readiness.json"));
}
