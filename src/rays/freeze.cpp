#include <raylidar/rays/freeze.h>

#include <raylidar/common/io.h>
#include <raylidar/rays/rays.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace raylidar::rays
{
namespace
{
namespace fs = std::filesystem;

constexpr int kExitInvalidInput = 4;
constexpr int kExitGateFailed = 6;
constexpr std::array<const char *, 4> kFileKeys = {"ray", "primary_mask", "secondary_mask",
                                                   "primary_unoccluded_mask"};
constexpr std::array<const char *, 4> kBucketKeys = {"0_20", "20_40", "40_50", "50_60"};

std::string FrameName(const int frame)
{
    char buffer[16] = {};
    std::snprintf(buffer, sizeof(buffer), "%010d", frame);
    return std::string(buffer);
}

const nlohmann::json &RequireObject(const nlohmann::json &document, const char *key,
                                    const std::string &context)
{
    if (!document.contains(key) || !document.at(key).is_object())
    {
        throw IoError(context + ": missing object '" + key + "'", kExitInvalidInput);
    }
    return document.at(key);
}

const nlohmann::json &RequireArray(const nlohmann::json &document, const char *key,
                                   const std::string &context)
{
    if (!document.contains(key) || !document.at(key).is_array())
    {
        throw IoError(context + ": missing array '" + key + "'", kExitInvalidInput);
    }
    return document.at(key);
}

std::string RequireString(const nlohmann::json &document, const char *key,
                          const std::string &context)
{
    if (!document.contains(key) || !document.at(key).is_string())
    {
        throw IoError(context + ": missing string '" + key + "'", kExitInvalidInput);
    }
    return document.at(key).get<std::string>();
}

std::uint64_t RequireCount(const nlohmann::json &document, const char *key,
                           const std::string &context)
{
    if (!document.contains(key) || !document.at(key).is_number_integer())
    {
        throw IoError(context + ": missing integer count '" + key + "'", kExitInvalidInput);
    }
    const auto value = document.at(key).get<std::int64_t>();
    if (value < 0)
    {
        throw IoError(context + ": negative count '" + key + "'", kExitInvalidInput);
    }
    return static_cast<std::uint64_t>(value);
}

std::string FileNameFor(const nlohmann::json &entry, const char *key, const std::string &context)
{
    const nlohmann::json &files = RequireObject(entry, "files", context);
    if (!files.contains(key) || !files.at(key).is_string())
    {
        throw IoError(context + ": missing file name for '" + key + "'", kExitInvalidInput);
    }
    return files.at(key).get<std::string>();
}

std::string StoredHash(const nlohmann::json &entry, const char *key, const std::string &context)
{
    const nlohmann::json &hashes = RequireObject(entry, "sha256", context);
    if (!hashes.contains(key) || !hashes.at(key).is_string())
    {
        throw IoError(context + ": missing sha256 for '" + key + "'", kExitInvalidInput);
    }
    return hashes.at(key).get<std::string>();
}

std::vector<int> ReadWindow(const nlohmann::json &manifest, const std::string &context)
{
    const nlohmann::json &window = RequireArray(manifest, "window", context);
    if (window.empty())
    {
        throw IoError(context + ": window is empty", kExitInvalidInput);
    }
    std::vector<int> frames;
    frames.reserve(window.size());
    for (const nlohmann::json &value : window)
    {
        if (!value.is_number_integer())
        {
            throw IoError(context + ": window entries must be integers", kExitInvalidInput);
        }
        frames.push_back(value.get<int>());
    }
    return frames;
}

nlohmann::json PairsToObject(const std::vector<std::pair<int, std::string>> &pairs)
{
    nlohmann::json object = nlohmann::json::object();
    for (const auto &[frame, hash] : pairs)
    {
        object[std::to_string(frame)] = hash;
    }
    return object;
}

std::vector<std::pair<int, std::string>> ReadHashObject(const nlohmann::json &document,
                                                        const char *key,
                                                        const std::string &context)
{
    const nlohmann::json &object = RequireObject(document, key, context);
    std::vector<std::pair<int, std::string>> pairs;
    pairs.reserve(object.size());
    for (auto it = object.begin(); it != object.end(); ++it)
    {
        if (!it.value().is_string())
        {
            throw IoError(context + ": '" + key + "' entries must be strings", kExitInvalidInput);
        }
        int frame = 0;
        try
        {
            std::size_t consumed = 0;
            frame = std::stoi(it.key(), &consumed);
            if (consumed != it.key().size())
            {
                throw std::invalid_argument("trailing characters");
            }
        }
        catch (const std::exception &)
        {
            throw IoError(context + ": '" + key + "' has a non-integer frame key", kExitInvalidInput);
        }
        pairs.emplace_back(frame, it.value().get<std::string>());
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
    return pairs;
}

std::string CombinedFrameHash(const std::string &ray, const std::string &primary,
                              const std::string &secondary, const std::string &unoccluded)
{
    return Sha256String(ray + primary + secondary + unoccluded);
}

nlohmann::json BuildPayload(const std::string &sequence, const std::vector<int> &window,
                            const std::string &configSha, const std::string &codeVersion,
                            const std::string &manifestSha,
                            const std::vector<std::pair<int, std::string>> &frameHashes,
                            const std::vector<std::pair<int, std::string>> &metaHashes)
{
    nlohmann::json payload = nlohmann::json::object();
    payload["schema_version"] = 1;
    payload["sequence"] = sequence;
    payload["window"] = window;
    payload["config_sha256"] = configSha;
    payload["code_version"] = codeVersion;
    payload["manifest_sha256"] = manifestSha;
    payload["per_frame_sha256"] = PairsToObject(frameHashes);
    payload["per_frame_meta_sha256"] = PairsToObject(metaHashes);
    return payload;
}

std::vector<std::uint8_t> ReadAllBytes(const fs::path &path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input.good())
    {
        throw IoError("cannot read mask file " + path.string(), kExitInvalidInput);
    }
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(input),
                                     std::istreambuf_iterator<char>());
}

struct DerivedCounts
{
    std::uint64_t nRays = 0;
    std::array<std::uint64_t, 4> buckets{0U, 0U, 0U, 0U};
    std::array<std::uint64_t, 2> cameras{0U, 0U};
    std::uint64_t occluded = 0;
    std::uint64_t occludedExcluded = 0;
};

void Compare(const std::string &label, const std::uint64_t expected,
             const std::uint64_t actual, std::vector<std::string> &mismatches)
{
    if (expected != actual)
    {
        mismatches.push_back(label + ": stored=" + std::to_string(expected) +
                             " derived=" + std::to_string(actual));
    }
}

DerivedCounts DeriveCounts(const fs::path &directory, const int frame,
                           const nlohmann::json &entry, const fs::path &metaPath,
                           std::vector<std::string> &mismatches)
{
    const std::string context = "frame " + std::to_string(frame);

    std::vector<Ray> rays;
    int headerFrame = -1;
    try
    {
        rays = ReadRayFrame(directory / FileNameFor(entry, "ray", context), &headerFrame);
    }
    catch (const std::exception &error)
    {
        mismatches.push_back(context + ": cannot re-read ray file: " + error.what());
        return {};
    }
    if (headerFrame != frame)
    {
        mismatches.push_back(context + ": ray header frame=" + std::to_string(headerFrame) +
                             " manifest frame=" + std::to_string(frame));
    }

    DerivedCounts derived;
    derived.nRays = rays.size();
    for (const Ray &ray : rays)
    {
        const std::size_t bucket = static_cast<std::size_t>(ray.bucket);
        if (bucket >= derived.buckets.size())
        {
            mismatches.push_back(context + ": ray bucket is out of range");
            continue;
        }
        ++derived.buckets[bucket];
        if (ray.camera < derived.cameras.size())
        {
            ++derived.cameras[ray.camera];
        }
        else
        {
            mismatches.push_back(context + ": ray camera is out of range");
        }
        if (ray.occluded != 0U)
        {
            ++derived.occluded;
        }
        if (ray.bucket != static_cast<std::uint8_t>(Bucket::B50_60) && ray.occluded == 0U)
        {
            ++derived.occludedExcluded;
        }
    }

    // Re-derive the mask one-counts and require the masks to match the rays.
    const std::array<const char *, 3> maskKeys = {"primary_mask", "secondary_mask",
                                                  "primary_unoccluded_mask"};
    std::array<std::uint64_t, 3> maskOnes{0U, 0U, 0U};
    for (std::size_t index = 0; index < maskKeys.size(); ++index)
    {
        const fs::path maskPath = directory / FileNameFor(entry, maskKeys[index], context);
        const std::vector<std::uint8_t> bytes = ReadAllBytes(maskPath);
        if (bytes.size() != derived.nRays)
        {
            mismatches.push_back(context + ": " + maskKeys[index] + " length=" +
                                 std::to_string(bytes.size()) + " rays=" +
                                 std::to_string(derived.nRays));
        }
        for (const std::uint8_t value : bytes)
        {
            if (value > 1U)
            {
                mismatches.push_back(context + ": " + maskKeys[index] + " has a value > 1");
                break;
            }
            maskOnes[index] += value == 1U ? 1U : 0U;
        }
    }
    Compare(context + " primary_mask ones", derived.nRays - derived.buckets[3],
            maskOnes[0], mismatches);
    Compare(context + " secondary_mask ones", derived.buckets[3], maskOnes[1], mismatches);
    Compare(context + " primary_unoccluded_mask ones", derived.occludedExcluded, maskOnes[2],
            mismatches);

    // Manifest counts.
    Compare(context + " n_rays", RequireCount(entry, "n_rays", context), derived.nRays,
            mismatches);
    Compare(context + " full_count", RequireCount(entry, "full_count", context), derived.nRays,
            mismatches);
    const nlohmann::json &buckets = RequireObject(entry, "bucket_counts", context);
    for (std::size_t index = 0; index < kBucketKeys.size(); ++index)
    {
        Compare(context + " bucket_counts." + kBucketKeys[index],
                RequireCount(buckets, kBucketKeys[index], context), derived.buckets[index],
                mismatches);
    }
    const nlohmann::json &cameras = RequireObject(entry, "camera_counts", context);
    Compare(context + " camera_counts.image_00", RequireCount(cameras, "image_00", context),
            derived.cameras[0], mismatches);
    Compare(context + " camera_counts.image_01", RequireCount(cameras, "image_01", context),
            derived.cameras[1], mismatches);
    Compare(context + " occluded_count", RequireCount(entry, "occluded_count", context),
            derived.occluded, mismatches);
    Compare(context + " occluded_excluded_count",
            RequireCount(entry, "occluded_excluded_count", context), derived.occludedExcluded,
            mismatches);

    // Per-frame meta must advertise the same counts and hashes.
    const nlohmann::json meta = ReadJson(metaPath);
    if (!meta.is_object())
    {
        throw IoError("frame " + std::to_string(frame) + ": meta is not an object",
                      kExitInvalidInput);
    }
    Compare(context + " meta n_rays", RequireCount(meta, "n_rays", context), derived.nRays,
            mismatches);
    Compare(context + " meta full_count", RequireCount(meta, "full_count", context), derived.nRays,
            mismatches);
    const nlohmann::json &metaBuckets = RequireObject(meta, "bucket_counts", context);
    for (std::size_t index = 0; index < kBucketKeys.size(); ++index)
    {
        Compare(context + " meta bucket_counts." + kBucketKeys[index],
                RequireCount(metaBuckets, kBucketKeys[index], context), derived.buckets[index],
                mismatches);
    }
    const nlohmann::json &metaCameras = RequireObject(meta, "camera_counts", context);
    Compare(context + " meta camera_counts.image_00", RequireCount(metaCameras, "image_00", context),
            derived.cameras[0], mismatches);
    Compare(context + " meta camera_counts.image_01", RequireCount(metaCameras, "image_01", context),
            derived.cameras[1], mismatches);
    Compare(context + " meta occluded_count", RequireCount(meta, "occluded_count", context),
            derived.occluded, mismatches);
    Compare(context + " meta occluded_excluded_count",
            RequireCount(meta, "occluded_excluded_count", context), derived.occludedExcluded,
            mismatches);

    return derived;
}

void ValidateFreezeDocument(const nlohmann::json &freeze, const std::string &context)
{
    if (!freeze.is_object())
    {
        throw IoError(context + ": freeze is not an object", kExitInvalidInput);
    }
    if (!freeze.contains("schema_version") || !freeze.at("schema_version").is_number_integer() ||
        freeze.at("schema_version").get<int>() != 1)
    {
        throw IoError(context + ": unsupported freeze schema_version", kExitInvalidInput);
    }
    static_cast<void>(RequireString(freeze, "sequence", context));
    static_cast<void>(RequireString(freeze, "config_sha256", context));
    static_cast<void>(RequireString(freeze, "code_version", context));
    static_cast<void>(RequireString(freeze, "manifest_sha256", context));
    static_cast<void>(RequireString(freeze, "ray_set_sha256", context));
    static_cast<void>(RequireArray(freeze, "window", context));
    static_cast<void>(ReadHashObject(freeze, "per_frame_sha256", context));
    static_cast<void>(ReadHashObject(freeze, "per_frame_meta_sha256", context));
    static_cast<void>(RequireObject(freeze, "counts", context));
}
} // namespace

RaySetFreeze FreezeRaySet(const fs::path &raysManifest, const fs::path &configPath,
                          const fs::path &freezePath, const std::string_view codeVersion)
{
    const std::string context = raysManifest.string();
    const nlohmann::json manifest = ReadJson(raysManifest);
    if (!manifest.is_object())
    {
        throw IoError(context + ": manifest is not an object", kExitInvalidInput);
    }
    const std::string sequence = RequireString(manifest, "sequence", context);
    const std::vector<int> window = ReadWindow(manifest, context);
    const nlohmann::json &frames = RequireArray(manifest, "frames", context);
    if (frames.size() != window.size())
    {
        throw IoError(context + ": frames array and window disagree", kExitInvalidInput);
    }

    const std::string configSha = Sha256File(configPath);
    const std::string manifestConfigSha = RequireString(manifest, "config_sha256", context);
    if (manifestConfigSha != configSha)
    {
        throw IoError(context + ": manifest was built with a different config", kExitGateFailed);
    }

    const std::string manifestSha = Sha256File(raysManifest);
    const fs::path directory = raysManifest.parent_path();

    RaySetFreeze freeze;
    freeze.sequence = sequence;
    freeze.window = window;
    freeze.config_sha256 = configSha;
    freeze.code_version = std::string(codeVersion);
    freeze.manifest_sha256 = manifestSha;

    for (const nlohmann::json &entry : frames)
    {
        if (!entry.is_object() || !entry.contains("frame") || !entry.at("frame").is_number_integer())
        {
            throw IoError(context + ": every frame entry needs an integer frame",
                          kExitInvalidInput);
        }
        const int frame = entry.at("frame").get<int>();
        const std::string frameContext = "frame " + std::to_string(frame);

        std::array<std::string, 4> hashes{};
        for (std::size_t index = 0; index < kFileKeys.size(); ++index)
        {
            const fs::path path = directory / FileNameFor(entry, kFileKeys[index], frameContext);
            hashes[index] = Sha256File(path);
            const std::string stored = StoredHash(entry, kFileKeys[index], frameContext);
            if (hashes[index] != stored)
            {
                throw IoError(frameContext + ": " + kFileKeys[index] +
                                  " hash disagrees with the manifest",
                              kExitGateFailed);
            }
        }

        const fs::path metaPath = directory / ("frame_" + FrameName(frame) + "_meta.json");
        const std::string metaHash = Sha256File(metaPath);

        freeze.per_frame_sha256.emplace_back(
            frame, CombinedFrameHash(hashes[0], hashes[1], hashes[2], hashes[3]));
        freeze.per_frame_meta_sha256.emplace_back(frame, metaHash);
    }

    freeze.counts = RequireObject(manifest, "totals", context);

    const nlohmann::json payload =
        BuildPayload(freeze.sequence, freeze.window, freeze.config_sha256, freeze.code_version,
                     freeze.manifest_sha256, freeze.per_frame_sha256,
                     freeze.per_frame_meta_sha256);
    freeze.ray_set_sha256 = Sha256Json(payload);
    freeze.frozen_at_utc = UtcNowIso8601();

    nlohmann::json document = nlohmann::json::object();
    document["schema_version"] = 1;
    document["sequence"] = freeze.sequence;
    document["window"] = freeze.window;
    document["config_sha256"] = freeze.config_sha256;
    document["code_version"] = freeze.code_version;
    document["frozen_at_utc"] = freeze.frozen_at_utc;
    document["manifest_sha256"] = freeze.manifest_sha256;
    document["ray_set_sha256"] = freeze.ray_set_sha256;
    document["per_frame_sha256"] = PairsToObject(freeze.per_frame_sha256);
    document["per_frame_meta_sha256"] = PairsToObject(freeze.per_frame_meta_sha256);
    document["counts"] = freeze.counts;

    EnsureDir(freezePath.parent_path());
    WriteJson(freezePath, document);
    return freeze;
}

nlohmann::json VerifyRaySet(const fs::path &raysManifest, const fs::path &freezePath)
{
    const std::string context = freezePath.string();
    const nlohmann::json freeze = ReadJson(freezePath);
    ValidateFreezeDocument(freeze, context);

    const nlohmann::json manifest = ReadJson(raysManifest);
    if (!manifest.is_object())
    {
        throw IoError(raysManifest.string() + ": manifest is not an object", kExitInvalidInput);
    }
    const std::string sequence = RequireString(manifest, "sequence", context);
    const std::vector<int> window = ReadWindow(manifest, context);
    const nlohmann::json &frames = RequireArray(manifest, "frames", context);
    const fs::path directory = raysManifest.parent_path();

    const std::string freezeSequence = RequireString(freeze, "sequence", context);
    const std::vector<int> freezeWindow = ReadWindow(freeze, context);
    const auto frozenHashes = ReadHashObject(freeze, "per_frame_sha256", context);
    const auto frozenMetaHashes = ReadHashObject(freeze, "per_frame_meta_sha256", context);

    std::vector<int> mismatchedFrames;
    std::vector<std::string> countMismatches;

    if (sequence != freezeSequence)
    {
        countMismatches.push_back("sequence: manifest=" + sequence + " freeze=" + freezeSequence);
    }
    if (window != freezeWindow)
    {
        countMismatches.push_back("window: manifest and freeze disagree");
    }

    const std::string storedManifestSha = RequireString(freeze, "manifest_sha256", context);
    const std::string actualManifestSha = Sha256File(raysManifest);
    if (storedManifestSha != actualManifestSha)
    {
        countMismatches.push_back("manifest_sha256: stored=" + storedManifestSha +
                                  " actual=" + actualManifestSha);
    }

    std::vector<std::pair<int, std::string>> actualFrameHashes;
    std::vector<std::pair<int, std::string>> actualMetaHashes;
    actualFrameHashes.reserve(frames.size());
    actualMetaHashes.reserve(frames.size());

    DerivedCounts totals;
    for (const nlohmann::json &entry : frames)
    {
        if (!entry.is_object() || !entry.contains("frame") || !entry.at("frame").is_number_integer())
        {
            throw IoError(context + ": every manifest frame needs an integer frame",
                          kExitInvalidInput);
        }
        const int frame = entry.at("frame").get<int>();
        const std::string frameContext = "frame " + std::to_string(frame);

        std::array<std::string, 4> hashes{};
        bool hashesMatch = true;
        for (std::size_t index = 0; index < kFileKeys.size(); ++index)
        {
            hashes[index] =
                Sha256File(directory / FileNameFor(entry, kFileKeys[index], frameContext));
            const std::string stored = StoredHash(entry, kFileKeys[index], frameContext);
            if (hashes[index] != stored)
            {
                hashesMatch = false;
            }
        }
        const std::string frameHash =
            CombinedFrameHash(hashes[0], hashes[1], hashes[2], hashes[3]);
        actualFrameHashes.emplace_back(frame, frameHash);

        const fs::path metaPath = directory / ("frame_" + FrameName(frame) + "_meta.json");
        const std::string metaHash = Sha256File(metaPath);
        actualMetaHashes.emplace_back(frame, metaHash);

        const auto frozen = std::find_if(frozenHashes.begin(), frozenHashes.end(),
                                         [frame](const auto &pair) { return pair.first == frame; });
        const auto frozenMeta =
            std::find_if(frozenMetaHashes.begin(), frozenMetaHashes.end(),
                         [frame](const auto &pair) { return pair.first == frame; });
        const bool metaMatches =
            frozenMeta != frozenMetaHashes.end() && frozenMeta->second == metaHash;
        if (!hashesMatch || frozen == frozenHashes.end() || frozen->second != frameHash ||
            !metaMatches)
        {
            mismatchedFrames.push_back(frame);
        }

        if (!hashesMatch || !metaMatches)
        {
            // A tampered file is reported as a hash failure; parsing it would
            // only convert the gate failure into a misleading input error.
            continue;
        }

        const DerivedCounts derived =
            DeriveCounts(directory, frame, entry, metaPath, countMismatches);
        totals.nRays += derived.nRays;
        for (std::size_t index = 0; index < totals.buckets.size(); ++index)
        {
            totals.buckets[index] += derived.buckets[index];
        }
        totals.cameras[0] += derived.cameras[0];
        totals.cameras[1] += derived.cameras[1];
        totals.occluded += derived.occluded;
        totals.occludedExcluded += derived.occludedExcluded;
    }

    // Re-derive manifest totals and compare against the advertised totals.
    if (manifest.contains("totals") && manifest.at("totals").is_object())
    {
        const nlohmann::json &manifestTotals = manifest.at("totals");
        Compare("totals.frames", RequireCount(manifestTotals, "frames", context), frames.size(),
                countMismatches);
        Compare("totals.n_rays", RequireCount(manifestTotals, "n_rays", context), totals.nRays,
                countMismatches);
        Compare("totals.full_count", RequireCount(manifestTotals, "full_count", context),
                totals.nRays, countMismatches);
        if (manifestTotals.contains("bucket_counts") &&
            manifestTotals.at("bucket_counts").is_object())
        {
            for (std::size_t index = 0; index < kBucketKeys.size(); ++index)
            {
                Compare(std::string("totals.bucket_counts.") + kBucketKeys[index],
                        RequireCount(manifestTotals.at("bucket_counts"), kBucketKeys[index],
                                     context),
                        totals.buckets[index], countMismatches);
            }
        }
        if (manifestTotals.contains("camera_counts") &&
            manifestTotals.at("camera_counts").is_object())
        {
            Compare("totals.camera_counts.image_00",
                    RequireCount(manifestTotals.at("camera_counts"), "image_00", context),
                    totals.cameras[0], countMismatches);
            Compare("totals.camera_counts.image_01",
                    RequireCount(manifestTotals.at("camera_counts"), "image_01", context),
                    totals.cameras[1], countMismatches);
        }
        Compare("totals.occluded_count", RequireCount(manifestTotals, "occluded_count", context),
                totals.occluded, countMismatches);
        Compare("totals.occluded_excluded_count",
                RequireCount(manifestTotals, "occluded_excluded_count", context),
                totals.occludedExcluded, countMismatches);
    }
    else
    {
        countMismatches.push_back("manifest has no totals object");
    }

    const std::string frozenConfigSha = RequireString(freeze, "config_sha256", context);
    const std::string frozenCodeVersion = RequireString(freeze, "code_version", context);
    const nlohmann::json payload =
        BuildPayload(sequence, window, frozenConfigSha, frozenCodeVersion, actualManifestSha,
                     actualFrameHashes, actualMetaHashes);
    const std::string expected = RequireString(freeze, "ray_set_sha256", context);
    const std::string actual = Sha256Json(payload);
    if (expected != actual)
    {
        countMismatches.push_back("ray_set_sha256: stored=" + expected + " actual=" + actual);
    }

    nlohmann::json result = nlohmann::json::object();
    result["ok"] = mismatchedFrames.empty() && countMismatches.empty();
    result["expected"] = expected;
    result["actual"] = actual;
    result["mismatched_frames"] = mismatchedFrames;
    result["count_mismatches"] = countMismatches;
    return result;
}
} // namespace raylidar::rays
