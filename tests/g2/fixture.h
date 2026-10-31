#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace raylidar::test
{
/// Options for the synthetic Phase-2 artifact tree used by the G2 tests.
///
/// The defaults keep the tree small (3 rays / 5 GT points per frame) while
/// preserving the full 60-frame structure the aggregator requires. Callers can
/// raise a single value to exercise a specific gate.
struct G2FixtureOptions
{
    int frameCount = 60;
    /// Rays per frame. When empty, `raysPerFrameValue` is used for every frame.
    std::vector<int> raysPerFrame;
    int raysPerFrameValue = 3;
    /// When positive, overrides the ray count of `overriddenFrame`.
    int overriddenFrameRays = 0;
    int overriddenFrame = 0;
    int minValidRays = 3;
    int gtPointsPerFrame = 5;
    /// Every ray uses image_00 when false, so `image_01` is absent.
    bool image01 = true;
    int protocolWindowFrames = 60;
    /// Skip writing the decisions sidecar (missing-prereg evidence).
    bool omitPrereg = false;
};

/// Paths and metadata for a synthetic Phase-2 tree.
struct SyntheticG2Tree
{
    std::filesystem::path root;
    std::filesystem::path datasetRoot;
    std::filesystem::path phaseDir;
    std::filesystem::path phase1Dir;
    std::filesystem::path configsDir;
    std::filesystem::path configPath;
    std::filesystem::path protocolPath;
    std::filesystem::path decisionsPath;
    std::filesystem::path sidecarPath;
    std::filesystem::path raysManifestPath;
    std::filesystem::path freezePath;
    std::string sequence;
    std::vector<int> frames;
};

/// Fresh directory under the system temp directory.
std::filesystem::path MakeG2TempRoot(std::string_view name);

/// Write a UTF-8 text file, creating parent directories as needed.
void WriteG2TextFile(const std::filesystem::path &path, std::string_view text);

/// Build the complete synthetic artifact tree (config, protocol, decisions,
/// segment, uncurl, mask, GT, rays, waterfall, freeze).
SyntheticG2Tree MakeSyntheticG2Tree(std::string_view name,
                                    const G2FixtureOptions &options = {});

/// Rebuild the freeze after a manifest mutation so a test can isolate a check
/// that does not depend on the hashes.
void RebuildG2Freeze(const SyntheticG2Tree &tree);
} // namespace raylidar::test
