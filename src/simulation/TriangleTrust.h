#pragma once

#include "rt/GaussianScene.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace raylidar::simulation
{
struct TriangleTrustSettings
{
    // A triangle is untrusted when its center is farther than this from the plane of the nearest 2D Gaussian (m).
    float MaxPlaneDistance = 0.08f;
    // Nearly transparent splats are ignored.
    float MinOpacity = 0.1f;
    // No usable splat within this distance of the triangle center (m) also makes the triangle untrusted.
    float SearchRadius = 0.75f;
};

struct TriangleTrustResult
{
    // One entry per triangle, 1 = untrusted. Pass this to HybridScanner::Scan.
    std::vector<std::uint8_t> Untrusted;
    std::size_t UntrustedCount = 0;
    std::size_t NoNearbySplatCount = 0;
};

// Ground-truth-free check of the mesh against the 2D Gaussians reconstructed from the same images.
// Where both reconstructions agree on the surface, the mesh is kept. Where they disagree, the triangle is flagged so
// that rays hitting it are re-traced against the Gaussians. Positions and Gaussians must use the same local origin.
[[nodiscard]]
TriangleTrustResult FindUntrustedTriangles(std::span<const std::array<float, 3>> positions,
                                           std::span<const std::uint32_t> indices,
                                           std::span<const rt::Gaussian> gaussians,
                                           const TriangleTrustSettings &settings = {});
} // namespace raylidar::simulation
