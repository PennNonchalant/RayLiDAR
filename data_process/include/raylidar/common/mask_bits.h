#pragma once

#include <cstdint>

/// Frozen point-mask bit ABI for Phase 2.
///
/// Every Phase-2 module that reads or writes a per-point mask (the uncurl
/// diagnostic, the mask builder, the static GT cache, and the waterfall) uses
/// exactly these bits. Do not redefine the enum elsewhere.
namespace raylidar::masking
{
enum MaskBit : std::uint8_t
{
    kDynamic2D = 1U << 0U,        ///< Projects into a camera with a dynamic semantic label.
    kDynamicBBox = 1U << 1U,      ///< Inside a dynamic 3D bbox sampled for this frame.
    kDynamicUnion = 1U << 2U,     ///< kDynamic2D OR kDynamicBBox.
    kEgoRoi = 1U << 3U,           ///< Raw Velodyne point inside the D-09 ego ROI box.
    kKeptStatic = 1U << 4U,       ///< Finite, not ego, not dynamic.
    kInvalidNonFinite = 1U << 5U, ///< Non-finite input point; never transformed.
};

/// True when the point survived every exclusion stage.
[[nodiscard]] inline bool IsKeptStatic(std::uint8_t bits) noexcept
{
    return (bits & static_cast<std::uint8_t>(kKeptStatic)) != 0U;
}

/// True when any dynamic source flagged the point (the union bit).
[[nodiscard]] inline bool IsDynamic(std::uint8_t bits) noexcept
{
    return (bits & static_cast<std::uint8_t>(kDynamicUnion)) != 0U;
}

/// True when the point falls inside the ego-vehicle ROI.
[[nodiscard]] inline bool IsEgo(std::uint8_t bits) noexcept
{
    return (bits & static_cast<std::uint8_t>(kEgoRoi)) != 0U;
}
} // namespace raylidar::masking
