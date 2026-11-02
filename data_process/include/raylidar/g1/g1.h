#pragma once

#include <raylidar/export.h>

#include <nlohmann/json.hpp>

#include <string>
#include <utility>
#include <vector>

namespace raylidar::g1
{
/// Hard G1 thresholds. These are code constants: there is no CLI flag and no
/// runtime override, so a run can never relax a gate to make itself pass.
/// `tests/g1/test_g1.cpp` asserts they match `configs/frames.yaml`.
struct RAYLIDAR_EXPORT G1Thresholds
{
    /// Camera-relative rigid transforms (`T_Ck_rect_V`) only.
    double roundtripAtol = 1e-9;
    /// `R^T R = I` on composed rotations (official-file precision is ~1e-6).
    double rotationAtol = 2e-6;
    /// `det(R) = +1` on composed rotations (official-file precision is ~5e-7).
    double rotationDetAtol = 2e-6;
    /// `T_W_C0_rect` vs `cam0_to_world.txt`.
    double cam0WorldAtol = 1e-3;
    double cameraHeightMinM = 1.4;
    double cameraHeightMaxM = 1.9;
    double cameraHeightStabilityM = 0.1;
    /// Maximum accepted offset in **original 1x pixels** (strict `<`).
    double overlayMaxOffsetPx = 2.0;
    int overlayMinFrames = 3;
    int overlayMinFeatures = 5;
};

/// The one and only threshold set.
RAYLIDAR_EXPORT const G1Thresholds &Thresholds();

/// Validate the human overlay verification schema. Returns a list of errors;
/// an empty list means valid. When `expected` is non-null, the verification
/// must contain exactly one entry for every `(frame, camera)` pair: absent
/// pairs are reported as `missing pair (frame, camera)` and repeated pairs as
/// `duplicate pair (frame, camera)`.
RAYLIDAR_EXPORT std::vector<std::string> ValidateOverlayVerification(
    const nlohmann::json &obj, const std::vector<std::pair<int, std::string>> *expected = nullptr);

/// Complete expected `(frame, camera)` pair set from an overlay report: the
/// cross-product of `frames` and `cams`, cross-checked against `results`.
/// Throws `std::invalid_argument` when a result references a frame or camera
/// absent from `frames`/`cams`, or when a cross-product pair has no result.
RAYLIDAR_EXPORT std::vector<std::pair<int, std::string>> ExpectedPairsFromOverlay(const nlohmann::json &overlayReport);

/// Aggregate every G1 criterion into `{schema_version, generated_at_utc,
/// sequence, checks, blocking_failures, pass}`.
///
/// `verification` is optional. When it is `nullptr`, `overlay_offsets` is
/// reported as `pending_human` and `pass` is false; a pending human check can
/// never be mistaken for a pass.
RAYLIDAR_EXPORT nlohmann::json EvaluateG1(const nlohmann::json &preflight, const nlohmann::json &readiness,
                                          const nlohmann::json &calibReport, const nlohmann::json &overlayReport,
                                          const nlohmann::json &prereg, const nlohmann::json *verification);

/// Render the G1 report as Markdown. The literal `G1 PASSED` appears only when
/// `report["pass"] == true`; otherwise the final line is `G1 NOT PASSED`.
RAYLIDAR_EXPORT std::string RenderMarkdown(const nlohmann::json &report);
} // namespace raylidar::g1
