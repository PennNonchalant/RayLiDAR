#include <raylidar/calib/frames.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace raylidar::calib
{
namespace
{
std::string Sci(double value)
{
    std::ostringstream out;
    out.precision(6);
    out << std::scientific << value;
    return out.str();
}

void RequireFinite(const Eigen::Matrix4d &matrix, const char *name)
{
    if (!matrix.allFinite())
    {
        throw std::runtime_error(std::string(name) + ": matrix contains non-finite values");
    }
}

Eigen::Vector3d MedianComponentWise(const std::vector<Eigen::Vector3d> &values)
{
    if (values.empty())
    {
        throw std::invalid_argument("MakeSegTransform: camera centre list is empty");
    }
    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> zs;
    xs.reserve(values.size());
    ys.reserve(values.size());
    zs.reserve(values.size());
    for (const Eigen::Vector3d &value : values)
    {
        if (!value.allFinite())
        {
            throw std::invalid_argument("MakeSegTransform: camera centre is not finite");
        }
        xs.push_back(value.x());
        ys.push_back(value.y());
        zs.push_back(value.z());
    }
    auto median = [](std::vector<double> &data) {
        const std::size_t middle = data.size() / 2;
        std::nth_element(data.begin(), data.begin() + static_cast<std::ptrdiff_t>(middle),
                         data.end());
        const double upper = data[middle];
        if (data.size() % 2 != 0)
        {
            return upper;
        }
        std::nth_element(data.begin(), data.begin() + static_cast<std::ptrdiff_t>(middle - 1),
                         data.end());
        return 0.5 * (upper + data[middle - 1]);
    };
    return Eigen::Vector3d(median(xs), median(ys), median(zs));
}

Eigen::Vector3d FitPlane(const Eigen::MatrixXd &candidates, const std::vector<int> &active,
                         std::size_t line)
{
    const std::size_t count = active.size();
    if (count < 3U)
    {
        throw std::invalid_argument("EstimateGroundPlane: fewer than 3 points for the plane fit");
    }
    Eigen::MatrixXd design(static_cast<Eigen::Index>(count), 3);
    Eigen::VectorXd target(static_cast<Eigen::Index>(count));
    for (std::size_t index = 0; index < count; ++index)
    {
        const int row = active[index];
        design(static_cast<Eigen::Index>(index), 0) = candidates(row, 0);
        design(static_cast<Eigen::Index>(index), 1) = candidates(row, 1);
        design(static_cast<Eigen::Index>(index), 2) = 1.0;
        target(static_cast<Eigen::Index>(index)) = candidates(row, 2);
    }
    const Eigen::Vector3d coefficients = design.colPivHouseholderQr().solve(target);
    if (!coefficients.allFinite())
    {
        (void)line;
        throw std::invalid_argument("EstimateGroundPlane: plane fit produced non-finite values");
    }
    return coefficients;
}
} // namespace

Eigen::Matrix4d Compose(const Eigen::Matrix4d &T_A_B, const Eigen::Matrix4d &T_B_C)
{
    RequireFinite(T_A_B, "Compose");
    RequireFinite(T_B_C, "Compose");
    Eigen::Matrix4d result = T_A_B * T_B_C;
    RequireFinite(result, "Compose");
    return result;
}

Eigen::Matrix4d AsHomogeneous(const Eigen::MatrixXd &mat)
{
    if (!mat.allFinite())
    {
        throw std::runtime_error("AsHomogeneous: matrix contains non-finite values");
    }
    if ((mat.rows() == 3 && mat.cols() == 4) || (mat.rows() == 4 && mat.cols() == 4))
    {
        Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
        result.topLeftCorner<3, 4>() = mat.topLeftCorner<3, 4>();
        if (mat.rows() == 4)
        {
            result.bottomRows<1>() = mat.bottomRows<1>();
            if (!result.allFinite())
            {
                throw std::runtime_error("AsHomogeneous: matrix contains non-finite values");
            }
        }
        return result;
    }
    throw std::runtime_error("AsHomogeneous: expected a 3x4 or 4x4 matrix");
}

Eigen::Matrix4d InvertRigid(const Eigen::Matrix4d &T, double atol, double relAtol)
{
    RequireFinite(T, "InvertRigid");
    const Eigen::Matrix3d rotation = T.topLeftCorner<3, 3>();
    AssertRotation(rotation);

    Eigen::Matrix4d analytic = Eigen::Matrix4d::Identity();
    analytic.topLeftCorner<3, 3>() = rotation.transpose();
    analytic.topRightCorner<3, 1>() = -rotation.transpose() * T.topRightCorner<3, 1>();

    const Eigen::Matrix4d reference = T.inverse();
    if (!reference.allFinite())
    {
        throw std::runtime_error("InvertRigid: 4x4 inverse is not finite");
    }

    const double scale = std::max(1.0, T.topRightCorner<3, 1>().norm());
    const double tolerance = atol + relAtol * scale;
    const double difference = (analytic - reference).cwiseAbs().maxCoeff();
    if (!(difference <= tolerance))
    {
        throw std::runtime_error("InvertRigid: analytic inverse disagrees with the 4x4 inverse by " +
                                 Sci(difference) + " > " + Sci(tolerance));
    }
    return analytic;
}

Eigen::Matrix3d RotationOf(const Eigen::Matrix4d &T)
{
    return T.topLeftCorner<3, 3>();
}

Eigen::Vector3d TranslationOf(const Eigen::Matrix4d &T)
{
    return T.topRightCorner<3, 1>();
}

void AssertRotation(const Eigen::Matrix3d &R, double atol, double detAtol)
{
    if (!R.allFinite())
    {
        throw std::runtime_error("AssertRotation: rotation contains non-finite values");
    }
    const double determinantError = std::abs(R.determinant() - 1.0);
    const double orthonormalityError =
        (R.transpose() * R - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff();
    if (!(determinantError <= detAtol) || !(orthonormalityError <= atol))
    {
        throw std::runtime_error("AssertRotation: det error " + Sci(determinantError) +
                                 " (limit " + Sci(detAtol) + "), orthonormality error " +
                                 Sci(orthonormalityError) + " (limit " + Sci(atol) + ")");
    }
}

void AssertRoundtrip(const Eigen::Matrix4d &T_AB, const Eigen::Matrix4d &T_BA, double atol)
{
    RequireFinite(T_AB, "AssertRoundtrip");
    RequireFinite(T_BA, "AssertRoundtrip");
    const double error = (T_AB * T_BA - Eigen::Matrix4d::Identity()).cwiseAbs().maxCoeff();
    if (!(error <= atol))
    {
        throw std::runtime_error("AssertRoundtrip: round-trip error " + Sci(error) + " > " +
                                 Sci(atol));
    }
}

std::pair<Eigen::Matrix4d, Eigen::Matrix4d> MakeSegTransform(
    const std::vector<Eigen::Vector3d> &camCentersW)
{
    const Eigen::Vector3d origin = MedianComponentWise(camCentersW);
    Eigen::Matrix4d T_S_W = Eigen::Matrix4d::Identity();
    T_S_W.topRightCorner<3, 1>() = -origin;
    Eigen::Matrix4d T_W_S = Eigen::Matrix4d::Identity();
    T_W_S.topRightCorner<3, 1>() = origin;
    return {T_S_W, T_W_S};
}

PoseLookup LookupPose(const std::map<int, Eigen::Matrix4d> &poses, int frame, int maxDelta)
{
    const auto exact = poses.find(frame);
    if (exact != poses.end())
    {
        PoseLookup lookup;
        lookup.frameRequested = frame;
        lookup.poseFrame = frame;
        lookup.delta = 0;
        lookup.T_W_I = exact->second;
        return lookup;
    }
    if (poses.empty())
    {
        throw std::out_of_range("LookupPose: no poses available for requested frame " +
                                std::to_string(frame));
    }

    int bestFrame = poses.begin()->first;
    long long bestDelta = std::llabs(static_cast<long long>(bestFrame) - frame);
    for (const auto &[candidateFrame, transform] : poses)
    {
        (void)transform;
        const long long delta =
            std::llabs(static_cast<long long>(candidateFrame) - static_cast<long long>(frame));
        // std::map iterates ascending, so a strict comparison keeps the lower
        // pose frame on a tie.
        if (delta < bestDelta)
        {
            bestDelta = delta;
            bestFrame = candidateFrame;
        }
    }
    if (bestDelta > static_cast<long long>(maxDelta))
    {
        throw std::out_of_range("LookupPose: requested frame " + std::to_string(frame) +
                                " has no pose within max_delta_frames=" + std::to_string(maxDelta) +
                                " (nearest is frame " + std::to_string(bestFrame) + ", delta " +
                                std::to_string(bestDelta) + ")");
    }

    PoseLookup lookup;
    lookup.frameRequested = frame;
    lookup.poseFrame = bestFrame;
    lookup.delta = bestFrame - frame;
    lookup.T_W_I = poses.at(bestFrame);
    return lookup;
}

GroundPlane EstimateGroundPlane(const Eigen::MatrixXd &pointsV, double radiusMin,
                                double radiusMax, double zMin, double zMax, int azimuthBins,
                                double trimFraction, int minInliers)
{
    if (pointsV.cols() != 3)
    {
        throw std::invalid_argument("EstimateGroundPlane: expected an Nx3 point matrix");
    }
    if (azimuthBins <= 0)
    {
        throw std::invalid_argument("EstimateGroundPlane: azimuth_bins must be positive");
    }
    if (!(trimFraction > 0.0 && trimFraction <= 1.0))
    {
        throw std::invalid_argument("EstimateGroundPlane: trim_fraction must be in (0, 1]");
    }

    // 2. Keep the single lowest-z return per azimuth sector.
    std::vector<double> lowestZ(static_cast<std::size_t>(azimuthBins),
                                std::numeric_limits<double>::infinity());
    std::vector<int> lowestRow(static_cast<std::size_t>(azimuthBins), -1);
    constexpr double kPi = 3.14159265358979323846;
    for (Eigen::Index row = 0; row < pointsV.rows(); ++row)
    {
        if (!pointsV.row(row).allFinite())
        {
            continue;
        }
        const double x = pointsV(row, 0);
        const double y = pointsV(row, 1);
        const double z = pointsV(row, 2);
        // 1. Horizontal radius gate.
        const double radius = std::hypot(x, y);
        if (radius < radiusMin || radius > radiusMax)
        {
            continue;
        }
        const double azimuth = std::atan2(y, x);
        int bin = static_cast<int>(std::floor((azimuth + kPi) / (2.0 * kPi) *
                                              static_cast<double>(azimuthBins)));
        if (bin < 0)
        {
            bin = 0;
        }
        if (bin >= azimuthBins)
        {
            bin = azimuthBins - 1;
        }
        const std::size_t index = static_cast<std::size_t>(bin);
        if (z < lowestZ[index])
        {
            lowestZ[index] = z;
            lowestRow[index] = static_cast<int>(row);
        }
    }

    // 3. Drop candidates outside the height band.
    std::vector<int> candidateRows;
    candidateRows.reserve(static_cast<std::size_t>(azimuthBins));
    for (const int row : lowestRow)
    {
        if (row < 0)
        {
            continue;
        }
        const double z = pointsV(row, 2);
        if (z >= zMin && z <= zMax)
        {
            candidateRows.push_back(row);
        }
    }
    if (static_cast<int>(candidateRows.size()) < minInliers)
    {
        throw std::invalid_argument("EstimateGroundPlane: only " +
                                    std::to_string(candidateRows.size()) +
                                    " ground candidates survived (min_inliers=" +
                                    std::to_string(minInliers) + ")");
    }

    Eigen::MatrixXd candidates(static_cast<Eigen::Index>(candidateRows.size()), 3);
    for (std::size_t index = 0; index < candidateRows.size(); ++index)
    {
        candidates.row(static_cast<Eigen::Index>(index)) =
            pointsV.row(candidateRows[index]).head<3>();
    }

    // 4. Iteratively trimmed least squares (6 refits).
    std::vector<int> active(candidateRows.size());
    for (std::size_t index = 0; index < active.size(); ++index)
    {
        active[index] = static_cast<int>(index);
    }
    Eigen::Vector3d coefficients = FitPlane(candidates, active, 0);
    for (int iteration = 0; iteration < 6; ++iteration)
    {
        Eigen::VectorXd residuals(candidates.rows());
        for (Eigen::Index row = 0; row < candidates.rows(); ++row)
        {
            residuals(row) = std::abs(coefficients[0] * candidates(row, 0) +
                                      coefficients[1] * candidates(row, 1) + coefficients[2] -
                                      candidates(row, 2));
        }
        const std::size_t total = static_cast<std::size_t>(candidates.rows());
        std::size_t keep = static_cast<std::size_t>(
            std::lround(trimFraction * static_cast<double>(total)));
        keep = std::max<std::size_t>(3U, std::min(keep, total));
        if (keep >= total)
        {
            active.resize(total);
            for (std::size_t index = 0; index < total; ++index)
            {
                active[index] = static_cast<int>(index);
            }
        }
        else
        {
            std::vector<double> ordered(residuals.data(),
                                        residuals.data() + static_cast<std::ptrdiff_t>(total));
            std::nth_element(ordered.begin(),
                             ordered.begin() + static_cast<std::ptrdiff_t>(keep - 1),
                             ordered.end());
            const double threshold = ordered[keep - 1];
            active.clear();
            for (std::size_t index = 0; index < total; ++index)
            {
                if (residuals(static_cast<Eigen::Index>(index)) <= threshold)
                {
                    active.push_back(static_cast<int>(index));
                }
            }
        }
        if (active.size() < 3U)
        {
            throw std::invalid_argument("EstimateGroundPlane: trimming left fewer than 3 points");
        }
        coefficients = FitPlane(candidates, active, static_cast<std::size_t>(iteration));
    }

    // 5. Normalise to n . p + offset = 0.
    const double a = coefficients[0];
    const double b = coefficients[1];
    const double c = coefficients[2];
    const double norm = std::sqrt(a * a + b * b + 1.0);
    if (!std::isfinite(norm) || norm <= 0.0)
    {
        throw std::invalid_argument("EstimateGroundPlane: invalid plane normal");
    }
    GroundPlane plane;
    plane.normal = Eigen::Vector3d(-a, -b, 1.0) / norm;
    plane.offset = -c / norm;
    return plane;
}

double CameraHeight(const GroundPlane &plane, const Eigen::Vector3d &cameraCentreV)
{
    return std::abs(plane.normal.dot(cameraCentreV) + plane.offset);
}

double CameraHeight(const Eigen::Matrix4d &T_V_C0, const GroundPlane &plane)
{
    return CameraHeight(plane, TranslationOf(T_V_C0));
}
} // namespace raylidar::calib
