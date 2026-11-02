#include <raylidar/overlay/distance_transform.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace raylidar::overlay
{
namespace
{
constexpr double kInf = 1e20;

/// Squared 1-D distance transform of `f` (already a squared-cost array). The
/// result is written to `d`. `v` and `z` are caller-provided scratch buffers
/// with at least `n + 1` entries each.
void Transform1d(const std::vector<double> &f, int n, std::vector<double> &d, std::vector<int> &v,
                 std::vector<double> &z)
{
    int k = 0;
    v[0] = 0;
    z[0] = -kInf;
    z[1] = kInf;
    for (int q = 1; q < n; ++q)
    {
        const double qd = static_cast<double>(q);
        double s = ((f[static_cast<std::size_t>(q)] + qd * qd) -
                    (f[static_cast<std::size_t>(v[k])] + static_cast<double>(v[k]) * static_cast<double>(v[k]))) /
                   (2.0 * qd - 2.0 * static_cast<double>(v[k]));
        while (s <= z[static_cast<std::size_t>(k)])
        {
            --k;
            s = ((f[static_cast<std::size_t>(q)] + qd * qd) -
                 (f[static_cast<std::size_t>(v[k])] + static_cast<double>(v[k]) * static_cast<double>(v[k]))) /
                (2.0 * qd - 2.0 * static_cast<double>(v[k]));
        }
        ++k;
        v[static_cast<std::size_t>(k)] = q;
        z[static_cast<std::size_t>(k)] = s;
        z[static_cast<std::size_t>(k) + 1U] = kInf;
    }
    k = 0;
    for (int q = 0; q < n; ++q)
    {
        while (z[static_cast<std::size_t>(k) + 1U] < static_cast<double>(q))
        {
            ++k;
        }
        const double delta = static_cast<double>(q) - static_cast<double>(v[static_cast<std::size_t>(k)]);
        d[static_cast<std::size_t>(q)] = delta * delta + f[static_cast<std::size_t>(v[static_cast<std::size_t>(k)])];
    }
}

void Validate(const std::vector<std::uint8_t> &buffer, int width, int height)
{
    if (width <= 0 || height <= 0)
    {
        throw std::invalid_argument("image dimensions must be positive");
    }
    const std::size_t expected = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (buffer.size() != expected)
    {
        throw std::invalid_argument("image buffer size does not match width * height");
    }
}
} // namespace

std::vector<double> EuclideanDistanceTransform(const std::vector<std::uint8_t> &binary, int width, int height)
{
    Validate(binary, width, height);
    const std::size_t pixels = binary.size();

    bool hasForeground = false;
    for (const std::uint8_t value : binary)
    {
        if (value != 0U)
        {
            hasForeground = true;
            break;
        }
    }
    if (!hasForeground)
    {
        return std::vector<double>(pixels, std::numeric_limits<double>::quiet_NaN());
    }

    // Seed: 0 on foreground pixels, +inf elsewhere.
    std::vector<double> grid(pixels, kInf);
    for (std::size_t index = 0; index < pixels; ++index)
    {
        if (binary[index] != 0U)
        {
            grid[index] = 0.0;
        }
    }

    const int maxDim = std::max(width, height);
    std::vector<double> line(static_cast<std::size_t>(maxDim));
    std::vector<double> out(static_cast<std::size_t>(maxDim));
    std::vector<int> v(static_cast<std::size_t>(maxDim) + 1U);
    std::vector<double> z(static_cast<std::size_t>(maxDim) + 1U);

    // Vertical pass: distance to the nearest foreground pixel within a column.
    for (int x = 0; x < width; ++x)
    {
        for (int y = 0; y < height; ++y)
        {
            line[static_cast<std::size_t>(y)] =
                grid[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)];
        }
        Transform1d(line, height, out, v, z);
        for (int y = 0; y < height; ++y)
        {
            grid[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)] =
                out[static_cast<std::size_t>(y)];
        }
    }

    // Horizontal pass completes the separable squared distance transform.
    for (int y = 0; y < height; ++y)
    {
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
        for (int x = 0; x < width; ++x)
        {
            line[static_cast<std::size_t>(x)] = grid[row + static_cast<std::size_t>(x)];
        }
        Transform1d(line, width, out, v, z);
        for (int x = 0; x < width; ++x)
        {
            grid[row + static_cast<std::size_t>(x)] = std::sqrt(out[static_cast<std::size_t>(x)]);
        }
    }

    return grid;
}

std::vector<double> GradientMagnitude(const std::vector<std::uint8_t> &gray, int width, int height)
{
    Validate(gray, width, height);
    std::vector<double> magnitude(gray.size(), 0.0);
    const auto at = [&gray, width](int x, int y) {
        return static_cast<double>(
            gray[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)]);
    };

    for (int y = 0; y < height; ++y)
    {
        const int ym = std::max(y - 1, 0);
        const int yp = std::min(y + 1, height - 1);
        for (int x = 0; x < width; ++x)
        {
            const int xm = std::max(x - 1, 0);
            const int xp = std::min(x + 1, width - 1);
            const double dx = (at(xp, y) - at(xm, y)) / static_cast<double>(xp - xm);
            const double dy = (at(x, yp) - at(x, ym)) / static_cast<double>(yp - ym);
            magnitude[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)] =
                std::hypot(dx, dy);
        }
    }
    return magnitude;
}

double Percentile(std::vector<double> values, double fraction)
{
    if (values.empty() || !std::isfinite(fraction))
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    fraction = std::clamp(fraction, 0.0, 1.0);
    std::sort(values.begin(), values.end());
    const double position = fraction * static_cast<double>(values.size() - 1U);
    const std::size_t low = static_cast<std::size_t>(std::floor(position));
    const std::size_t high = static_cast<std::size_t>(std::ceil(position));
    const double weight = position - static_cast<double>(low);
    return values[low] * (1.0 - weight) + values[high] * weight;
}
} // namespace raylidar::overlay
