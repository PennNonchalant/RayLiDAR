#include <raylidar/common/io.h>
#include <raylidar/lidar/velodyne.h>

#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace raylidar::lidar
{
namespace
{
constexpr int kExitInvalidInput = 4;
constexpr std::size_t kTimestampLimit = 1000000U;

[[nodiscard]] bool IsDigit(const char ch) noexcept
{
    return std::isdigit(static_cast<unsigned char>(ch)) != 0;
}

[[nodiscard]] int TwoDigits(const std::string_view text, std::size_t offset, const char *what)
{
    if (!IsDigit(text[offset]) || !IsDigit(text[offset + 1U]))
    {
        throw IoError(std::string("timestamp: ") + what + " is not two digits", kExitInvalidInput);
    }
    return (text[offset] - '0') * 10 + (text[offset + 1U] - '0');
}

[[nodiscard]] std::string Trim(std::string_view text)
{
    std::size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0)
    {
        ++begin;
    }
    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1U])) != 0)
    {
        --end;
    }
    return std::string(text.substr(begin, end - begin));
}
} // namespace

std::vector<PointXYZI> LoadVelodyneScan(const std::filesystem::path &binPath, const std::size_t maxPoints)
{
    std::error_code error;
    const std::uintmax_t size = std::filesystem::file_size(binPath, error);
    if (error)
    {
        throw IoError("cannot read Velodyne scan " + binPath.string() + ": " + error.message(), kExitInvalidInput);
    }
    constexpr std::uintmax_t kRecordBytes = 4U * sizeof(float);
    if (size == 0U)
    {
        throw IoError("Velodyne scan is empty: " + binPath.string(), kExitInvalidInput);
    }
    if (size % kRecordBytes != 0U)
    {
        throw IoError("Velodyne scan size is not a multiple of 16 bytes: " + binPath.string(), kExitInvalidInput);
    }
    const std::uintmax_t count = size / kRecordBytes;
    if (count > maxPoints)
    {
        throw IoError("Velodyne scan has " + std::to_string(count) + " points, above the " + std::to_string(maxPoints) +
                          " point cap: " + binPath.string(),
                      kExitInvalidInput);
    }

    std::ifstream input(binPath, std::ios::binary);
    if (!input.good())
    {
        throw IoError("cannot open Velodyne scan " + binPath.string(), kExitInvalidInput);
    }

    std::vector<PointXYZI> points(static_cast<std::size_t>(count));
    for (PointXYZI &point : points)
    {
        float values[4] = {};
        input.read(reinterpret_cast<char *>(values), static_cast<std::streamsize>(kRecordBytes));
        if (!input.good())
        {
            throw IoError("truncated Velodyne scan " + binPath.string(), kExitInvalidInput);
        }
        point = PointXYZI{values[0], values[1], values[2], values[3]};
    }
    return points;
}

std::int64_t ParseUtcTimestamp(const std::string_view text)
{
    constexpr std::size_t kWidth = 29U; // YYYY-MM-DD HH:MM:SS.nnnnnnnnn
    if (text.size() != kWidth)
    {
        throw IoError("timestamp must be exactly 29 characters: '" + std::string(text) + "'", kExitInvalidInput);
    }
    if (text[4] != '-' || text[7] != '-' || text[10] != ' ' || text[13] != ':' || text[16] != ':' || text[19] != '.')
    {
        throw IoError("timestamp separators are malformed: '" + std::string(text) + "'", kExitInvalidInput);
    }
    for (std::size_t index = 0; index < kWidth; ++index)
    {
        if (index == 4U || index == 7U || index == 10U || index == 13U || index == 16U || index == 19U)
        {
            continue;
        }
        if (!IsDigit(text[index]))
        {
            throw IoError("timestamp contains a non-digit: '" + std::string(text) + "'", kExitInvalidInput);
        }
    }

    const int year = (text[0] - '0') * 1000 + (text[1] - '0') * 100 + (text[2] - '0') * 10 + (text[3] - '0');
    const int month = TwoDigits(text, 5U, "month");
    const int day = TwoDigits(text, 8U, "day");
    const int hour = TwoDigits(text, 11U, "hour");
    const int minute = TwoDigits(text, 14U, "minute");
    const int second = TwoDigits(text, 17U, "second");

    const std::chrono::year_month_day calendar{std::chrono::year{year},
                                               std::chrono::month{static_cast<unsigned>(month)},
                                               std::chrono::day{static_cast<unsigned>(day)}};
    if (!calendar.ok())
    {
        throw IoError("timestamp calendar date is out of range: '" + std::string(text) + "'", kExitInvalidInput);
    }
    if (hour > 23 || minute > 59 || second > 59)
    {
        throw IoError("timestamp clock is out of range: '" + std::string(text) + "'", kExitInvalidInput);
    }

    std::int64_t nanoseconds = 0;
    for (std::size_t index = 20U; index < kWidth; ++index)
    {
        nanoseconds = nanoseconds * 10 + (text[index] - '0');
    }

    const std::int64_t days = std::chrono::sys_days{calendar}.time_since_epoch().count();
    const std::int64_t seconds = days * 86400 + hour * 3600 + minute * 60 + second;
    return seconds * 1000000000LL + nanoseconds;
}

std::vector<std::int64_t> LoadVelodyneTimestamps(const std::filesystem::path &timestampsPath)
{
    std::ifstream input(timestampsPath);
    if (!input.good())
    {
        throw IoError("cannot read timestamps " + timestampsPath.string(), kExitInvalidInput);
    }

    std::vector<std::int64_t> values;
    std::string line;
    while (std::getline(input, line))
    {
        const std::string trimmed = Trim(line);
        if (trimmed.empty())
        {
            continue;
        }
        if (values.size() >= kTimestampLimit)
        {
            throw IoError("timestamps file exceeds " + std::to_string(kTimestampLimit) + " lines", kExitInvalidInput);
        }
        values.push_back(ParseUtcTimestamp(trimmed));
    }
    if (input.bad())
    {
        throw IoError("error while reading timestamps " + timestampsPath.string(), kExitInvalidInput);
    }
    if (values.empty())
    {
        throw IoError("timestamps file has no timestamps: " + timestampsPath.string(), kExitInvalidInput);
    }
    return values;
}
} // namespace raylidar::lidar
