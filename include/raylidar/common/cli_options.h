#pragma once

#include <raylidar/export.h>

#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace raylidar
{
/// Command-line usage error (exit code 2).
class RAYLIDAR_EXPORT UsageError : public std::runtime_error
{
public:
    explicit UsageError(const std::string &what);
    [[nodiscard]] int exitCode() const noexcept;
};

/// Result of ParseArgs. Keys may repeat; Get returns the last value.
struct RAYLIDAR_EXPORT ParsedArgs
{
    ParsedArgs() = default;
    explicit ParsedArgs(std::map<std::string, std::vector<std::string>> values,
                        std::set<std::string> flags);

    [[nodiscard]] bool Has(std::string_view key) const;
    [[nodiscard]] std::string Get(std::string_view key, std::string_view fallback = {}) const;
    [[nodiscard]] std::vector<std::string> GetAll(std::string_view key) const;
    [[nodiscard]] bool Flag(std::string_view key) const;

private:
    std::map<std::string, std::vector<std::string>> values_;
    std::set<std::string> flags_;
};

/// Parse `--key value`, `--key=value`, repeatable keys, and flags. `--help` is
/// always accepted as a flag. Unknown keys and missing values throw UsageError.
RAYLIDAR_EXPORT ParsedArgs ParseArgs(std::span<const std::string_view> args,
                                     const std::set<std::string> &valueKeys,
                                     const std::set<std::string> &flagKeys);
} // namespace raylidar
