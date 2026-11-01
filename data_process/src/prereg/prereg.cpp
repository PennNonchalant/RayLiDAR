#include <raylidar/prereg/prereg.h>

#include <raylidar/common/io.h>

#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>

namespace raylidar::prereg
{
namespace
{
namespace fs = std::filesystem;

constexpr const char *kFrozenMarker = "**Frozen at (UTC):**";
constexpr int kExitInvalidInput = 4;

std::string ReadText(const fs::path &path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        throw IoError("cannot open file: " + path.string(), kExitInvalidInput);
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    if (input.bad())
    {
        throw IoError("read failure: " + path.string(), kExitInvalidInput);
    }
    return contents.str();
}

void WriteText(const fs::path &path, std::string_view text)
{
    if (path.has_parent_path())
    {
        EnsureDir(path.parent_path());
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
    {
        throw IoError("cannot open file for writing: " + path.string(), kExitInvalidInput);
    }
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!output)
    {
        throw IoError("failed to write file: " + path.string(), kExitInvalidInput);
    }
}

bool IsHex(char ch)
{
    return std::isxdigit(static_cast<unsigned char>(ch)) != 0;
}

/// Normalise a 64-character SHA-256 token to lowercase.
std::string NormaliseHex(std::string_view token)
{
    std::string out;
    out.reserve(token.size());
    for (const char ch : token)
    {
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    return out;
}
} // namespace

std::string Sha256Decisions(const fs::path &path)
{
    return Sha256File(path);
}

std::string ReadFrozenAt(const fs::path &decisions)
{
    const std::string text = ReadText(decisions);
    const std::size_t marker = text.find(kFrozenMarker);
    if (marker == std::string::npos)
    {
        return {};
    }

    std::size_t start = marker + std::string_view(kFrozenMarker).size();
    while (start < text.size() && (text[start] == ' ' || text[start] == '\t'))
    {
        ++start;
    }
    std::size_t end = text.find('\n', start);
    if (end == std::string::npos)
    {
        end = text.size();
    }
    while (end > start && (text[end - 1] == '\r' || text[end - 1] == ' ' || text[end - 1] == '\t'))
    {
        --end;
    }
    return text.substr(start, end - start);
}

void ReplaceFrozenAt(const fs::path &decisions, std::string_view utc)
{
    std::string text = ReadText(decisions);
    const std::size_t marker = text.find(kFrozenMarker);
    if (marker == std::string::npos)
    {
        throw IoError(decisions.string() + ": missing '" + kFrozenMarker + "' line",
                      kExitInvalidInput);
    }

    std::size_t lineStart = 0;
    if (marker != 0U)
    {
        const std::size_t previousNewline = text.rfind('\n', marker - 1U);
        lineStart = previousNewline == std::string::npos ? 0 : previousNewline + 1U;
    }
    const std::size_t newline = text.find('\n', marker);
    const std::size_t lineEnd = newline == std::string::npos ? text.size() : newline;

    std::string replacement(kFrozenMarker);
    replacement += ' ';
    replacement += utc;

    text = text.substr(0, lineStart) + replacement + text.substr(lineEnd);
    if (text.empty() || text.back() != '\n')
    {
        text.push_back('\n');
    }
    WriteText(decisions, text);
}

nlohmann::json FreezeDecisions(const fs::path &decisions, const fs::path &protocol,
                               const fs::path &sidecar, std::string_view nowUtc)
{
    const std::string utc = nowUtc.empty() ? UtcNowIso8601() : std::string(nowUtc);

    ReplaceFrozenAt(decisions, utc);

    const std::string decisionsSha = Sha256Decisions(decisions);
    const std::string protocolSha = Sha256File(protocol);

    const std::string sidecarLine =
        decisionsSha + "  " + decisions.generic_string() + "\n";
    WriteText(sidecar, sidecarLine);

    nlohmann::json record;
    record["schema_version"] = 1;
    record["frozen_at_utc"] = utc;
    record["decisions_path"] = decisions.generic_string();
    record["decisions_sha256"] = decisionsSha;
    record["protocol_path"] = protocol.generic_string();
    record["protocol_sha256"] = protocolSha;

    fs::path sidecarJson = sidecar;
    sidecarJson += ".json";
    WriteJson(sidecarJson, record);

    return record;
}

nlohmann::json VerifyDecisions(const fs::path &decisions, const fs::path &sidecar)
{
    if (!fs::is_regular_file(sidecar))
    {
        throw IoError("hash sidecar not found: " + sidecar.string(), kExitInvalidInput);
    }

    const std::string contents = ReadText(sidecar);
    if (contents.size() < 64U)
    {
        throw IoError("malformed hash sidecar (shorter than 64 characters): " +
                          sidecar.string(),
                      kExitInvalidInput);
    }

    const std::string_view first(contents.data(), 64U);
    for (const char ch : first)
    {
        if (!IsHex(ch))
        {
            throw IoError("malformed hash sidecar (non-hex digest): " + sidecar.string(),
                          kExitInvalidInput);
        }
    }

    const std::string expected = NormaliseHex(first);
    const std::string actual = Sha256Decisions(decisions);

    nlohmann::json result;
    result["ok"] = expected == actual;
    result["expected"] = expected;
    result["actual"] = actual;
    result["frozen_at_utc"] = ReadFrozenAt(decisions);
    return result;
}
} // namespace raylidar::prereg
