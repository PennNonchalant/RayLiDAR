#include <raylidar/common/io.h>

#include <openssl/evp.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <system_error>
#include <vector>

namespace raylidar
{
namespace
{
std::string HexDigest(const unsigned char *digest, unsigned int length)
{
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(static_cast<std::size_t>(length) * 2U);
    for (unsigned int i = 0; i < length; ++i)
    {
        out.push_back(kHex[(digest[i] >> 4U) & 0x0FU]);
        out.push_back(kHex[digest[i] & 0x0FU]);
    }
    return out;
}

class DigestContext
{
  public:
    DigestContext() : ctx_(EVP_MD_CTX_new())
    {
        if (ctx_ == nullptr)
        {
            throw IoError("failed to allocate a SHA-256 context");
        }
        if (EVP_DigestInit_ex(ctx_, EVP_sha256(), nullptr) != 1)
        {
            EVP_MD_CTX_free(ctx_);
            throw IoError("failed to initialise SHA-256");
        }
    }

    ~DigestContext()
    {
        EVP_MD_CTX_free(ctx_);
    }
    DigestContext(const DigestContext &) = delete;
    DigestContext &operator=(const DigestContext &) = delete;

    void Update(const void *data, std::size_t size)
    {
        if (size == 0U)
        {
            return;
        }
        if (EVP_DigestUpdate(ctx_, data, size) != 1)
        {
            throw IoError("failed to update SHA-256");
        }
    }

    std::string Final()
    {
        std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
        unsigned int length = 0;
        if (EVP_DigestFinal_ex(ctx_, digest.data(), &length) != 1)
        {
            throw IoError("failed to finalise SHA-256");
        }
        return HexDigest(digest.data(), length);
    }

  private:
    EVP_MD_CTX *ctx_;
};
} // namespace

IoError::IoError(const std::string &what, int exitCode) : std::runtime_error(what), exitCode_(exitCode)
{
}

int IoError::exitCode() const noexcept
{
    return exitCode_;
}

std::string Sha256String(std::string_view text)
{
    return Sha256Bytes(std::span<const std::byte>(reinterpret_cast<const std::byte *>(text.data()), text.size()));
}

std::string Sha256Bytes(std::span<const std::byte> bytes)
{
    DigestContext ctx;
    ctx.Update(bytes.data(), bytes.size());
    return ctx.Final();
}

std::string Sha256File(const std::filesystem::path &path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        throw IoError("cannot open file for SHA-256: " + path.string());
    }

    DigestContext ctx;
    std::vector<char> buffer(1U << 16U);
    while (in)
    {
        in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = in.gcount();
        if (count > 0)
        {
            ctx.Update(buffer.data(), static_cast<std::size_t>(count));
        }
    }
    if (in.bad())
    {
        throw IoError("read failure while hashing: " + path.string());
    }
    return ctx.Final();
}

nlohmann::json ReadJson(const std::filesystem::path &path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        throw IoError("cannot open JSON file: " + path.string());
    }
    try
    {
        nlohmann::json value;
        in >> value;
        return value;
    }
    catch (const nlohmann::json::exception &error)
    {
        throw IoError("invalid JSON in " + path.string() + ": " + error.what());
    }
}

void WriteJson(const std::filesystem::path &path, const nlohmann::json &value)
{
    if (path.has_parent_path())
    {
        EnsureDir(path.parent_path());
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
    {
        throw IoError("cannot open JSON file for writing: " + path.string());
    }
    out << value.dump(2) << '\n';
    if (!out)
    {
        throw IoError("failed to write JSON file: " + path.string());
    }
}

nlohmann::json CanonicalJson(const nlohmann::json &value)
{
    if (value.is_object())
    {
        // nlohmann::json's default object type is std::map, so inserting by key
        // yields lexicographic order regardless of the original insertion order.
        nlohmann::json out = nlohmann::json::object();
        for (auto it = value.begin(); it != value.end(); ++it)
        {
            out[it.key()] = CanonicalJson(it.value());
        }
        return out;
    }
    if (value.is_array())
    {
        nlohmann::json out = nlohmann::json::array();
        for (const auto &element : value)
        {
            out.push_back(CanonicalJson(element));
        }
        return out;
    }
    return value;
}

std::string Sha256Json(const nlohmann::json &value)
{
    return Sha256String(CanonicalJson(value).dump());
}

YAML::Node ReadYaml(const std::filesystem::path &path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        throw IoError("cannot open YAML file: " + path.string());
    }
    std::ostringstream contents;
    contents << in.rdbuf();
    if (in.bad())
    {
        throw IoError("read failure while loading YAML: " + path.string());
    }
    const std::string text = contents.str();
    if (text.find_first_not_of(" \t\r\n") == std::string::npos)
    {
        return YAML::Node(YAML::NodeType::Map);
    }
    try
    {
        return YAML::Load(text);
    }
    catch (const YAML::Exception &error)
    {
        throw IoError("invalid YAML in " + path.string() + ": " + error.what());
    }
}

std::string UtcNowIso8601()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t seconds = std::chrono::system_clock::to_time_t(now);
    std::tm broken_down{};
#if defined(_WIN32)
    gmtime_s(&broken_down, &seconds);
#else
    gmtime_r(&seconds, &broken_down);
#endif
    std::array<char, 32> buffer{};
    std::strftime(buffer.data(), buffer.size(), "%Y-%m-%dT%H:%M:%SZ", &broken_down);
    return std::string(buffer.data());
}

std::filesystem::path EnsureDir(const std::filesystem::path &path)
{
    if (path.empty())
    {
        return path;
    }
    std::error_code error;
    std::filesystem::create_directories(path, error);
    if (error)
    {
        throw IoError("cannot create directory " + path.string() + ": " + error.message());
    }
    return path;
}
} // namespace raylidar
