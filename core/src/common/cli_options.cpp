#include <raylidar/common/cli_options.h>

#include <utility>

namespace raylidar
{
UsageError::UsageError(const std::string &what) : std::runtime_error(what)
{
}

int UsageError::exitCode() const noexcept
{
    return 2;
}

ParsedArgs::ParsedArgs(std::map<std::string, std::vector<std::string>> values, std::set<std::string> flags)
    : values_(std::move(values)), flags_(std::move(flags))
{
}

bool ParsedArgs::Has(std::string_view key) const
{
    const std::string owned(key);
    return values_.find(owned) != values_.end() || flags_.find(owned) != flags_.end();
}

std::string ParsedArgs::Get(std::string_view key, std::string_view fallback) const
{
    const auto it = values_.find(std::string(key));
    if (it == values_.end() || it->second.empty())
    {
        return std::string(fallback);
    }
    return it->second.back();
}

std::vector<std::string> ParsedArgs::GetAll(std::string_view key) const
{
    const auto it = values_.find(std::string(key));
    if (it == values_.end())
    {
        return {};
    }
    return it->second;
}

bool ParsedArgs::Flag(std::string_view key) const
{
    return flags_.find(std::string(key)) != flags_.end();
}

ParsedArgs ParseArgs(std::span<const std::string_view> args, const std::set<std::string> &valueKeys,
                     const std::set<std::string> &flagKeys)
{
    std::map<std::string, std::vector<std::string>> values;
    std::set<std::string> flags;

    for (std::size_t index = 0; index < args.size(); ++index)
    {
        const std::string_view argument = args[index];
        if (argument == "--help")
        {
            flags.insert("help");
            continue;
        }
        if (argument.size() < 2U || argument.substr(0U, 2U) != "--")
        {
            // Positional argument; preserved under the empty key.
            values[std::string()].emplace_back(argument);
            continue;
        }

        const std::string_view body = argument.substr(2U);
        const auto equals = body.find('=');
        const std::string key = std::string(equals == std::string_view::npos ? body : body.substr(0U, equals));
        const bool hasInlineValue = equals != std::string_view::npos;
        const std::string inlineValue = hasInlineValue ? std::string(body.substr(equals + 1U)) : std::string();

        if (key.empty())
        {
            throw UsageError("empty option name in '" + std::string(argument) + "'");
        }

        if (key == "help" || flagKeys.find(key) != flagKeys.end())
        {
            if (hasInlineValue)
            {
                throw UsageError("option --" + key + " does not take a value");
            }
            flags.insert(key);
            continue;
        }

        if (valueKeys.find(key) == valueKeys.end())
        {
            throw UsageError("unknown option --" + key);
        }

        if (hasInlineValue)
        {
            values[key].push_back(inlineValue);
            continue;
        }

        if (index + 1U >= args.size())
        {
            throw UsageError("missing value for option --" + key);
        }
        ++index;
        values[key].emplace_back(args[index]);
    }

    return ParsedArgs(std::move(values), std::move(flags));
}
} // namespace raylidar
