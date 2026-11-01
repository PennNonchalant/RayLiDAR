#include <raylidar/cli/command.h>
#include <raylidar/cli/dispatch.h>

#include <raylidar/version.h>

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace raylidar::cli
{
namespace
{
struct Entry
{
    std::string help;
    CommandFn function;
};

// Function-local static avoids static-initialization-order problems: command
// registrars may run before any other translation unit has initialised state.
std::map<std::string, Entry> &Registry()
{
    static std::map<std::string, Entry> registry;
    return registry;
}

void PrintUsage(std::ostream &out)
{
    out << "Usage: raylidar_cli <command> [options]\n"
        << "       raylidar_cli --help\n"
        << "       raylidar_cli --version\n";

    const auto &registry = Registry();
    if (registry.empty())
    {
        out << "\nNo commands are registered.\n";
        return;
    }

    out << "\nCommands:\n";
    for (const auto &[name, entry] : registry)
    {
        out << "  " << name;
        if (!entry.help.empty())
        {
            out << "  " << entry.help;
        }
        out << '\n';
    }
}
} // namespace

void RegisterCommand(std::string_view name, std::string_view help, CommandFn fn)
{
    auto &registry = Registry();
    const std::string key(name);
    if (registry.find(key) != registry.end())
    {
        throw std::runtime_error("command already registered: " + key);
    }
    registry.emplace(key, Entry{std::string(help), fn});
}

CommandRegistrar::CommandRegistrar(std::string_view name, std::string_view help, CommandFn fn)
{
    RegisterCommand(name, help, fn);
}

int Dispatch(int argc, char *argv[])
{
    if (argc <= 1)
    {
        PrintUsage(std::cout);
        return 0;
    }

    const std::string_view first(argv[1]);
    if (first == "--version")
    {
        std::cout << "RayLiDAR " << GetVersion() << '\n';
        return 0;
    }
    if (first == "--help" || first == "-h")
    {
        PrintUsage(std::cout);
        return 0;
    }

    const auto &registry = Registry();
    const auto found = registry.find(std::string(first));
    if (found == registry.end())
    {
        std::cerr << "Unknown command: " << first << "\n\n";
        PrintUsage(std::cerr);
        return 2;
    }

    std::vector<std::string_view> arguments;
    arguments.reserve(static_cast<std::size_t>(argc > 2 ? argc - 2 : 0));
    for (int index = 2; index < argc; ++index)
    {
        arguments.emplace_back(argv[index]);
    }
    return found->second.function(arguments);
}
} // namespace raylidar::cli
