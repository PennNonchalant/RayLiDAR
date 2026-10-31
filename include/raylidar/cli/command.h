#pragma once

#include <raylidar/export.h>

#include <span>
#include <string_view>

namespace raylidar::cli
{
/// Signature of every CLI subcommand handler. Arguments are those following the
/// command name (argv[2..]).
using CommandFn = int (*)(std::span<const std::string_view> args);

/// Register a subcommand. Registration happens from each command translation
/// unit through RAYLIDAR_REGISTER_COMMAND, so dispatch.cpp never needs editing.
RAYLIDAR_EXPORT void RegisterCommand(std::string_view name, std::string_view help,
                                     CommandFn fn);

/// Helper object instantiated by RAYLIDAR_REGISTER_COMMAND.
struct CommandRegistrar
{
    CommandRegistrar(std::string_view name, std::string_view help, CommandFn fn);
};
} // namespace raylidar::cli

#define RAYLIDAR_REGISTER_COMMAND(commandNameLiteral, helpText, functionName)                    \
    namespace                                                                                    \
    {                                                                                            \
    const ::raylidar::cli::CommandRegistrar g_registrar_##functionName(commandNameLiteral,       \
                                                                        helpText, functionName); \
    }
