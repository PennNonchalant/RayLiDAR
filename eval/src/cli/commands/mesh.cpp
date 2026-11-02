#include <raylidar/cli/command.h>
#include <raylidar/cli/commands_phase4.h>

#include <raylidar/common/cli_options.h>
#include <raylidar/common/io.h>
#include <raylidar/io/sim_io.h>
#include <raylidar/mesh/manifest.h>

#include <Eigen/Core>

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::IoError;
using raylidar::ParseArgs;
using raylidar::ParsedArgs;
using raylidar::UsageError;

constexpr int kExitInvalidInput = 4;

const char *FormatName(const raylidar::io::MeshFormat format)
{
    switch (format)
    {
    case raylidar::io::MeshFormat::Ply:
        return "ply";
    case raylidar::io::MeshFormat::Glb:
        return "glb";
    case raylidar::io::MeshFormat::Obj:
        return "obj";
    }
    return "unknown";
}

void PrintHelp(std::ostream &out)
{
    out << "Usage: raylidar_cli mesh validate [options]\n"
        << "\n"
        << "Validate a Phase-4 W-frame metre mesh artifact against the frozen\n"
        << "docs/sim-io.md contract.\n"
        << "\n"
        << "Forms:\n"
        << "  --file <path>                     validate one PLY/GLB/OBJ structurally\n"
        << "  --mesh <path>                     alias for --file\n"
        << "  --manifest <path> --run-dir <dir> validate a mesh_manifest.json and its\n"
        << "                                    declared files (paths resolve under --run-dir)\n"
        << "\n"
        << "Options:\n"
        << "  --verify-hashes  recompute every recorded SHA-256 (manifest form only)\n"
        << "  --help           show this help\n"
        << "\n"
        << "Exit codes: 0 success, 2 usage, 4 invalid/missing input, 6 schema/hash drift.\n";
}

void PrintBbox(std::ostream &out, const Eigen::Vector3d &value)
{
    out << value.x() << ',' << value.y() << ',' << value.z();
}

int ExecuteFileValidation(const fs::path &path)
{
    std::error_code error;
    if (!fs::is_regular_file(path, error))
    {
        throw IoError("mesh file is missing: " + path.string(), kExitInvalidInput);
    }
    const raylidar::io::MeshStats stats = raylidar::io::ValidateMeshFile(path);
    std::cout << "OK format=" << FormatName(stats.format) << " vertices=" << stats.vertex_count
              << " triangles=" << stats.triangle_count << '\n';
    return 0;
}

int ExecuteManifestValidation(const fs::path &manifestPath, const fs::path &runDir, const bool verifyHashes)
{
    const raylidar::mesh::MeshManifestValidation validation =
        raylidar::mesh::ValidateMeshManifest(manifestPath, runDir, verifyHashes);
    if (!validation.ok)
    {
        throw IoError(validation.message, kExitInvalidInput);
    }
    std::cout << "OK vertices=" << validation.vertex_count << " triangles=" << validation.triangle_count
              << " bbox_min=";
    PrintBbox(std::cout, validation.bbox_min);
    std::cout << " bbox_max=";
    PrintBbox(std::cout, validation.bbox_max);
    std::cout << '\n';
    return 0;
}

int Execute(const std::span<const std::string_view> args)
{
    const ParsedArgs parsed = ParseArgs(args, {"manifest", "run-dir", "file", "mesh"}, {"help", "verify-hashes"});
    if (parsed.Flag("help"))
    {
        PrintHelp(std::cout);
        return 0;
    }

    const std::vector<std::string> modes = parsed.GetAll("");
    if (modes.empty())
    {
        PrintHelp(std::cerr);
        return 2;
    }
    if (modes.size() != 1U || modes.front() != "validate")
    {
        throw UsageError("expected exactly one mode: validate");
    }

    std::string fileText = parsed.Get("file");
    if (fileText.empty())
    {
        fileText = parsed.Get("mesh");
    }
    const std::string manifestText = parsed.Get("manifest");
    const bool haveFile = !fileText.empty();
    const bool haveManifest = !manifestText.empty();
    if (haveFile && haveManifest)
    {
        throw UsageError("use either --file/--mesh or --manifest, not both");
    }
    if (!haveFile && !haveManifest)
    {
        PrintHelp(std::cerr);
        return 2;
    }

    if (haveFile)
    {
        if (parsed.Flag("verify-hashes"))
        {
            throw UsageError("--verify-hashes requires --manifest");
        }
        return ExecuteFileValidation(fs::path(fileText));
    }

    const std::string runDirText = parsed.Get("run-dir");
    if (runDirText.empty())
    {
        throw UsageError("mesh validate --manifest requires --run-dir <dir>");
    }
    return ExecuteManifestValidation(fs::path(manifestText), fs::path(runDirText), parsed.Flag("verify-hashes"));
}
} // namespace

int RunMesh(const std::span<const std::string_view> args)
{
    try
    {
        return Execute(args);
    }
    catch (const UsageError &error)
    {
        std::cerr << "error: " << error.what() << "\n\n";
        PrintHelp(std::cerr);
        return error.exitCode();
    }
    catch (const IoError &error)
    {
        std::cout.flush();
        std::cerr << "INVALID " << error.what() << '\n';
        return error.exitCode();
    }
    catch (const std::exception &error)
    {
        std::cerr << "INVALID " << error.what() << '\n';
        return kExitInvalidInput;
    }
}

RAYLIDAR_REGISTER_COMMAND("mesh", "Validate Phase-4 mesh artifacts", RunMesh);
