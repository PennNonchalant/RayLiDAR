# RayLiDAR

RayLiDAR explores GPU-accelerated LiDAR simulation using adaptive hybrid mesh and 3D Gaussian scene representations. The project aims to reduce simulation latency while preserving scan accuracy, with validation against KITTI-360. UE5 and Cesium support scene preparation and final visualization.

## Toolchain

| Area | Choice |
| --- | --- |
| Language | C++20; Windows/MSVC first, with Linux portability |
| GPU API | Vulkan 1.4, Vulkan-Hpp `vk::raii`, and VMA |
| Hardware RT | Compute shaders with `VK_KHR_ray_query` and BLAS/TLAS |
| Viewer | Dynamic rendering, GLFW, and Dear ImGui |
| Shaders | Slang compiled to SPIR-V; `shader-slang` managed through vcpkg |
| Build | CMake, Ninja, and CMake Presets |
| Dependencies | vcpkg Manifest Mode primarily; FetchContent as a fallback; Git submodules only for exceptional cases |

## Build Pipeline

```text
CMake Presets + vcpkg manifest
  ├── C++20 sources -> raylidar.dll + CLI/viewer executables
  └── Slang sources -> slangc -> SPIR-V shader assets
```

CMake coordinates dependency resolution, shader compilation, and application builds. Dependency revisions are pinned through the vcpkg baseline. Machine-specific settings belong in an untracked `CMakeUserPresets.json`.

## Architecture

- **raylidar:** Shared simulation library containing GPU resources, scene and sensor data, scanning algorithms, and timing. Mesh ray queries form the initial baseline; Gaussian and hybrid paths follow.
- **CLI:** Headless scanning, batch experiments, point-cloud export, and benchmarking.
- **Viewer:** Interactive scene and scan inspection, sensor controls, and performance feedback. It shares simulation code and GPU results with the library.

```text
apps/
    cli/                # Headless scans, export, and benchmarks
    viewer/             # Lightweight visualization and debugging
include/raylidar/       # Public library API
src/
    vulkan/             # Device, buffers, and acceleration structures
    simulation/         # Scene, sensor, and tracing implementation
shaders/                # Slang shaders and shared modules
cmake/                  # Build helpers
```

The simulator exchanges scene assets, sensor trajectories, and scan outputs with the UE5/Cesium workflow through files.

## Build and Run

| Prerequisite | Version / requirement |
| --- | --- |
| Visual Studio | 2022 with MSVC v143, C++20, and x64 tools; tested with VS 17.14 / MSVC 19.44 |
| CMake | 3.29+ |
| Ninja | 1.11+ |
| vcpkg | Manifest mode; tested with tool version 2026-09-26; dependency baseline pinned in `vcpkg.json` |
| GPU driver | Vulkan 1.4 with acceleration-structure and ray-query support |

Slang **2026.20** and Vulkan headers **1.4.363.0** are installed automatically from the pinned vcpkg manifest.

From an **x64 VS Developer PowerShell** at the repository root:

```powershell
$env:VCPKG_ROOT = "C:/path/to/vcpkg"
cmake --preset windows-debug
cmake --build --preset windows-debug
.\build\windows-debug\raylidar_cli.exe --version
ctest --preset windows-debug
```

The `raylidar` shared library and `raylidar_cli` targets produce `raylidar.dll` and `raylidar_cli.exe` together in `build/windows-debug/`. The CLI currently supports `--help` and `--version`; CTest runs the configured tests. Use `windows-release` for an optimized build. In Visual Studio, open the repository folder, select a preset and executable target, then build and run.

## Run Checks

After configuring and building, run from an **x64 VS Developer PowerShell** at the repository root:

```powershell
$env:PATH = "$env:VSINSTALLDIR/VC/Tools/Llvm/x64/bin;$env:PATH"
$sources = git ls-files -- '*.cpp' '*.cc' '*.cxx' '*.h' '*.hh' '*.hpp' '*.hxx'
clang-format --dry-run --Werror --style=file $sources

git ls-files -- '*.cpp' '*.cc' '*.cxx' | ForEach-Object {
    clang-tidy --quiet -p build/windows-debug $_
    if ($LASTEXITCODE -ne 0) { throw "clang-tidy failed: $_" }
}
ctest --preset windows-debug -LE gpu --output-on-failure --no-tests=error
```
