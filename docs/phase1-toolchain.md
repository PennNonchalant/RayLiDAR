# Phase 1 Toolchain Reference

Verified inside the project container (`hls-verify:ubuntu2504`): CMake 3.31.6,
Ninja 1.12.1, GCC 14.2.0, vcpkg at `VCPKG_ROOT=/work/vcpkg` with the manifest
baseline `d186940b2f0c4e8e5b4ff804af57d2a26fbfa091`.

## Verified vcpkg ports and CMake targets

Every port below was confirmed with `"$VCPKG_ROOT/vcpkg" search <port>` and its
`share/<port>/usage` file before it was added to `vcpkg.json`.

| Port | Version | CMake package | Consumed as |
|------|---------|---------------|-------------|
| `eigen3` | 5.0.1 | `find_package(Eigen3 CONFIG REQUIRED)` | `Eigen3::Eigen` |
| `yaml-cpp` | 0.9.0#1 | `find_package(yaml-cpp CONFIG REQUIRED)` | `yaml-cpp::yaml-cpp` |
| `nlohmann-json` | 3.12.0#2 | `find_package(nlohmann_json CONFIG REQUIRED)` | `nlohmann_json::nlohmann_json` |
| `catch2` | 3.16.1 | `find_package(Catch2 3 CONFIG REQUIRED)` | `Catch2::Catch2WithMain` (tests only) |
| `stb` | 2026-08-02 | `find_package(Stb REQUIRED)` (Find module) | `${Stb_INCLUDE_DIR}` header-only |
| `openssl` | 3.6.5#2 | `find_package(OpenSSL REQUIRED)` | `OpenSSL::Crypto` (SHA-256) |
| `libarchive` | 3.8.9 | Find module only | `LibArchive::LibArchive` |
| `vulkan-headers` | 1.4.363.0 | `find_package(VulkanHeaders CONFIG REQUIRED)` | `Vulkan::Headers` |
| `shader-slang` | 2026.20 | `find_package(slang CONFIG REQUIRED)` | shader compile host tool |

### libarchive package mode

The vcpkg `libarchive` port ships **only a CMake Find module** (no CONFIG
package): `share/libarchive/` contains `vcpkg-cmake-wrapper.cmake` and `usage`,
not a `*Config.cmake`. `CMakeLists.txt` therefore tries
`find_package(LibArchive CONFIG QUIET)` first and falls back to
`find_package(LibArchive REQUIRED)`. Both paths resolve to the imported target
`LibArchive::LibArchive` on CMake >= 3.17, which is what `src/CMakeLists.txt`
links (PUBLIC, because later dataset tests build archive fixtures). The
configure log confirms `Found LibArchive: .../libarchive.a (found version
3.8.9)`.

### Catch2

`catch2` is a test-only dependency. It is pulled by the manifest but linked only
by targets under `tests/`.

## STB single-translation-unit rule

`src/common/stb_impl.cpp` is the **only** translation unit that defines
`STB_IMAGE_IMPLEMENTATION` and `STB_IMAGE_WRITE_IMPLEMENTATION`, then includes
`<stb_image.h>` and `<stb_image_write.h>`. Because `src/CMakeLists.txt` globs
`src/**/*.cpp`, that file compiles into `raylidar` once; no other target may add
an STB implementation TU.

Every other translation unit includes `<stb_image.h>` / `<stb_image_write.h>`
**without** the implementation macros. `tests/common/test_io.cpp` encodes a 2×2
RGB PNG (`stbi_write_png_to_func`) and reads its dimensions back with
`stbi_info_from_memory`, proving the implementations resolve inside the library.

Note: the library uses hidden visibility, so `stb_impl.cpp` wraps the STB
includes in a focused `#pragma GCC visibility push(default)` block to keep the
STB entry points importable by the tests. The definitions still exist exactly
once.

## `RAYLIDAR_EXPORT` rule

`raylidar` is built with `CXX_VISIBILITY_PRESET hidden` and
`VISIBILITY_INLINES_HIDDEN ON`. Every public function/class in
`include/raylidar/**` must be annotated with `RAYLIDAR_EXPORT` from
`<raylidar/export.h>`; inline/template definitions in headers do not need it.
Tests link the shared library, so a missing annotation fails at link time.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `RAYLIDAR_CONTAINER_RUN_CMD` | Exact `docker run ...` prefix that executes every job (from `.env.container`) |
| `RAYLIDAR_CONTAINER_WORKDIR` | Repository mount inside the container (`/src`) |
| `RAYLIDAR_PRESET` | CMake configure/build/test preset (`linux-debug`) |
| `RAYLIDAR_ARTIFACTS_ROOT` | Artifact root (default `/data-20T/hybrid-lidar-sim/artifacts`) |

`.env.container` is local-only and gitignored; `.env.container.example` contains
only the `<IMAGE>` placeholder. Never put credentials in either file.

## Preset commands

```bash
bash scripts/configure.sh                       # cmake --preset linux-debug
bash scripts/build.sh                           # configure + cmake --build --preset linux-debug
bash scripts/test.sh -L common --output-on-failure
bash scripts/test.sh -R cli --output-on-failure
bash scripts/test.sh -LE gpu --output-on-failure --no-tests=error
bash scripts/raylidar.sh <subcommand> [options]
```

All of them execute through `scripts/run-in-container.sh`, which sources
`.env.container`, requires `RAYLIDAR_CONTAINER_RUN_CMD`, and forwards
`RAYLIDAR_PRESET` and `RAYLIDAR_ARTIFACTS_ROOT`.

## Test labels

Phase 1 Catch2 domains are labelled `common`, `cli`, `dataset`, `calib`,
`prereg`, `overlay`, `g1`. All Phase 1 tests are **CPU-only**: they never
require a GPU or the Vulkan smoke test. `vulkan_smoke` keeps the `gpu` label and
is excluded with `-LE gpu`.
