#include "fixture.h"

#include <raylidar/common/io.h>
#include <raylidar/masking/semantics.h>

#include <stb_image_write.h>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::IoError;
using raylidar::masking::IsDynamicLabel;
using raylidar::masking::LabelAt;
using raylidar::masking::LoadSemantics;
using raylidar::masking::SemanticImage;
using raylidar::test::MakeMaskingTempRoot;
using raylidar::test::WriteGrayscalePng;
} // namespace

TEST_CASE("LoadSemantics reads one-channel 8-bit labels", "[masking]")
{
    const fs::path root = MakeMaskingTempRoot("semantics_load");
    const std::vector<std::uint8_t> labels = {7U, 26U, 11U, 21U, 24U, 33U, 32U, 28U, 0U};
    const fs::path path = root / "semantic.png";
    WriteGrayscalePng(path, 3, 3, labels);

    const SemanticImage image = LoadSemantics(path);
    CHECK(image.width == 3);
    CHECK(image.height == 3);
    REQUIRE(image.labels.size() == labels.size());
    CHECK(image.labels[0] == 7U);
    CHECK(image.labels[1] == 26U);

    CHECK(LabelAt(image, 0.0, 0.0) == 7U);
    CHECK(LabelAt(image, 1.0, 0.0) == 26U);
    CHECK(LabelAt(image, 0.4, 0.4) == 7U);
    CHECK(LabelAt(image, 0.6, 0.0) == 26U);
    CHECK(LabelAt(image, 3.0, 0.0) == 65535U);
    CHECK(LabelAt(image, -0.6, 0.0) == 65535U);
    CHECK(LabelAt(image, 0.0, 3.0) == 65535U);
}

TEST_CASE("LoadSemantics rejects multi-channel and oversized images", "[masking]")
{
    const fs::path root = MakeMaskingTempRoot("semantics_reject");

    SECTION("multi-channel")
    {
        const fs::path path = root / "rgb.png";
        const std::vector<std::uint8_t> rgb = {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U, 11U, 12U};
        REQUIRE(stbi_write_png(path.string().c_str(), 2, 2, 3, rgb.data(), 6) != 0);
        CHECK_THROWS_AS(LoadSemantics(path), IoError);
    }

    SECTION("pixel cap")
    {
        const fs::path path = root / "big.png";
        WriteGrayscalePng(path, 8, 8, std::vector<std::uint8_t>(64U, 7U));
        CHECK_THROWS_AS(LoadSemantics(path, 10U), IoError);
    }

    SECTION("missing file")
    {
        CHECK_THROWS_AS(LoadSemantics(root / "absent.png"), IoError);
    }
}

TEST_CASE("IsDynamicLabel uses the frozen closed set", "[masking]")
{
    for (const std::uint16_t id : {24U, 25U, 26U, 27U, 28U, 32U, 33U})
    {
        CHECK(IsDynamicLabel(id));
    }
    for (const std::uint16_t id : {0U, 7U, 11U, 21U, 23U, 29U, 31U, 34U, 65535U})
    {
        CHECK_FALSE(IsDynamicLabel(id));
    }
}
