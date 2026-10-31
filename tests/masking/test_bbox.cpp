#include "fixture.h"

#include <raylidar/common/io.h>
#include <raylidar/masking/bbox.h>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::IoError;
using raylidar::masking::ApplicableBoxes;
using raylidar::masking::DynamicBBox;
using raylidar::masking::LoadDynamicBBoxes;
using raylidar::masking::PointInBBox;
using raylidar::test::BBoxSpec;
using raylidar::test::MakeBBoxXml;
using raylidar::test::MakeMaskingTempRoot;
using raylidar::test::TranslatedScaled;
using raylidar::test::WriteTextFile;

BBoxSpec MakeSpec(const std::string &label, const int timestamp, const bool dynamic)
{
    BBoxSpec spec;
    spec.label = label;
    spec.semanticId = 26;
    spec.timestamp = timestamp;
    spec.startFrame = 100;
    spec.endFrame = 200;
    spec.dynamic = dynamic;
    spec.T_W_B = TranslatedScaled(10.0, 20.0, 30.0, 2.0);
    return spec;
}
} // namespace

TEST_CASE("LoadDynamicBBoxes parses dynamic objects strictly", "[masking]")
{
    const fs::path root = MakeMaskingTempRoot("bbox_parse");
    const fs::path path = root / "boxes.xml";
    WriteTextFile(path, MakeBBoxXml({MakeSpec("car", 150, true), MakeSpec("building", -1, false)}));

    const std::vector<DynamicBBox> boxes = LoadDynamicBBoxes(path);
    REQUIRE(boxes.size() == 2U);
    CHECK(boxes[0].label == "car");
    CHECK(boxes[0].semanticId == 26);
    CHECK(boxes[0].timestamp == 150);
    CHECK(boxes[0].startFrame == 100);
    CHECK(boxes[0].endFrame == 200);
    CHECK(boxes[0].dynamic);
    CHECK_FALSE(boxes[1].dynamic);
    CHECK(boxes[1].label == "building");
    CHECK(boxes[0].T_W_B.isApprox(TranslatedScaled(10.0, 20.0, 30.0, 2.0)));
}

TEST_CASE("LoadDynamicBBoxes rejects malformed and unsafe documents", "[masking]")
{
    const fs::path root = MakeMaskingTempRoot("bbox_reject");

    SECTION("external entities")
    {
        const fs::path path = root / "entity.xml";
        WriteTextFile(path, "<?xml version=\"1.0\"?>\n<!DOCTYPE foo [<!ENTITY x \"y\">]>\n"
                            "<opencv_storage></opencv_storage>\n");
        CHECK_THROWS_AS(LoadDynamicBBoxes(path), IoError);
    }

    SECTION("truncated object block")
    {
        const fs::path path = root / "truncated.xml";
        WriteTextFile(path, "<?xml version=\"1.0\"?>\n<opencv_storage>\n<object1>\n"
                            "  <label>car</label>\n");
        CHECK_THROWS_AS(LoadDynamicBBoxes(path), IoError);
    }

    SECTION("wrong transform value count")
    {
        const fs::path path = root / "short_transform.xml";
        WriteTextFile(path, "<?xml version=\"1.0\"?>\n<opencv_storage>\n<object1>\n"
                            "  <label>car</label>\n"
                            "  <semanticId>26</semanticId>\n"
                            "  <timestamp>1</timestamp>\n"
                            "  <dynamic>1</dynamic>\n"
                            "  <start_frame>0</start_frame>\n"
                            "  <end_frame>9</end_frame>\n"
                            "  <transform><data>1 0 0 0 1 0 0 0 1</data></transform>\n"
                            "</object1>\n</opencv_storage>\n");
        CHECK_THROWS_AS(LoadDynamicBBoxes(path), IoError);
    }

    SECTION("object cap")
    {
        const fs::path path = root / "capped.xml";
        WriteTextFile(path,
                      MakeBBoxXml({MakeSpec("car", 150, true), MakeSpec("car", 151, true)}));
        CHECK_THROWS_AS(LoadDynamicBBoxes(path, 1U), IoError);
    }
}

TEST_CASE("PointInBBox honours the transformed unit cube", "[masking]")
{
    const fs::path root = MakeMaskingTempRoot("bbox_membership");
    const fs::path path = root / "boxes.xml";
    WriteTextFile(path, MakeBBoxXml({MakeSpec("car", 150, true)}));

    const std::vector<DynamicBBox> boxes = LoadDynamicBBoxes(path);
    REQUIRE(boxes.size() == 1U);
    CHECK(PointInBBox(Eigen::Vector3d(10.0, 20.0, 30.0), boxes[0]));
    CHECK(PointInBBox(Eigen::Vector3d(10.9, 20.0, 30.0), boxes[0]));
    CHECK_FALSE(PointInBBox(Eigen::Vector3d(11.2, 20.0, 30.0), boxes[0]));
    CHECK_FALSE(PointInBBox(Eigen::Vector3d(100.0, 20.0, 30.0), boxes[0]));
}

TEST_CASE("ApplicableBoxes filters by dynamic flag, frame, and timestamp", "[masking]")
{
    BBoxSpec first = MakeSpec("car", 150, true);
    BBoxSpec second = MakeSpec("car", 151, true);
    BBoxSpec outOfWindow = MakeSpec("car", 150, true);
    outOfWindow.startFrame = 1;
    outOfWindow.endFrame = 5;
    BBoxSpec staticBox = MakeSpec("building", 150, false);

    const fs::path root = MakeMaskingTempRoot("bbox_applicable");
    const fs::path path = root / "boxes.xml";
    WriteTextFile(path, MakeBBoxXml({first, second, outOfWindow, staticBox}));
    const std::vector<DynamicBBox> boxes = LoadDynamicBBoxes(path);

    const std::vector<const DynamicBBox *> applicable =
        ApplicableBoxes(boxes, 150, 2);
    REQUIRE(applicable.size() == 2U);
    CHECK(applicable[0] == &boxes[0]);
    CHECK(applicable[1] == &boxes[1]);

    CHECK(ApplicableBoxes(boxes, 150, 0).size() == 1U);
    CHECK(ApplicableBoxes(boxes, 400, 2).empty());
}
