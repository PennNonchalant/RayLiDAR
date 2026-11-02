#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <raylidar/common/io.h>
#include <raylidar/eval/output.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using Catch::Approx;
using raylidar::eval::CellMetrics;
using raylidar::eval::ChamferMetrics;
using raylidar::eval::EvalReport;
using raylidar::eval::MetricsInputs;
using raylidar::eval::PrfMetrics;
using raylidar::eval::ProtocolConfig;
using raylidar::eval::RowMetrics;
using raylidar::eval::TauPrf;
using raylidar::rays::Bucket;

const std::string kHashA = std::string(64, 'a');
const std::string kHashB = std::string(64, 'b');
const std::string kHashC = std::string(64, 'c');
const std::string kHashD = std::string(64, 'd');
const std::string kHashE = std::string(64, 'e');

const char *const kCsvHeader = "schema_version,frame,camera,bucket,n_raw,n_valid,n_hits,n_misses,coverage,"
                               "range_mae_m,range_rmse_m,range_medae_m,range_bias_m,range_p90_m,range_p95_m,"
                               "miss_rate,ray_recall,"
                               "precision_nn_tau_0.05,recall_nn_tau_0.05,f_score_nn_tau_0.05,"
                               "precision_greedy_tau_0.05,recall_greedy_tau_0.05,f_score_greedy_tau_0.05,"
                               "precision_nn_tau_0.20,recall_nn_tau_0.20,f_score_nn_tau_0.20,"
                               "precision_greedy_tau_0.20,recall_greedy_tau_0.20,f_score_greedy_tau_0.20,"
                               "precision_nn_tau_0.50,recall_nn_tau_0.50,f_score_nn_tau_0.50,"
                               "precision_greedy_tau_0.50,recall_greedy_tau_0.50,f_score_greedy_tau_0.50,"
                               "cd_l1_m,cd_l2_m2,cd_accuracy_l1_m,cd_completeness_l1_m,"
                               "cd_accuracy_l2_m2,cd_completeness_l2_m2";

ProtocolConfig MakeProtocol()
{
    ProtocolConfig protocol;
    protocol.schema_version = 1;
    protocol.protocol_version = 1;
    protocol.t_min_m = 1.0e-4;
    protocol.r_min_m = 1.5;
    protocol.t_max_m = 60.0;
    protocol.taus_m = {0.05, 0.20, 0.50};
    protocol.primary_tau_m = 0.20;
    protocol.primary_buckets = {std::make_pair(0.0, 20.0), std::make_pair(20.0, 40.0), std::make_pair(40.0, 50.0)};
    protocol.secondary_bucket = std::make_pair(50.0, 60.0);
    return protocol;
}

MetricsInputs MakeInputs()
{
    MetricsInputs inputs;
    inputs.mode = "placeholder";
    inputs.config_sha256 = kHashA;
    inputs.protocol_sha256 = kHashB;
    inputs.ray_set_sha256 = kHashC;
    inputs.mesh_sha256 = kHashD;
    inputs.hit_set_sha256 = kHashE;
    inputs.scene_generator = "procedural_street_v1";
    inputs.scene_seed = 20261101;
    inputs.scene_box_count = 18;
    inputs.scene_thin_box_count = 8;
    inputs.predictor = "linear_range_bias_scale_v1";
    inputs.predictor_parameters = {{"range_bias_m", 0.02}, {"range_scale", 0.01}, {"miss_every_n", 97}};
    return inputs;
}

PrfMetrics MakePrf(const bool filled)
{
    PrfMetrics prf;
    if (!filled)
    {
        return prf;
    }
    prf.precision = 0.5;
    prf.recall = 1.0 / 3.0;
    prf.f_score = 0.4;
    prf.tp_pred = 1;
    prf.tp_gt = 1;
    prf.n_pred = 2;
    prf.n_gt = 3;
    return prf;
}

ChamferMetrics MakeCd(const bool filled)
{
    ChamferMetrics cd;
    if (!filled)
    {
        return cd;
    }
    cd.cd_l1 = 0.10;
    cd.cd_l2 = 0.02;
    cd.accuracy_l1 = 0.09;
    cd.completeness_l1 = 0.11;
    cd.accuracy_l2 = 0.018;
    cd.completeness_l2 = 0.022;
    cd.n_pred = 2;
    cd.n_gt = 3;
    return cd;
}

CellMetrics MakeFilledCell()
{
    CellMetrics cell;
    cell.n_raw = 0;
    cell.n_valid = 3;
    cell.n_hits = 2;
    cell.n_misses = 1;
    cell.coverage = std::nullopt;
    cell.range.mae = 0.02;
    cell.range.rmse = 0.03;
    cell.range.medae = 0.025;
    cell.range.bias = 0.01;
    cell.range.p90 = 0.04;
    cell.range.p95 = 0.05;
    cell.range.n_hits = 2;
    cell.miss.miss_rate = 1.0 / 3.0;
    cell.miss.ray_recall = 2.0 / 3.0;
    cell.miss.n_total = 3;
    cell.miss.n_hits = 2;
    cell.miss.n_misses = 1;
    return cell;
}

RowMetrics MakeRow(const std::uint32_t frame, const std::uint16_t camera, const int bucket, const bool filled)
{
    RowMetrics row;
    row.frame = frame;
    row.camera = camera;
    row.bucket = static_cast<Bucket>(bucket);
    if (filled)
    {
        row.cell = MakeFilledCell();
        for (std::size_t index = 0; index < row.prf.size(); ++index)
        {
            row.prf[index].tau = raylidar::eval::kTausM[index];
            row.prf[index].nearest = MakePrf(true);
            row.prf[index].greedy = MakePrf(true);
        }
        row.cd = MakeCd(true);
    }
    else
    {
        for (std::size_t index = 0; index < row.prf.size(); ++index)
        {
            row.prf[index].tau = raylidar::eval::kTausM[index];
        }
    }
    return row;
}

EvalReport MakeReport()
{
    EvalReport report;
    report.micro_primary = MakeFilledCell();
    report.micro_secondary = CellMetrics{};
    for (std::size_t index = 0; index < report.micro_primary_prf.size(); ++index)
    {
        report.micro_primary_prf[index].tau = raylidar::eval::kTausM[index];
        report.micro_primary_prf[index].nearest = MakePrf(true);
        report.micro_primary_prf[index].greedy = MakePrf(true);
    }
    report.micro_primary_cd = MakeCd(true);

    raylidar::eval::FrameMetrics frame;
    frame.frame = 0;
    frame.primary = MakeFilledCell();
    for (std::size_t index = 0; index < frame.buckets.size(); ++index)
    {
        frame.buckets[index] = index == 0U ? MakeFilledCell() : CellMetrics{};
    }
    report.per_frame.push_back(frame);

    raylidar::eval::CameraMetrics camera0;
    camera0.camera = 0;
    camera0.primary = MakeFilledCell();
    report.per_camera.push_back(camera0);
    raylidar::eval::CameraMetrics camera1;
    camera1.camera = 1;
    camera1.primary = MakeFilledCell();
    report.per_camera.push_back(camera1);

    report.macro.n_frames_total = 1;
    report.macro.n_frames_with_range = 1;
    report.macro.n_frames_without_range = 0;
    report.macro.mae_mean = 0.02;
    report.macro.rmse_mean = 0.03;
    report.macro.miss_rate_mean = 1.0 / 3.0;
    report.macro.ray_recall_mean = 2.0 / 3.0;
    report.macro.coverage_mean = 0.75;

    for (std::uint16_t camera = 0; camera < 2U; ++camera)
    {
        for (int bucket = 0; bucket < 4; ++bucket)
        {
            report.rows.push_back(MakeRow(0, camera, bucket, bucket == 0));
        }
    }
    return report;
}

struct TempDir
{
    TempDir()
    {
        static std::uint64_t counter = 0;
        static std::random_device device;
        path = fs::temp_directory_path() /
               ("raylidar-eval-output-" + std::to_string(device()) + "-" + std::to_string(counter++));
        fs::create_directories(path);
    }
    ~TempDir()
    {
        std::error_code error;
        fs::remove_all(path, error);
    }
    fs::path path;
};

std::vector<std::string> ReadLines(const fs::path &path)
{
    std::ifstream input(path);
    REQUIRE(input.good());
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line))
    {
        lines.push_back(line);
    }
    return lines;
}

void WriteLines(const fs::path &path, const std::vector<std::string> &lines)
{
    std::ofstream output(path, std::ios::trunc);
    REQUIRE(output.good());
    for (const auto &line : lines)
    {
        output << line << '\n';
    }
}

std::vector<std::string> SplitCsv(const std::string &line)
{
    std::vector<std::string> fields;
    std::string field;
    for (const char character : line)
    {
        if (character == ',')
        {
            fields.push_back(field);
            field.clear();
        }
        else
        {
            field.push_back(character);
        }
    }
    fields.push_back(field);
    return fields;
}

std::string JoinCsv(const std::vector<std::string> &fields)
{
    std::string line;
    for (std::size_t index = 0; index < fields.size(); ++index)
    {
        if (index != 0U)
        {
            line.push_back(',');
        }
        line += fields[index];
    }
    return line;
}

void RequireJsonReject(const nlohmann::json &document, const int exitCode)
{
    bool rejected = false;
    try
    {
        raylidar::eval::ValidateMetricsJson(document);
    }
    catch (const raylidar::IoError &error)
    {
        rejected = true;
        CHECK(error.exitCode() == exitCode);
    }
    CHECK(rejected);
}

void RequireCsvReject(const fs::path &csvPath, const nlohmann::json &document, const int exitCode)
{
    bool rejected = false;
    try
    {
        raylidar::eval::ValidateMetricsCsv(csvPath, document);
    }
    catch (const raylidar::IoError &error)
    {
        rejected = true;
        CHECK(error.exitCode() == exitCode);
    }
    CHECK(rejected);
}
} // namespace

TEST_CASE("metrics.json carries schema, hashes, protocol, aggregates, rows, caveats", "[eval][output]")
{
    const EvalReport report = MakeReport();
    const nlohmann::json document =
        raylidar::eval::BuildMetricsJson(report, MakeInputs(), MakeProtocol(), "2026-11-01T00:00:00Z");

    CHECK(document.at("schema_version").get<int>() == 1);
    CHECK(document.at("generated_at_utc").get<std::string>() == "2026-11-01T00:00:00Z");

    const nlohmann::json &inputs = document.at("inputs");
    CHECK(inputs.at("config_sha256").get<std::string>() == kHashA);
    CHECK(inputs.at("protocol_sha256").get<std::string>() == kHashB);
    CHECK(inputs.at("ray_set_sha256").get<std::string>() == kHashC);
    CHECK(inputs.at("mesh_sha256").get<std::string>() == kHashD);
    CHECK(inputs.at("hit_set_sha256").get<std::string>() == kHashE);
    CHECK(inputs.at("scene_generator").get<std::string>() == "procedural_street_v1");
    CHECK(inputs.at("scene_seed").get<std::uint64_t>() == 20261101U);
    CHECK(inputs.at("scene_box_count").get<std::uint64_t>() == 18U);
    CHECK(inputs.at("scene_thin_box_count").get<std::uint64_t>() == 8U);
    CHECK(inputs.at("predictor").get<std::string>() == "linear_range_bias_scale_v1");
    CHECK(inputs.at("predictor_parameters").at("range_bias_m").get<double>() == Approx(0.02));

    const nlohmann::json &protocol = document.at("protocol");
    CHECK(protocol.at("t_min_m").get<double>() == Approx(1.0e-4));
    CHECK(protocol.at("r_min_m").get<double>() == Approx(1.5));
    CHECK(protocol.at("t_max_m").get<double>() == Approx(60.0));
    CHECK(protocol.at("primary_tau_m").get<double>() == Approx(0.20));
    CHECK(protocol.at("taus_m").size() == 3U);

    CHECK(document.at("format").at("units").get<std::string>() == "m");
    CHECK(document.at("format").at("frame").get<std::string>() == "W");

    const nlohmann::json &aggregates = document.at("aggregates");
    CHECK(aggregates.contains("micro_primary"));
    CHECK(aggregates.contains("micro_secondary"));
    CHECK(aggregates.contains("macro"));
    CHECK(aggregates.at("per_camera").size() == 2U);
    CHECK(aggregates.at("micro_primary").at("cell").at("n_valid").get<std::uint64_t>() == 3U);
    CHECK(aggregates.at("macro").at("n_frames_total").get<std::uint64_t>() == 1U);

    REQUIRE(document.at("rows").size() == 8U);
    CHECK(document.at("rows")[0].at("frame").get<std::uint32_t>() == 0U);
    CHECK(document.at("rows")[0].at("camera").get<std::uint16_t>() == 0U);
    CHECK(document.at("rows")[0].at("bucket").get<std::string>() == "0_20");
    CHECK(document.at("rows")[7].at("camera").get<std::uint16_t>() == 1U);
    CHECK(document.at("rows")[7].at("bucket").get<std::string>() == "50_60");

    const nlohmann::json &caveats = document.at("caveats");
    REQUIRE(caveats.size() == 2U);
    CHECK(caveats[0].get<std::string>() == "cd_f_non_independent_under_gt_rays");
    CHECK(caveats[1].get<std::string>() == "published_f_at_5cm_uses_squared_distance");
}

TEST_CASE("nullopt serializes to JSON null and an empty CSV field, never NaN/Inf", "[eval][output]")
{
    const EvalReport report = MakeReport();
    const nlohmann::json document =
        raylidar::eval::BuildMetricsJson(report, MakeInputs(), MakeProtocol(), "2026-11-01T00:00:00Z");

    // Empty bucket: every undefined metric is JSON null, not 0.
    const nlohmann::json &emptyRow = document.at("rows")[1];
    CHECK(emptyRow.at("coverage").is_null());
    CHECK(emptyRow.at("range").at("mae").is_null());
    CHECK(emptyRow.at("prf").at("0.20").at("nearest").at("precision").is_null());
    CHECK(emptyRow.at("cd").at("cd_l1").is_null());

    const std::string dumped = document.dump();
    CHECK(dumped.find("nan") == std::string::npos);
    CHECK(dumped.find("NaN") == std::string::npos);
    CHECK(dumped.find("inf") == std::string::npos);
    CHECK(dumped.find("Inf") == std::string::npos);

    TempDir temp;
    const fs::path csvPath = temp.path / "metrics.csv";
    raylidar::eval::WriteMetricsCsv(csvPath, report, MakeProtocol());
    const std::vector<std::string> lines = ReadLines(csvPath);
    REQUIRE(lines.size() == 9U);
    const std::vector<std::string> emptyFields = SplitCsv(lines[2]);
    REQUIRE(emptyFields.size() == 41U);
    for (std::size_t index = 8U; index < emptyFields.size(); ++index)
    {
        CHECK(emptyFields[index].empty());
    }
    for (const auto &line : lines)
    {
        CHECK(line.find("nan") == std::string::npos);
        CHECK(line.find("NaN") == std::string::npos);
        CHECK(line.find("inf") == std::string::npos);
        CHECK(line.find("Inf") == std::string::npos);
    }
}

TEST_CASE("CSV has the frozen header and one ordered row per frame/camera/bucket", "[eval][output]")
{
    const EvalReport report = MakeReport();
    TempDir temp;
    const fs::path csvPath = temp.path / "metrics.csv";
    raylidar::eval::WriteMetricsCsv(csvPath, report, MakeProtocol());

    const std::vector<std::string> lines = ReadLines(csvPath);
    REQUIRE(lines.size() == 9U);
    CHECK(lines[0] == std::string(kCsvHeader));

    int expectedRow = 1;
    for (std::uint16_t camera = 0; camera < 2U; ++camera)
    {
        for (int bucket = 0; bucket < 4; ++bucket)
        {
            const std::vector<std::string> fields = SplitCsv(lines[expectedRow]);
            REQUIRE(fields.size() == 41U);
            CHECK(fields[0] == "1");
            CHECK(fields[1] == "0");
            CHECK(fields[2] == std::to_string(camera));
            CHECK(fields[3] == raylidar::rays::BucketName(static_cast<Bucket>(bucket)));
            ++expectedRow;
        }
    }

    // Empty bucket: zero counts and empty coverage/metric cells.
    const std::vector<std::string> emptyFields = SplitCsv(lines[2]);
    CHECK(emptyFields[4] == "0"); // n_raw
    CHECK(emptyFields[5] == "0"); // n_valid
    CHECK(emptyFields[6] == "0"); // n_hits
    CHECK(emptyFields[7] == "0"); // n_misses
    CHECK(emptyFields[8].empty());

    // Populated row: integers for counts, a number for coverage when defined.
    const std::vector<std::string> filledFields = SplitCsv(lines[1]);
    CHECK(filledFields[5] == "3");
    CHECK(filledFields[6] == "2");
    CHECK(filledFields[7] == "1");
    CHECK(std::stod(filledFields[9]) == Approx(0.02));
}

TEST_CASE("ValidateMetricsJson accepts the built document and rejects every drift", "[eval][output]")
{
    const EvalReport report = MakeReport();
    const nlohmann::json document =
        raylidar::eval::BuildMetricsJson(report, MakeInputs(), MakeProtocol(), "2026-11-01T00:00:00Z");
    REQUIRE_NOTHROW(raylidar::eval::ValidateMetricsJson(document));

    SECTION("missing required top-level key")
    {
        nlohmann::json mutated = document;
        mutated.erase("caveats");
        RequireJsonReject(mutated, 4);
    }
    SECTION("bad 64-hex hash")
    {
        nlohmann::json mutated = document;
        mutated["inputs"]["config_sha256"] = "not-a-sha256";
        RequireJsonReject(mutated, 6);
    }
    SECTION("protocol drift in the recorded block")
    {
        nlohmann::json mutated = document;
        mutated["protocol"]["t_max_m"] = 50.0;
        RequireJsonReject(mutated, 6);
    }
    SECTION("non-finite value")
    {
        nlohmann::json mutated = document;
        mutated["rows"][0]["range"]["mae"] = std::numeric_limits<double>::quiet_NaN();
        RequireJsonReject(mutated, 4);
    }
    SECTION("missing row")
    {
        nlohmann::json mutated = document;
        mutated["rows"].erase(mutated["rows"].end() - 1);
        RequireJsonReject(mutated, 4);
    }
    SECTION("duplicate row")
    {
        nlohmann::json mutated = document;
        mutated["rows"].push_back(mutated["rows"][0]);
        RequireJsonReject(mutated, 4);
    }
    SECTION("counts do not sum to n_valid")
    {
        nlohmann::json mutated = document;
        mutated["rows"][0]["n_hits"] = 99;
        RequireJsonReject(mutated, 4);
    }
    SECTION("row without a coverage field")
    {
        nlohmann::json mutated = document;
        mutated["rows"][0].erase("coverage");
        RequireJsonReject(mutated, 4);
    }
    SECTION("undefined metric encoded as zero")
    {
        nlohmann::json mutated = document;
        mutated["rows"][2]["range"]["mae"] = 0.0;
        RequireJsonReject(mutated, 4);
    }
    SECTION("missing caveat identifier")
    {
        nlohmann::json mutated = document;
        mutated["caveats"] = nlohmann::json::array({"cd_f_non_independent_under_gt_rays"});
        RequireJsonReject(mutated, 4);
    }
}

TEST_CASE("ValidateMetricsCsv accepts the written CSV and rejects every drift", "[eval][output]")
{
    const EvalReport report = MakeReport();
    const nlohmann::json document =
        raylidar::eval::BuildMetricsJson(report, MakeInputs(), MakeProtocol(), "2026-11-01T00:00:00Z");

    TempDir temp;
    const fs::path csvPath = temp.path / "metrics.csv";
    raylidar::eval::WriteMetricsCsv(csvPath, report, MakeProtocol());
    REQUIRE_NOTHROW(raylidar::eval::ValidateMetricsCsv(csvPath, document));

    SECTION("wrong header")
    {
        std::vector<std::string> lines = ReadLines(csvPath);
        lines[0] = "schema_version,frame,camera,bucket";
        WriteLines(csvPath, lines);
        RequireCsvReject(csvPath, document, 4);
    }
    SECTION("missing row")
    {
        std::vector<std::string> lines = ReadLines(csvPath);
        lines.erase(lines.begin() + 3);
        WriteLines(csvPath, lines);
        RequireCsvReject(csvPath, document, 4);
    }
    SECTION("duplicate row")
    {
        std::vector<std::string> lines = ReadLines(csvPath);
        lines.push_back(lines[1]);
        WriteLines(csvPath, lines);
        RequireCsvReject(csvPath, document, 4);
    }
    SECTION("non-integer count")
    {
        std::vector<std::string> lines = ReadLines(csvPath);
        std::vector<std::string> fields = SplitCsv(lines[1]);
        fields[5] = "1.5";
        lines[1] = JoinCsv(fields);
        WriteLines(csvPath, lines);
        RequireCsvReject(csvPath, document, 4);
    }
    SECTION("nan token in a metric cell")
    {
        std::vector<std::string> lines = ReadLines(csvPath);
        std::vector<std::string> fields = SplitCsv(lines[1]);
        fields[9] = "nan";
        lines[1] = JoinCsv(fields);
        WriteLines(csvPath, lines);
        RequireCsvReject(csvPath, document, 4);
    }
    SECTION("CSV row set disagrees with the JSON row set")
    {
        std::vector<std::string> lines = ReadLines(csvPath);
        std::vector<std::string> fields = SplitCsv(lines[1]);
        fields[3] = "20_40";
        lines[1] = JoinCsv(fields);
        WriteLines(csvPath, lines);
        RequireCsvReject(csvPath, document, 4);
    }
}
