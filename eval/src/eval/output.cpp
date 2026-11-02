#include <raylidar/eval/output.h>

#include <raylidar/common/io.h>
#include <raylidar/eval/metrics.h>
#include <raylidar/rays/rays.h>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace raylidar::eval
{
namespace
{
namespace fs = std::filesystem;

constexpr int kExitInvalidInput = 4;
constexpr int kExitProtocolDrift = 6;

[[noreturn]] void Reject(const std::string &message, const int exitCode = kExitInvalidInput)
{
    throw IoError(message, exitCode);
}

bool Near(const double left, const double right)
{
    return std::abs(left - right) <= 1.0e-12;
}

std::string TauKey(const double tau)
{
    std::ostringstream buffer;
    buffer.imbue(std::locale::classic());
    buffer << std::fixed << std::setprecision(2) << tau;
    return buffer.str();
}

std::string NumberCell(const std::optional<double> &value)
{
    if (!value.has_value())
    {
        return {};
    }
    if (!std::isfinite(*value))
    {
        Reject("metrics contain a non-finite value");
    }
    std::ostringstream buffer;
    buffer.imbue(std::locale::classic());
    buffer << std::setprecision(std::numeric_limits<double>::max_digits10) << *value;
    return buffer.str();
}

nlohmann::json OptionalNumber(const std::optional<double> &value)
{
    if (!value.has_value())
    {
        return nullptr;
    }
    if (!std::isfinite(*value))
    {
        Reject("metrics contain a non-finite value");
    }
    return *value;
}

nlohmann::json CellJson(const CellMetrics &cell)
{
    nlohmann::json range;
    range["mae"] = OptionalNumber(cell.range.mae);
    range["rmse"] = OptionalNumber(cell.range.rmse);
    range["medae"] = OptionalNumber(cell.range.medae);
    range["bias"] = OptionalNumber(cell.range.bias);
    range["p90"] = OptionalNumber(cell.range.p90);
    range["p95"] = OptionalNumber(cell.range.p95);
    range["n_hits"] = cell.range.n_hits;

    nlohmann::json miss;
    miss["miss_rate"] = OptionalNumber(cell.miss.miss_rate);
    miss["ray_recall"] = OptionalNumber(cell.miss.ray_recall);
    miss["n_total"] = cell.miss.n_total;
    miss["n_hits"] = cell.miss.n_hits;
    miss["n_misses"] = cell.miss.n_misses;

    nlohmann::json out;
    out["n_raw"] = cell.n_raw;
    out["n_valid"] = cell.n_valid;
    out["n_hits"] = cell.n_hits;
    out["n_misses"] = cell.n_misses;
    out["coverage"] = OptionalNumber(cell.coverage);
    out["range"] = std::move(range);
    out["miss"] = std::move(miss);
    return out;
}

nlohmann::json PrfJson(const PrfMetrics &prf)
{
    nlohmann::json out;
    out["precision"] = OptionalNumber(prf.precision);
    out["recall"] = OptionalNumber(prf.recall);
    out["f_score"] = OptionalNumber(prf.f_score);
    out["tp_pred"] = prf.tp_pred;
    out["tp_gt"] = prf.tp_gt;
    out["n_pred"] = prf.n_pred;
    out["n_gt"] = prf.n_gt;
    out["greedy_truncated"] = prf.greedy_truncated;
    return out;
}

nlohmann::json TauPrfBlock(const std::array<TauPrf, 3> &prfs)
{
    if (prfs[0].tau <= 0.0)
    {
        return nullptr; // micro block not computed
    }
    nlohmann::json out = nlohmann::json::object();
    for (const auto &entry : prfs)
    {
        nlohmann::json tau;
        tau["nearest"] = PrfJson(entry.nearest);
        tau["greedy"] = PrfJson(entry.greedy);
        out[TauKey(entry.tau)] = std::move(tau);
    }
    return out;
}

nlohmann::json CdJson(const ChamferMetrics &cd)
{
    nlohmann::json out;
    out["cd_l1"] = OptionalNumber(cd.cd_l1);
    out["cd_l2"] = OptionalNumber(cd.cd_l2);
    out["accuracy_l1"] = OptionalNumber(cd.accuracy_l1);
    out["completeness_l1"] = OptionalNumber(cd.completeness_l1);
    out["accuracy_l2"] = OptionalNumber(cd.accuracy_l2);
    out["completeness_l2"] = OptionalNumber(cd.completeness_l2);
    out["n_pred"] = cd.n_pred;
    out["n_gt"] = cd.n_gt;
    return out;
}

nlohmann::json MacroJson(const MacroSummary &macro)
{
    nlohmann::json out;
    out["n_frames_total"] = macro.n_frames_total;
    out["n_frames_with_range"] = macro.n_frames_with_range;
    out["n_frames_without_range"] = macro.n_frames_without_range;
    out["mae_mean"] = OptionalNumber(macro.mae_mean);
    out["mae_std"] = OptionalNumber(macro.mae_std);
    out["rmse_mean"] = OptionalNumber(macro.rmse_mean);
    out["rmse_std"] = OptionalNumber(macro.rmse_std);
    out["medae_mean"] = OptionalNumber(macro.medae_mean);
    out["medae_std"] = OptionalNumber(macro.medae_std);
    out["bias_mean"] = OptionalNumber(macro.bias_mean);
    out["bias_std"] = OptionalNumber(macro.bias_std);
    out["p90_mean"] = OptionalNumber(macro.p90_mean);
    out["p90_std"] = OptionalNumber(macro.p90_std);
    out["p95_mean"] = OptionalNumber(macro.p95_mean);
    out["p95_std"] = OptionalNumber(macro.p95_std);
    out["miss_rate_mean"] = OptionalNumber(macro.miss_rate_mean);
    out["miss_rate_std"] = OptionalNumber(macro.miss_rate_std);
    out["ray_recall_mean"] = OptionalNumber(macro.ray_recall_mean);
    out["ray_recall_std"] = OptionalNumber(macro.ray_recall_std);
    out["coverage_mean"] = OptionalNumber(macro.coverage_mean);
    out["coverage_std"] = OptionalNumber(macro.coverage_std);
    out["precision_mean"] = OptionalNumber(macro.precision_mean);
    out["precision_std"] = OptionalNumber(macro.precision_std);
    out["recall_mean"] = OptionalNumber(macro.recall_mean);
    out["recall_std"] = OptionalNumber(macro.recall_std);
    out["f_score_mean"] = OptionalNumber(macro.f_score_mean);
    out["f_score_std"] = OptionalNumber(macro.f_score_std);
    return out;
}

nlohmann::json AggregateBlock(const CellMetrics &cell, const std::array<TauPrf, 3> &prf, const ChamferMetrics &cd)
{
    nlohmann::json out;
    out["cell"] = CellJson(cell);
    out["prf"] = TauPrfBlock(prf);
    out["cd"] = CdJson(cd);
    return out;
}

nlohmann::json RowJson(const RowMetrics &row)
{
    nlohmann::json out = CellJson(row.cell);
    out["frame"] = row.frame;
    out["camera"] = row.camera;
    out["bucket"] = rays::BucketName(row.bucket);
    out["prf"] = TauPrfBlock(row.prf);
    out["cd"] = CdJson(row.cd);
    return out;
}

std::vector<RowMetrics> SortedRows(const EvalReport &report)
{
    std::vector<RowMetrics> rows = report.rows;
    std::sort(rows.begin(), rows.end(), [](const RowMetrics &left, const RowMetrics &right) {
        return std::tie(left.frame, left.camera, left.bucket) < std::tie(right.frame, right.camera, right.bucket);
    });
    return rows;
}

std::string JoinFields(const std::vector<std::string> &fields)
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

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

const nlohmann::json &RequireObject(const nlohmann::json &parent, const std::string_view key,
                                    const std::string_view where)
{
    const std::string ownedKey(key);
    const std::string ownedWhere(where);
    if (!parent.is_object() || !parent.contains(ownedKey))
    {
        Reject(ownedWhere + " is missing required key '" + ownedKey + "'");
    }
    const nlohmann::json &value = parent.at(ownedKey);
    if (!value.is_object())
    {
        Reject(ownedWhere + "." + ownedKey + " must be an object");
    }
    return value;
}

const nlohmann::json &RequireArray(const nlohmann::json &parent, const std::string_view key,
                                   const std::string_view where)
{
    const std::string ownedKey(key);
    const std::string ownedWhere(where);
    if (!parent.is_object() || !parent.contains(ownedKey))
    {
        Reject(ownedWhere + " is missing required key '" + ownedKey + "'");
    }
    const nlohmann::json &value = parent.at(ownedKey);
    if (!value.is_array())
    {
        Reject(ownedWhere + "." + ownedKey + " must be an array");
    }
    return value;
}

const nlohmann::json &RequireKey(const nlohmann::json &parent, const std::string_view key, const std::string_view where)
{
    const std::string ownedKey(key);
    const std::string ownedWhere(where);
    if (!parent.is_object() || !parent.contains(ownedKey))
    {
        Reject(ownedWhere + " is missing required key '" + ownedKey + "'");
    }
    return parent.at(ownedKey);
}

std::string RequireString(const nlohmann::json &parent, const std::string_view key, const std::string_view where)
{
    const nlohmann::json &value = RequireKey(parent, key, where);
    if (!value.is_string())
    {
        Reject(std::string(where) + "." + std::string(key) + " must be a string");
    }
    return value.get<std::string>();
}

std::uint64_t RequireUint(const nlohmann::json &parent, const std::string_view key, const std::string_view where)
{
    const nlohmann::json &value = RequireKey(parent, key, where);
    if (value.is_number_unsigned())
    {
        return value.get<std::uint64_t>();
    }
    if (value.is_number_integer())
    {
        const auto signedValue = value.get<std::int64_t>();
        if (signedValue >= 0)
        {
            return static_cast<std::uint64_t>(signedValue);
        }
    }
    Reject(std::string(where) + "." + std::string(key) + " must be a non-negative integer");
}

double RequireNumber(const nlohmann::json &parent, const std::string_view key, const std::string_view where)
{
    const nlohmann::json &value = RequireKey(parent, key, where);
    if (!value.is_number())
    {
        Reject(std::string(where) + "." + std::string(key) + " must be a number");
    }
    const double number = value.get<double>();
    if (!std::isfinite(number))
    {
        Reject(std::string(where) + "." + std::string(key) + " must be finite");
    }
    return number;
}

std::optional<double> RequireOptionalNumber(const nlohmann::json &parent, const std::string_view key,
                                            const std::string_view where)
{
    const nlohmann::json &value = RequireKey(parent, key, where);
    if (value.is_null())
    {
        return std::nullopt;
    }
    if (!value.is_number())
    {
        Reject(std::string(where) + "." + std::string(key) + " must be a number or null");
    }
    const double number = value.get<double>();
    if (!std::isfinite(number))
    {
        Reject(std::string(where) + "." + std::string(key) + " must be finite");
    }
    return number;
}

void RequireFiniteRecursive(const nlohmann::json &value, const std::string &where)
{
    if (value.is_number_float())
    {
        if (!std::isfinite(value.get<double>()))
        {
            Reject(where + " contains a non-finite number");
        }
        return;
    }
    if (value.is_array())
    {
        std::size_t index = 0;
        for (const auto &child : value)
        {
            RequireFiniteRecursive(child, where + "[" + std::to_string(index) + "]");
            ++index;
        }
        return;
    }
    if (value.is_object())
    {
        for (const auto &item : value.items())
        {
            RequireFiniteRecursive(item.value(), where + "." + item.key());
        }
    }
}

bool IsSha256Hex(const std::string &text)
{
    if (text.size() != 64U)
    {
        return false;
    }
    for (const char character : text)
    {
        const bool digit = character >= '0' && character <= '9';
        const bool lower = character >= 'a' && character <= 'f';
        if (!digit && !lower)
        {
            return false;
        }
    }
    return true;
}

int BucketIndexOf(const std::string &name)
{
    for (int index = 0; index < 4; ++index)
    {
        if (name == rays::BucketName(static_cast<rays::Bucket>(index)))
        {
            return index;
        }
    }
    return -1;
}

void ValidateRange(const nlohmann::json &range, const std::uint64_t nHits, const std::string &where)
{
    const std::uint64_t rangeHits = RequireUint(range, "n_hits", where);
    if (rangeHits != nHits)
    {
        Reject(where + ".range.n_hits disagrees with the cell n_hits");
    }
    static const std::array<const char *, 6> keys = {"mae", "rmse", "medae", "bias", "p90", "p95"};
    for (const char *key : keys)
    {
        const std::optional<double> value = RequireOptionalNumber(range, key, where);
        if (nHits == 0U)
        {
            if (value.has_value())
            {
                Reject(where + ".range." + key + " must be null when there are no hits");
            }
        }
        else if (!value.has_value())
        {
            Reject(where + ".range." + key + " must be defined when there are hits");
        }
    }
}

void ValidateMiss(const nlohmann::json &miss, const std::uint64_t nValid, const std::uint64_t nHits,
                  const std::uint64_t nMisses, const std::string &where)
{
    if (RequireUint(miss, "n_total", where) != nValid)
    {
        Reject(where + ".miss.n_total must equal n_valid");
    }
    if (RequireUint(miss, "n_hits", where) != nHits)
    {
        Reject(where + ".miss.n_hits must equal the cell n_hits");
    }
    if (RequireUint(miss, "n_misses", where) != nMisses)
    {
        Reject(where + ".miss.n_misses must equal the cell n_misses");
    }
    const std::optional<double> missRate = RequireOptionalNumber(miss, "miss_rate", where);
    const std::optional<double> rayRecall = RequireOptionalNumber(miss, "ray_recall", where);
    if (nValid == 0U)
    {
        if (missRate.has_value() || rayRecall.has_value())
        {
            Reject(where + ".miss rates must be null when there are no valid rays");
        }
        return;
    }
    if (!missRate.has_value() || !rayRecall.has_value())
    {
        Reject(where + ".miss rates must be defined when there are valid rays");
    }
    if (*missRate < 0.0 || *missRate > 1.0 || *rayRecall < 0.0 || *rayRecall > 1.0)
    {
        Reject(where + ".miss rates must be fractions in [0, 1]");
    }
}

void ValidateCell(const nlohmann::json &cell, const std::string &where)
{
    const std::uint64_t nRaw = RequireUint(cell, "n_raw", where);
    const std::uint64_t nValid = RequireUint(cell, "n_valid", where);
    const std::uint64_t nHits = RequireUint(cell, "n_hits", where);
    const std::uint64_t nMisses = RequireUint(cell, "n_misses", where);
    if (nHits + nMisses != nValid)
    {
        Reject(where + ": n_hits + n_misses must equal n_valid");
    }
    if (nHits > nValid)
    {
        Reject(where + ": n_hits cannot exceed n_valid");
    }
    const std::optional<double> coverage = RequireOptionalNumber(cell, "coverage", where);
    if (nRaw == 0U)
    {
        if (coverage.has_value())
        {
            Reject(where + ".coverage must be null when n_raw is 0");
        }
    }
    else
    {
        if (nValid > nRaw)
        {
            Reject(where + ": n_valid cannot exceed n_raw");
        }
        if (!coverage.has_value())
        {
            Reject(where + ".coverage must be defined when n_raw is positive");
        }
        if (*coverage < 0.0 || *coverage > 1.0)
        {
            Reject(where + ".coverage must be a fraction in [0, 1]");
        }
    }
    ValidateRange(RequireObject(cell, "range", where), nHits, where);
    ValidateMiss(RequireObject(cell, "miss", where), nValid, nHits, nMisses, where);
}

void ValidatePrfMetrics(const nlohmann::json &prf, const std::string &where)
{
    const std::uint64_t tpPred = RequireUint(prf, "tp_pred", where);
    const std::uint64_t tpGt = RequireUint(prf, "tp_gt", where);
    const std::uint64_t nPred = RequireUint(prf, "n_pred", where);
    const std::uint64_t nGt = RequireUint(prf, "n_gt", where);
    if (tpPred > nPred || tpGt > nGt)
    {
        Reject(where + ": true positives cannot exceed the cloud sizes");
    }
    const nlohmann::json &truncated = RequireKey(prf, "greedy_truncated", where);
    if (!truncated.is_boolean())
    {
        Reject(where + ".greedy_truncated must be a boolean");
    }

    const std::optional<double> precision = RequireOptionalNumber(prf, "precision", where);
    const std::optional<double> recall = RequireOptionalNumber(prf, "recall", where);
    const std::optional<double> fScore = RequireOptionalNumber(prf, "f_score", where);

    if (nPred == 0U)
    {
        if (precision.has_value())
        {
            Reject(where + ".precision must be null when the prediction cloud is empty");
        }
    }
    else if (!precision.has_value() || *precision < 0.0 || *precision > 1.0)
    {
        Reject(where + ".precision must be a fraction in [0, 1]");
    }
    if (nGt == 0U)
    {
        if (recall.has_value())
        {
            Reject(where + ".recall must be null when the GT cloud is empty");
        }
    }
    else if (!recall.has_value() || *recall < 0.0 || *recall > 1.0)
    {
        Reject(where + ".recall must be a fraction in [0, 1]");
    }
    if (!precision.has_value() || !recall.has_value())
    {
        if (fScore.has_value())
        {
            Reject(where + ".f_score must be null when precision or recall is null");
        }
        return;
    }
    if (!fScore.has_value() || *fScore < 0.0 || *fScore > 1.0)
    {
        Reject(where + ".f_score must be a fraction in [0, 1]");
    }
    if (*precision + *recall == 0.0 && *fScore != 0.0)
    {
        Reject(where + ".f_score must be 0 when precision + recall is 0");
    }
}

void ValidatePrfBlock(const nlohmann::json &prf, const std::string &where)
{
    if (!prf.is_object())
    {
        Reject(where + ".prf must be an object");
    }
    for (const double tau : kTausM)
    {
        const std::string key = TauKey(tau);
        const nlohmann::json &entry = RequireObject(prf, key, where + ".prf");
        ValidatePrfMetrics(RequireObject(entry, "nearest", where + ".prf." + key), where + ".prf." + key + ".nearest");
        ValidatePrfMetrics(RequireObject(entry, "greedy", where + ".prf." + key), where + ".prf." + key + ".greedy");
    }
}

void ValidateCd(const nlohmann::json &cd, const std::string &where)
{
    const std::uint64_t nPred = RequireUint(cd, "n_pred", where);
    const std::uint64_t nGt = RequireUint(cd, "n_gt", where);
    static const std::array<const char *, 6> keys = {"cd_l1",           "cd_l2",       "accuracy_l1",
                                                     "completeness_l1", "accuracy_l2", "completeness_l2"};
    const bool undefined = nPred == 0U || nGt == 0U;
    for (const char *key : keys)
    {
        const std::optional<double> value = RequireOptionalNumber(cd, key, where);
        if (undefined)
        {
            if (value.has_value())
            {
                Reject(where + "." + key + " must be null when either cloud is empty");
            }
        }
        else if (!value.has_value() || *value < 0.0)
        {
            Reject(where + "." + key + " must be a non-negative distance");
        }
    }
}

void ValidateMacro(const nlohmann::json &macro, const std::string &where)
{
    const std::uint64_t total = RequireUint(macro, "n_frames_total", where);
    const std::uint64_t withRange = RequireUint(macro, "n_frames_with_range", where);
    const std::uint64_t withoutRange = RequireUint(macro, "n_frames_without_range", where);
    if (withRange + withoutRange != total)
    {
        Reject(where + ": frame counts must sum to n_frames_total");
    }
    static const std::array<const char *, 24> keys = {
        "mae_mean",       "mae_std",       "rmse_mean",       "rmse_std",       "medae_mean",    "medae_std",
        "bias_mean",      "bias_std",      "p90_mean",        "p90_std",        "p95_mean",      "p95_std",
        "miss_rate_mean", "miss_rate_std", "ray_recall_mean", "ray_recall_std", "coverage_mean", "coverage_std",
        "precision_mean", "precision_std", "recall_mean",     "recall_std",     "f_score_mean",  "f_score_std"};
    for (const char *key : keys)
    {
        static_cast<void>(RequireOptionalNumber(macro, key, where));
    }
}

void ValidateAggregates(const nlohmann::json &aggregates)
{
    const nlohmann::json &microPrimary = RequireObject(aggregates, "micro_primary", "aggregates");
    ValidateCell(RequireObject(microPrimary, "cell", "aggregates.micro_primary"), "aggregates.micro_primary.cell");
    ValidatePrfBlock(RequireObject(microPrimary, "prf", "aggregates.micro_primary"), "aggregates.micro_primary");
    ValidateCd(RequireObject(microPrimary, "cd", "aggregates.micro_primary"), "aggregates.micro_primary.cd");

    const nlohmann::json &microSecondary = RequireObject(aggregates, "micro_secondary", "aggregates");
    ValidateCell(RequireObject(microSecondary, "cell", "aggregates.micro_secondary"),
                 "aggregates.micro_secondary.cell");
    const nlohmann::json &secondaryPrf = RequireKey(microSecondary, "prf", "aggregates.micro_secondary");
    if (!secondaryPrf.is_null())
    {
        ValidatePrfBlock(secondaryPrf, "aggregates.micro_secondary");
    }
    ValidateCd(RequireObject(microSecondary, "cd", "aggregates.micro_secondary"), "aggregates.micro_secondary.cd");

    const nlohmann::json &cameras = RequireArray(aggregates, "per_camera", "aggregates");
    std::set<std::uint64_t> seen;
    for (std::size_t index = 0; index < cameras.size(); ++index)
    {
        const std::string where = "aggregates.per_camera[" + std::to_string(index) + "]";
        if (!cameras[index].is_object())
        {
            Reject(where + " must be an object");
        }
        const std::uint64_t camera = RequireUint(cameras[index], "camera", where);
        if (camera > 1U)
        {
            Reject(where + ".camera must be 0 or 1");
        }
        if (!seen.insert(camera).second)
        {
            Reject(where + ": duplicate camera block");
        }
        ValidateCell(RequireObject(cameras[index], "cell", where), where + ".cell");
    }

    ValidateMacro(RequireObject(aggregates, "macro", "aggregates"), "aggregates.macro");
}

// ---------------------------------------------------------------------------
// YAML helpers
// ---------------------------------------------------------------------------

YAML::Node RequireProtocolNode(const YAML::Node &root, const char *key, const std::string &where)
{
    const YAML::Node node = root[key];
    if (!node || !node.IsDefined())
    {
        Reject(where + " is missing '" + key + "'", kExitProtocolDrift);
    }
    return node;
}

double YamlDouble(const YAML::Node &node, const char *key, const std::string &where)
{
    const YAML::Node value = RequireProtocolNode(node, key, where);
    if (!value.IsScalar())
    {
        Reject(where + "." + key + " must be a scalar", kExitProtocolDrift);
    }
    try
    {
        return value.as<double>();
    }
    catch (const std::exception &)
    {
        Reject(where + "." + key + " must be a number", kExitProtocolDrift);
    }
}

std::array<double, 3> YamlTaus(const YAML::Node &node, const std::string &where)
{
    const YAML::Node value = RequireProtocolNode(node, "taus_m", where);
    if (!value.IsSequence() || value.size() != 3U)
    {
        Reject(where + ".taus_m must be a sequence of three numbers", kExitProtocolDrift);
    }
    std::array<double, 3> taus = {};
    for (std::size_t index = 0; index < 3U; ++index)
    {
        try
        {
            taus[index] = value[index].as<double>();
        }
        catch (const std::exception &)
        {
            Reject(where + ".taus_m must contain numbers", kExitProtocolDrift);
        }
    }
    return taus;
}

std::pair<double, double> YamlPair(const YAML::Node &node, const std::string &where)
{
    if (!node.IsSequence() || node.size() != 2U)
    {
        Reject(where + " must be a two-element sequence", kExitProtocolDrift);
    }
    try
    {
        return std::make_pair(node[0].as<double>(), node[1].as<double>());
    }
    catch (const std::exception &)
    {
        Reject(where + " must contain two numbers", kExitProtocolDrift);
    }
}

void CheckFrozenProtocol(const ProtocolConfig &protocol)
{
    if (protocol.schema_version != 1)
    {
        Reject("protocol schema_version must be 1", kExitProtocolDrift);
    }
    if (!Near(protocol.t_min_m, rays::kRayTMinM))
    {
        Reject("protocol drift: t_min_m differs from the frozen value", kExitProtocolDrift);
    }
    if (!Near(protocol.r_min_m, rays::kRayMinRangeM))
    {
        Reject("protocol drift: r_min_m differs from the frozen value", kExitProtocolDrift);
    }
    if (!Near(protocol.t_max_m, rays::kRayMaxRangeM))
    {
        Reject("protocol drift: t_max_m differs from the frozen value", kExitProtocolDrift);
    }
    for (std::size_t index = 0; index < kTausM.size(); ++index)
    {
        if (!Near(protocol.taus_m[index], kTausM[index]))
        {
            Reject("protocol drift: taus_m differs from the frozen tau set", kExitProtocolDrift);
        }
    }
    if (!Near(protocol.primary_tau_m, kPrimaryTauM))
    {
        Reject("protocol drift: primary_tau_m differs from the frozen value", kExitProtocolDrift);
    }
    const std::array<std::pair<double, double>, 3> frozenPrimary = {
        std::make_pair(0.0, 20.0), std::make_pair(20.0, 40.0), std::make_pair(40.0, 50.0)};
    for (std::size_t index = 0; index < frozenPrimary.size(); ++index)
    {
        if (!Near(protocol.primary_buckets[index].first, frozenPrimary[index].first) ||
            !Near(protocol.primary_buckets[index].second, frozenPrimary[index].second))
        {
            Reject("protocol drift: primary_buckets differs from the frozen edges", kExitProtocolDrift);
        }
    }
    if (!Near(protocol.secondary_bucket.first, 50.0) || !Near(protocol.secondary_bucket.second, 60.0))
    {
        Reject("protocol drift: secondary_bucket differs from (50, 60]", kExitProtocolDrift);
    }
}

void CheckSiblingProtocol(const ProtocolConfig &protocol, const fs::path &configPath)
{
    const fs::path sibling = configPath.parent_path() / "protocol.yaml";
    std::error_code error;
    if (!fs::is_regular_file(sibling, error))
    {
        return; // stand-alone config (tests)
    }
    YAML::Node root;
    try
    {
        root = raylidar::ReadYaml(sibling);
    }
    catch (const std::exception &)
    {
        Reject("cannot read " + sibling.string(), kExitProtocolDrift);
    }
    const std::array<double, 3> taus = YamlTaus(root, "configs/protocol.yaml");
    for (std::size_t index = 0; index < kTausM.size(); ++index)
    {
        if (!Near(taus[index], protocol.taus_m[index]))
        {
            Reject("protocol drift: taus_m disagrees with configs/protocol.yaml", kExitProtocolDrift);
        }
    }
    if (!Near(YamlDouble(root, "primary_tau_m", "configs/protocol.yaml"), protocol.primary_tau_m))
    {
        Reject("protocol drift: primary_tau_m disagrees with configs/protocol.yaml", kExitProtocolDrift);
    }
    const YAML::Node buckets = RequireProtocolNode(root, "buckets", "configs/protocol.yaml");
    const YAML::Node primary = RequireProtocolNode(buckets, "primary", "configs/protocol.yaml");
    if (!primary.IsSequence() || primary.size() != 3U)
    {
        Reject("protocol drift: configs/protocol.yaml primary buckets changed", kExitProtocolDrift);
    }
    for (std::size_t index = 0; index < 3U; ++index)
    {
        const std::pair<double, double> edge = YamlPair(primary[index], "configs/protocol.yaml.buckets.primary");
        if (!Near(edge.first, protocol.primary_buckets[index].first) ||
            !Near(edge.second, protocol.primary_buckets[index].second))
        {
            Reject("protocol drift: primary buckets disagree with configs/protocol.yaml", kExitProtocolDrift);
        }
    }
    const YAML::Node secondary = RequireProtocolNode(buckets, "secondary", "configs/protocol.yaml");
    if (!secondary.IsSequence() || secondary.size() != 1U)
    {
        Reject("protocol drift: configs/protocol.yaml secondary bucket changed", kExitProtocolDrift);
    }
    const std::pair<double, double> secondaryEdge = YamlPair(secondary[0], "configs/protocol.yaml.buckets.secondary");
    if (!Near(secondaryEdge.first, protocol.secondary_bucket.first) ||
        !Near(secondaryEdge.second, protocol.secondary_bucket.second))
    {
        Reject("protocol drift: secondary bucket disagrees with configs/protocol.yaml", kExitProtocolDrift);
    }
}
} // namespace

const char *MetricsCsvHeader()
{
    return "schema_version,frame,camera,bucket,n_raw,n_valid,n_hits,n_misses,coverage,"
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
}

ProtocolConfig LoadProtocolConfig(const fs::path &path)
{
    const YAML::Node root = raylidar::ReadYaml(path);
    if (!root.IsMap())
    {
        Reject("config " + path.string() + " must be a mapping", kExitInvalidInput);
    }

    ProtocolConfig protocol;
    protocol.schema_version = static_cast<int>(YamlDouble(root, "schema_version", "config"));
    const std::string where = "configs/phase3.yaml.protocol";
    const YAML::Node node = RequireProtocolNode(root, "protocol", "configs/phase3.yaml");
    if (!node.IsMap())
    {
        Reject(where + " must be a mapping", kExitProtocolDrift);
    }
    protocol.protocol_version = static_cast<int>(YamlDouble(node, "protocol_version", where));
    if (protocol.protocol_version < 1)
    {
        Reject(where + ".protocol_version must be >= 1", kExitProtocolDrift);
    }
    protocol.t_min_m = YamlDouble(node, "t_min_m", where);
    protocol.r_min_m = YamlDouble(node, "r_min_m", where);
    protocol.t_max_m = YamlDouble(node, "t_max_m", where);
    protocol.taus_m = YamlTaus(node, where);
    protocol.primary_tau_m = YamlDouble(node, "primary_tau_m", where);

    const YAML::Node primary = RequireProtocolNode(node, "primary_buckets", where);
    if (!primary.IsSequence() || primary.size() != 3U)
    {
        Reject(where + ".primary_buckets must hold three edges", kExitProtocolDrift);
    }
    for (std::size_t index = 0; index < 3U; ++index)
    {
        protocol.primary_buckets[index] = YamlPair(primary[index], where + ".primary_buckets");
    }
    const YAML::Node secondary = RequireProtocolNode(node, "secondary_bucket", where);
    protocol.secondary_bucket = YamlPair(secondary, where + ".secondary_bucket");

    CheckFrozenProtocol(protocol);
    CheckSiblingProtocol(protocol, path);
    return protocol;
}

nlohmann::json BuildMetricsJson(const EvalReport &report, const MetricsInputs &inputs, const ProtocolConfig &protocol,
                                const std::string_view generated_at_utc)
{
    nlohmann::json document;
    document["schema_version"] = protocol.schema_version;
    document["generated_at_utc"] = std::string(generated_at_utc);
    document["mode"] = inputs.mode;

    document["format"] = {
        {"frame", "W"},
        {"units", "m"},
        {"distance", "true_euclidean"},
        {"null_policy", "json_null_empty_csv"},
    };

    nlohmann::json protocolJson;
    protocolJson["schema_version"] = protocol.schema_version;
    protocolJson["protocol_version"] = protocol.protocol_version;
    protocolJson["t_min_m"] = protocol.t_min_m;
    protocolJson["r_min_m"] = protocol.r_min_m;
    protocolJson["t_max_m"] = protocol.t_max_m;
    protocolJson["taus_m"] = nlohmann::json::array({protocol.taus_m[0], protocol.taus_m[1], protocol.taus_m[2]});
    protocolJson["primary_tau_m"] = protocol.primary_tau_m;
    nlohmann::json primaryBuckets = nlohmann::json::array();
    for (const auto &edge : protocol.primary_buckets)
    {
        primaryBuckets.push_back(nlohmann::json::array({edge.first, edge.second}));
    }
    protocolJson["primary_buckets"] = std::move(primaryBuckets);
    protocolJson["secondary_bucket"] =
        nlohmann::json::array({protocol.secondary_bucket.first, protocol.secondary_bucket.second});
    protocolJson["bucket_names"] = nlohmann::json::array({"0_20", "20_40", "40_50", "50_60"});
    protocolJson["percentile_rule"] = "nearest_rank";
    protocolJson["macro_std"] = "sample_n_minus_1";
    protocolJson["prf_primary"] = "nearest_neighbour_within_tau";
    protocolJson["prf_secondary"] = "one_to_one_greedy";
    protocolJson["greedy_candidate_cap"] = kMaxGreedyCandidatesPerPoint;
    protocolJson["cd_convention"] = "symmetric_half_average";
    protocolJson["null_policy"] = "json_null_empty_csv";
    document["protocol"] = std::move(protocolJson);

    nlohmann::json inputsJson;
    inputsJson["mode"] = inputs.mode;
    inputsJson["config_sha256"] = inputs.config_sha256;
    inputsJson["protocol_sha256"] = inputs.protocol_sha256;
    inputsJson["ray_set_sha256"] = inputs.ray_set_sha256;
    inputsJson["mesh_sha256"] = inputs.mesh_sha256;
    inputsJson["hit_set_sha256"] = inputs.hit_set_sha256;
    inputsJson["scene_generator"] = inputs.scene_generator;
    inputsJson["scene_seed"] = inputs.scene_seed;
    inputsJson["scene_box_count"] = inputs.scene_box_count;
    inputsJson["scene_thin_box_count"] = inputs.scene_thin_box_count;
    inputsJson["predictor"] = inputs.predictor;
    inputsJson["predictor_parameters"] = inputs.predictor_parameters;
    document["inputs"] = std::move(inputsJson);

    nlohmann::json aggregateJson;
    aggregateJson["micro_primary"] =
        AggregateBlock(report.micro_primary, report.micro_primary_prf, report.micro_primary_cd);
    aggregateJson["micro_secondary"] =
        AggregateBlock(report.micro_secondary, report.micro_secondary_prf, report.micro_secondary_cd);
    nlohmann::json cameras = nlohmann::json::array();
    for (const auto &camera : report.per_camera)
    {
        nlohmann::json entry;
        entry["camera"] = camera.camera;
        entry["cell"] = CellJson(camera.primary);
        cameras.push_back(std::move(entry));
    }
    aggregateJson["per_camera"] = std::move(cameras);
    aggregateJson["macro"] = MacroJson(report.macro);
    document["aggregates"] = std::move(aggregateJson);

    nlohmann::json rows = nlohmann::json::array();
    for (const auto &row : SortedRows(report))
    {
        rows.push_back(RowJson(row));
    }
    document["rows"] = std::move(rows);

    document["caveats"] =
        nlohmann::json::array({"cd_f_non_independent_under_gt_rays", "published_f_at_5cm_uses_squared_distance"});
    return document;
}

void WriteMetricsJson(const fs::path &path, const nlohmann::json &document)
{
    raylidar::WriteJson(path, document);
}

void WriteMetricsCsv(const fs::path &path, const EvalReport &report, const ProtocolConfig &protocol)
{
    if (path.has_parent_path())
    {
        raylidar::EnsureDir(path.parent_path());
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
    {
        Reject("cannot write " + path.string());
    }
    output << MetricsCsvHeader() << '\n';

    for (const auto &row : SortedRows(report))
    {
        std::vector<std::string> fields;
        fields.reserve(41);
        fields.push_back(std::to_string(protocol.schema_version));
        fields.push_back(std::to_string(row.frame));
        fields.push_back(std::to_string(row.camera));
        fields.push_back(rays::BucketName(row.bucket));
        fields.push_back(std::to_string(row.cell.n_raw));
        fields.push_back(std::to_string(row.cell.n_valid));
        fields.push_back(std::to_string(row.cell.n_hits));
        fields.push_back(std::to_string(row.cell.n_misses));
        fields.push_back(NumberCell(row.cell.coverage));
        fields.push_back(NumberCell(row.cell.range.mae));
        fields.push_back(NumberCell(row.cell.range.rmse));
        fields.push_back(NumberCell(row.cell.range.medae));
        fields.push_back(NumberCell(row.cell.range.bias));
        fields.push_back(NumberCell(row.cell.range.p90));
        fields.push_back(NumberCell(row.cell.range.p95));
        fields.push_back(NumberCell(row.cell.miss.miss_rate));
        fields.push_back(NumberCell(row.cell.miss.ray_recall));
        for (const auto &tau : row.prf)
        {
            fields.push_back(NumberCell(tau.nearest.precision));
            fields.push_back(NumberCell(tau.nearest.recall));
            fields.push_back(NumberCell(tau.nearest.f_score));
            fields.push_back(NumberCell(tau.greedy.precision));
            fields.push_back(NumberCell(tau.greedy.recall));
            fields.push_back(NumberCell(tau.greedy.f_score));
        }
        fields.push_back(NumberCell(row.cd.cd_l1));
        fields.push_back(NumberCell(row.cd.cd_l2));
        fields.push_back(NumberCell(row.cd.accuracy_l1));
        fields.push_back(NumberCell(row.cd.completeness_l1));
        fields.push_back(NumberCell(row.cd.accuracy_l2));
        fields.push_back(NumberCell(row.cd.completeness_l2));
        output << JoinFields(fields) << '\n';
    }
    if (!output)
    {
        Reject("failed to write " + path.string());
    }
}

void ValidateMetricsJson(const nlohmann::json &document)
{
    if (!document.is_object())
    {
        Reject("metrics.json must be an object");
    }
    RequireFiniteRecursive(document, "metrics.json");

    const std::uint64_t schemaVersion = RequireUint(document, "schema_version", "metrics.json");
    if (schemaVersion != 1U)
    {
        Reject("metrics.json schema_version must be 1", kExitProtocolDrift);
    }
    if (RequireString(document, "generated_at_utc", "metrics.json").empty())
    {
        Reject("metrics.json generated_at_utc must not be empty");
    }
    static_cast<void>(RequireString(document, "mode", "metrics.json"));

    const nlohmann::json &format = RequireObject(document, "format", "metrics.json");
    if (RequireString(format, "frame", "metrics.json.format") != "W" ||
        RequireString(format, "units", "metrics.json.format") != "m" ||
        RequireString(format, "distance", "metrics.json.format") != "true_euclidean")
    {
        Reject("metrics.json format must be W/metres/true_euclidean", kExitProtocolDrift);
    }

    const nlohmann::json &protocol = RequireObject(document, "protocol", "metrics.json");
    const ProtocolConfig frozen = [&]() {
        ProtocolConfig expected;
        expected.schema_version = static_cast<int>(RequireUint(protocol, "schema_version", "metrics.json.protocol"));
        expected.protocol_version =
            static_cast<int>(RequireUint(protocol, "protocol_version", "metrics.json.protocol"));
        expected.t_min_m = RequireNumber(protocol, "t_min_m", "metrics.json.protocol");
        expected.r_min_m = RequireNumber(protocol, "r_min_m", "metrics.json.protocol");
        expected.t_max_m = RequireNumber(protocol, "t_max_m", "metrics.json.protocol");
        const nlohmann::json &taus = RequireArray(protocol, "taus_m", "metrics.json.protocol");
        if (taus.size() != 3U)
        {
            Reject("metrics.json.protocol.taus_m must hold three values", kExitProtocolDrift);
        }
        for (std::size_t index = 0; index < 3U; ++index)
        {
            expected.taus_m[index] = taus[index].get<double>();
        }
        expected.primary_tau_m = RequireNumber(protocol, "primary_tau_m", "metrics.json.protocol");
        const nlohmann::json &primary = RequireArray(protocol, "primary_buckets", "metrics.json.protocol");
        if (primary.size() != 3U)
        {
            Reject("metrics.json.protocol.primary_buckets must hold three edges", kExitProtocolDrift);
        }
        for (std::size_t index = 0; index < 3U; ++index)
        {
            if (!primary[index].is_array() || primary[index].size() != 2U)
            {
                Reject("metrics.json.protocol.primary_buckets entries must be pairs", kExitProtocolDrift);
            }
            expected.primary_buckets[index] =
                std::make_pair(primary[index][0].get<double>(), primary[index][1].get<double>());
        }
        const nlohmann::json &secondary = RequireArray(protocol, "secondary_bucket", "metrics.json.protocol");
        if (secondary.size() != 2U)
        {
            Reject("metrics.json.protocol.secondary_bucket must be a pair", kExitProtocolDrift);
        }
        expected.secondary_bucket = std::make_pair(secondary[0].get<double>(), secondary[1].get<double>());
        return expected;
    }();
    CheckFrozenProtocol(frozen);

    const nlohmann::json &inputs = RequireObject(document, "inputs", "metrics.json");
    static const std::array<const char *, 5> hashKeys = {"config_sha256", "protocol_sha256", "ray_set_sha256",
                                                         "mesh_sha256", "hit_set_sha256"};
    for (const char *key : hashKeys)
    {
        const std::string value = RequireString(inputs, key, "metrics.json.inputs");
        if (!IsSha256Hex(value))
        {
            Reject(std::string("metrics.json.inputs.") + key + " must be a lowercase 64-hex SHA-256",
                   kExitProtocolDrift);
        }
    }
    static_cast<void>(RequireString(inputs, "scene_generator", "metrics.json.inputs"));
    static_cast<void>(RequireUint(inputs, "scene_seed", "metrics.json.inputs"));
    static_cast<void>(RequireUint(inputs, "scene_box_count", "metrics.json.inputs"));
    static_cast<void>(RequireUint(inputs, "scene_thin_box_count", "metrics.json.inputs"));
    static_cast<void>(RequireString(inputs, "predictor", "metrics.json.inputs"));

    const nlohmann::json &aggregates = RequireObject(document, "aggregates", "metrics.json");
    ValidateAggregates(aggregates);

    const nlohmann::json &rows = RequireArray(document, "rows", "metrics.json");
    if (rows.empty())
    {
        Reject("metrics.json.rows must not be empty");
    }
    std::set<std::tuple<std::uint32_t, std::uint16_t, int>> seenRows;
    std::set<std::uint32_t> frames;
    std::tuple<std::uint32_t, std::uint16_t, int> previous{};
    bool havePrevious = false;
    for (std::size_t index = 0; index < rows.size(); ++index)
    {
        const std::string where = "metrics.json.rows[" + std::to_string(index) + "]";
        const nlohmann::json &row = rows[index];
        if (!row.is_object())
        {
            Reject(where + " must be an object");
        }
        const std::uint64_t frame = RequireUint(row, "frame", where);
        const std::uint64_t camera = RequireUint(row, "camera", where);
        if (camera > 1U)
        {
            Reject(where + ".camera must be 0 or 1");
        }
        const std::string bucketName = RequireString(row, "bucket", where);
        const int bucket = BucketIndexOf(bucketName);
        if (bucket < 0)
        {
            Reject(where + ".bucket must be one of 0_20|20_40|40_50|50_60");
        }
        const auto key = std::make_tuple(static_cast<std::uint32_t>(frame), static_cast<std::uint16_t>(camera), bucket);
        if (!seenRows.insert(key).second)
        {
            Reject(where + ": duplicate frame x camera x bucket row");
        }
        if (havePrevious && !(previous < key))
        {
            Reject("metrics.json.rows must be ordered by frame, camera, bucket");
        }
        previous = key;
        havePrevious = true;
        frames.insert(static_cast<std::uint32_t>(frame));

        ValidateCell(row, where);
        ValidatePrfBlock(RequireKey(row, "prf", where), where);
        ValidateCd(RequireObject(row, "cd", where), where + ".cd");
    }

    for (const std::uint32_t frame : frames)
    {
        for (std::uint16_t camera = 0; camera < 2U; ++camera)
        {
            for (int bucket = 0; bucket < 4; ++bucket)
            {
                if (seenRows.find(std::make_tuple(frame, camera, bucket)) == seenRows.end())
                {
                    Reject("metrics.json.rows is missing frame " + std::to_string(frame) + ", camera " +
                           std::to_string(camera) + ", bucket " + std::to_string(bucket));
                }
            }
        }
    }

    const nlohmann::json &caveats = RequireArray(document, "caveats", "metrics.json");
    bool sawCdF = false;
    bool sawSquared = false;
    for (const auto &caveat : caveats)
    {
        if (!caveat.is_string())
        {
            Reject("metrics.json.caveats entries must be strings");
        }
        const std::string text = caveat.get<std::string>();
        sawCdF = sawCdF || text == "cd_f_non_independent_under_gt_rays";
        sawSquared = sawSquared || text == "published_f_at_5cm_uses_squared_distance";
    }
    if (!sawCdF || !sawSquared)
    {
        Reject("metrics.json.caveats must contain both frozen caveat identifiers");
    }
}

namespace
{
std::vector<std::string> SplitCsvLine(const std::string &line)
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

bool IsStrictUnsignedInteger(const std::string &text)
{
    if (text.empty())
    {
        return false;
    }
    for (const char character : text)
    {
        if (character < '0' || character > '9')
        {
            return false;
        }
    }
    return true;
}

double ParseFiniteDouble(const std::string &text, const std::string &where)
{
    std::size_t consumed = 0;
    double value = 0.0;
    try
    {
        value = std::stod(text, &consumed);
    }
    catch (const std::exception &)
    {
        Reject(where + " must be a number");
    }
    if (consumed != text.size() || !std::isfinite(value))
    {
        Reject(where + " must be a finite number");
    }
    return value;
}

bool LooksLikeNonFiniteToken(const std::string &text)
{
    std::string lowered;
    lowered.reserve(text.size());
    for (const char character : text)
    {
        lowered.push_back(static_cast<char>(character >= 'A' && character <= 'Z' ? character - 'A' + 'a' : character));
    }
    return lowered == "nan" || lowered == "inf" || lowered == "+inf" || lowered == "-inf" || lowered == "infinity" ||
           lowered == "+infinity" || lowered == "-infinity";
}
} // namespace

void ValidateMetricsCsv(const fs::path &path, const nlohmann::json &document)
{
    // Validate the JSON first: the CSV row set is compared against it.
    ValidateMetricsJson(document);

    std::ifstream input(path, std::ios::binary);
    if (!input.good())
    {
        Reject("cannot read " + path.string());
    }

    std::string header;
    if (!std::getline(input, header))
    {
        Reject("metrics CSV is empty: " + path.string());
    }
    if (!header.empty() && header.back() == '\r')
    {
        header.pop_back();
    }
    if (header != MetricsCsvHeader())
    {
        Reject("metrics CSV header differs from the frozen column list");
    }

    std::vector<std::tuple<std::uint32_t, std::uint16_t, std::string>> expected;
    std::vector<bool> expectedCoverageDefined;
    for (const auto &row : document.at("rows"))
    {
        expected.emplace_back(row.at("frame").get<std::uint32_t>(), row.at("camera").get<std::uint16_t>(),
                              row.at("bucket").get<std::string>());
        expectedCoverageDefined.push_back(!row.at("coverage").is_null());
    }

    std::size_t rowIndex = 0;
    std::string line;
    while (std::getline(input, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        if (line.empty())
        {
            continue;
        }
        const std::string where = "metrics CSV row " + std::to_string(rowIndex + 1U);
        if (rowIndex >= expected.size())
        {
            Reject("metrics CSV has more rows than metrics.json");
        }
        const std::vector<std::string> fields = SplitCsvLine(line);
        if (fields.size() != 41U)
        {
            Reject(where + " has " + std::to_string(fields.size()) + " fields, expected 41");
        }
        for (const auto &field : fields)
        {
            if (LooksLikeNonFiniteToken(field))
            {
                Reject(where + " contains a nan/inf token");
            }
        }
        if (!IsStrictUnsignedInteger(fields[0]) || fields[0] != "1")
        {
            Reject(where + ": schema_version must be 1");
        }
        if (!IsStrictUnsignedInteger(fields[1]) || !IsStrictUnsignedInteger(fields[2]))
        {
            Reject(where + ": frame and camera must be integers");
        }
        for (std::size_t index = 4U; index < 8U; ++index)
        {
            if (!IsStrictUnsignedInteger(fields[index]))
            {
                Reject(where + ": count columns must be integers");
            }
        }
        const auto expectedKey = expected[rowIndex];
        const std::uint32_t frame = static_cast<std::uint32_t>(std::stoul(fields[1]));
        const std::uint16_t camera = static_cast<std::uint16_t>(std::stoul(fields[2]));
        if (frame != std::get<0>(expectedKey) || camera != std::get<1>(expectedKey) ||
            fields[3] != std::get<2>(expectedKey))
        {
            Reject(where + ": frame/camera/bucket disagrees with metrics.json");
        }
        if (expectedCoverageDefined[rowIndex])
        {
            if (fields[8].empty())
            {
                Reject(where + ": coverage must be defined");
            }
            static_cast<void>(ParseFiniteDouble(fields[8], where + " coverage"));
        }
        else if (!fields[8].empty())
        {
            Reject(where + ": coverage must be empty when undefined");
        }
        for (std::size_t index = 9U; index < fields.size(); ++index)
        {
            if (!fields[index].empty())
            {
                static_cast<void>(ParseFiniteDouble(fields[index], where + " field " + std::to_string(index)));
            }
        }
        ++rowIndex;
    }
    if (rowIndex != expected.size())
    {
        Reject("metrics CSV is missing rows present in metrics.json");
    }
}
} // namespace raylidar::eval
