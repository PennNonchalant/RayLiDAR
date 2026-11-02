#include <raylidar/calib/kitti360.h>

#include <raylidar/calib/frames.h>
#include <raylidar/common/io.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace raylidar::calib
{
namespace
{
namespace fs = std::filesystem;

constexpr int kImageWidth = 1408;
constexpr int kImageHeight = 376;

std::string FrameName(int frame)
{
    std::ostringstream out;
    out << std::setw(10) << std::setfill('0') << frame;
    return out.str();
}

std::string Trim(std::string value)
{
    const auto notSpace = [](unsigned char ch) { return std::isspace(ch) == 0; };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), notSpace));
    value.erase(std::find_if(value.rbegin(), value.rend(), notSpace).base(), value.end());
    return value;
}

std::vector<std::string> SplitWhitespace(const std::string &text)
{
    std::istringstream stream(text);
    std::vector<std::string> tokens;
    std::string token;
    while (stream >> token)
    {
        tokens.push_back(token);
    }
    return tokens;
}

[[noreturn]] void ParseError(const fs::path &path, std::size_t line, const std::string &message)
{
    throw IoError(path.string() + ":" + std::to_string(line) + ": " + message);
}

double ParseDouble(const fs::path &path, std::size_t line, const std::string &token)
{
    if (token.empty())
    {
        ParseError(path, line, "empty numeric token");
    }
    char *end = nullptr;
    const double value = std::strtod(token.c_str(), &end);
    if (end == token.c_str() || *end != '\0' || !std::isfinite(value))
    {
        ParseError(path, line, "invalid numeric token '" + token + "'");
    }
    return value;
}

std::vector<double> ParseDoubles(const fs::path &path, std::size_t line, const std::vector<std::string> &tokens)
{
    std::vector<double> values;
    values.reserve(tokens.size());
    for (const std::string &token : tokens)
    {
        values.push_back(ParseDouble(path, line, token));
    }
    return values;
}

std::ifstream OpenInput(const fs::path &path)
{
    std::ifstream input(path);
    if (!input)
    {
        throw IoError("cannot open " + path.string(), 4);
    }
    return input;
}

Eigen::Matrix4d FromRowMajor12(const std::vector<double> &values)
{
    Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            matrix(row, col) = values[static_cast<std::size_t>(row * 4 + col)];
        }
    }
    return matrix;
}

std::string CamNameFromSuffix(const std::string &suffix)
{
    if (suffix.rfind("image_", 0) == 0)
    {
        return suffix;
    }
    return "image_" + suffix;
}

fs::path ImagePath(const fs::path &root, std::string_view sequence, const std::string &camera, int frame)
{
    return root / "data_2d_raw" / std::string(sequence) / camera / "data_rect" / (FrameName(frame) + ".png");
}

fs::path LidarPath(const fs::path &root, std::string_view sequence, int frame)
{
    return root / "data_3d_raw" / std::string(sequence) / "velodyne_points" / "data" / (FrameName(frame) + ".bin");
}

bool IsRegularFile(const fs::path &path)
{
    std::error_code error;
    return fs::is_regular_file(path, error);
}
} // namespace

std::map<std::string, Eigen::Matrix4d> LoadCalibCamToPose(const std::filesystem::path &path)
{
    std::ifstream input = OpenInput(path);
    std::map<std::string, Eigen::Matrix4d> result;
    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(input, line))
    {
        ++lineNumber;
        const std::string trimmed = Trim(line);
        if (trimmed.empty() || trimmed.front() == '#')
        {
            continue;
        }
        const std::size_t colon = trimmed.find(':');
        if (colon == std::string::npos)
        {
            ParseError(path, lineNumber, "expected 'camera: 12 values'");
        }
        const std::string name = Trim(trimmed.substr(0, colon));
        if (name.empty())
        {
            ParseError(path, lineNumber, "empty camera name");
        }
        const std::vector<std::string> tokens = SplitWhitespace(trimmed.substr(colon + 1));
        if (tokens.size() != 12U)
        {
            ParseError(path, lineNumber, "expected 12 values for " + name + ", got " + std::to_string(tokens.size()));
        }
        const std::vector<double> values = ParseDoubles(path, lineNumber, tokens);
        if (result.find(name) != result.end())
        {
            ParseError(path, lineNumber, "duplicate camera entry '" + name + "'");
        }
        result.emplace(name, FromRowMajor12(values));
    }
    if (result.empty())
    {
        throw IoError(path.string() + ": no camera entries found", 4);
    }
    return result;
}

Eigen::Matrix4d LoadCalibCamToVelo(const std::filesystem::path &path)
{
    std::ifstream input = OpenInput(path);
    std::vector<double> values;
    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(input, line))
    {
        ++lineNumber;
        const std::string trimmed = Trim(line);
        if (trimmed.empty() || trimmed.front() == '#')
        {
            continue;
        }
        const std::vector<std::string> tokens = SplitWhitespace(trimmed);
        const std::vector<double> parsed = ParseDoubles(path, lineNumber, tokens);
        values.insert(values.end(), parsed.begin(), parsed.end());
    }
    if (values.size() != 12U)
    {
        throw IoError(path.string() + ": expected exactly 12 values, got " + std::to_string(values.size()), 4);
    }
    return FromRowMajor12(values);
}

std::pair<std::map<std::string, Eigen::Matrix<double, 3, 4>>, std::map<std::string, Eigen::Matrix3d>> LoadPerspective(
    const std::filesystem::path &path)
{
    std::ifstream input = OpenInput(path);
    std::map<std::string, Eigen::Matrix<double, 3, 4>> projections;
    std::map<std::string, Eigen::Matrix3d> rectifications;
    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(input, line))
    {
        ++lineNumber;
        const std::string trimmed = Trim(line);
        if (trimmed.empty() || trimmed.front() == '#')
        {
            continue;
        }
        const std::size_t colon = trimmed.find(':');
        if (colon == std::string::npos)
        {
            continue;
        }
        const std::string key = Trim(trimmed.substr(0, colon));
        const std::vector<std::string> tokens = SplitWhitespace(trimmed.substr(colon + 1));
        if (key.rfind("P_rect_", 0) == 0)
        {
            if (tokens.size() != 12U)
            {
                ParseError(path, lineNumber,
                           "expected 12 values for " + key + ", got " + std::to_string(tokens.size()));
            }
            const std::vector<double> values = ParseDoubles(path, lineNumber, tokens);
            Eigen::Matrix<double, 3, 4> matrix;
            for (int row = 0; row < 3; ++row)
            {
                for (int col = 0; col < 4; ++col)
                {
                    matrix(row, col) = values[static_cast<std::size_t>(row * 4 + col)];
                }
            }
            projections[CamNameFromSuffix(key.substr(7))] = matrix;
        }
        else if (key.rfind("R_rect_", 0) == 0)
        {
            if (tokens.size() != 9U)
            {
                ParseError(path, lineNumber, "expected 9 values for " + key + ", got " + std::to_string(tokens.size()));
            }
            const std::vector<double> values = ParseDoubles(path, lineNumber, tokens);
            Eigen::Matrix3d matrix;
            for (int row = 0; row < 3; ++row)
            {
                for (int col = 0; col < 3; ++col)
                {
                    matrix(row, col) = values[static_cast<std::size_t>(row * 3 + col)];
                }
            }
            rectifications[CamNameFromSuffix(key.substr(7))] = matrix;
        }
    }
    if (projections.empty())
    {
        throw IoError(path.string() + ": no P_rect entries found", 4);
    }
    if (rectifications.empty())
    {
        throw IoError(path.string() + ": no R_rect entries found", 4);
    }
    return {projections, rectifications};
}

std::map<int, Eigen::Matrix4d> LoadPoses(const std::filesystem::path &path)
{
    std::ifstream input = OpenInput(path);
    std::map<int, Eigen::Matrix4d> result;
    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(input, line))
    {
        ++lineNumber;
        const std::string trimmed = Trim(line);
        if (trimmed.empty() || trimmed.front() == '#')
        {
            continue;
        }
        const std::vector<std::string> tokens = SplitWhitespace(trimmed);
        if (tokens.size() != 13U)
        {
            ParseError(path, lineNumber,
                       "expected 13 tokens (frame + 12 values), got " + std::to_string(tokens.size()));
        }
        const std::vector<double> values = ParseDoubles(path, lineNumber, tokens);
        const int frame = static_cast<int>(std::llround(values[0]));
        std::vector<double> rest(values.begin() + 1, values.end());
        if (result.find(frame) != result.end())
        {
            ParseError(path, lineNumber, "duplicate frame " + std::to_string(frame));
        }
        result.emplace(frame, FromRowMajor12(rest));
    }
    return result;
}

std::map<int, Eigen::Matrix4d> LoadCam0ToWorld(const std::filesystem::path &path)
{
    std::ifstream input = OpenInput(path);
    std::map<int, Eigen::Matrix4d> result;
    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(input, line))
    {
        ++lineNumber;
        const std::string trimmed = Trim(line);
        if (trimmed.empty() || trimmed.front() == '#')
        {
            continue;
        }
        const std::vector<std::string> tokens = SplitWhitespace(trimmed);
        if (tokens.size() != 17U)
        {
            ParseError(path, lineNumber,
                       "expected 17 tokens (frame + 16 values), got " + std::to_string(tokens.size()));
        }
        const std::vector<double> values = ParseDoubles(path, lineNumber, tokens);
        const int frame = static_cast<int>(std::llround(values[0]));
        if (result.find(frame) != result.end())
        {
            ParseError(path, lineNumber, "duplicate frame " + std::to_string(frame));
        }
        Eigen::Matrix4d matrix;
        for (int row = 0; row < 4; ++row)
        {
            for (int col = 0; col < 4; ++col)
            {
                matrix(row, col) = values[static_cast<std::size_t>(1 + row * 4 + col)];
            }
        }
        result.emplace(frame, matrix);
    }
    return result;
}

Eigen::Matrix4d Rect4(const Eigen::Matrix3d &R)
{
    Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();
    matrix.topLeftCorner<3, 3>() = R;
    return matrix;
}

Kitti360Calib LoadKitti360Calib(const std::filesystem::path &calibrationDir)
{
    Kitti360Calib calib;
    calib.T_V_C0 = LoadCalibCamToVelo(calibrationDir / "calib_cam_to_velo.txt");
    calib.T_I_Ck = LoadCalibCamToPose(calibrationDir / "calib_cam_to_pose.txt");
    auto perspective = LoadPerspective(calibrationDir / "perspective.txt");
    calib.P_rect = std::move(perspective.first);
    calib.R_rect = std::move(perspective.second);

    for (const auto &[camera, projection] : calib.P_rect)
    {
        const double fx = projection(0, 0);
        const double fy = projection(1, 1);
        const double cx = projection(0, 2);
        const double cy = projection(1, 2);
        if (!(fx > 0.0) || !(fy > 0.0))
        {
            throw IoError(calibrationDir.string() + ": " + camera + " has a non-positive focal length", 4);
        }
        if (!(cx > 0.0 && cx < static_cast<double>(kImageWidth)) ||
            !(cy > 0.0 && cy < static_cast<double>(kImageHeight)))
        {
            throw IoError(calibrationDir.string() + ": " + camera + " principal point is outside the 1408x376 image",
                          4);
        }
        calib.K_rect[camera] = projection.leftCols<3>();
    }

    for (const auto &[camera, rectification] : calib.R_rect)
    {
        try
        {
            AssertRotation(rectification);
        }
        catch (const std::exception &error)
        {
            throw std::runtime_error("R_rect_" + camera + " is not orthonormal: " + error.what());
        }
    }

    // calib_cam_to_pose.txt also describes image_02/image_03 (the fisheye
    // cameras), but perspective.txt only provides rectification for image_00
    // and image_01. Require the reverse direction instead: every camera that has
    // a rectified projection must also have a pose entry.
    for (const auto &[camera, projection] : calib.P_rect)
    {
        (void)projection;
        if (calib.T_I_Ck.find(camera) == calib.T_I_Ck.end())
        {
            throw IoError(calibrationDir.string() + ": " + camera + " has P_rect but no calib_cam_to_pose entry", 4);
        }
        if (calib.R_rect.find(camera) == calib.R_rect.end())
        {
            throw IoError(calibrationDir.string() + ": " + camera + " has P_rect but no R_rect", 4);
        }
    }
    if (calib.T_I_Ck.find("image_00") == calib.T_I_Ck.end())
    {
        throw IoError(calibrationDir.string() + ": calib_cam_to_pose.txt has no image_00", 4);
    }
    return calib;
}

Eigen::Matrix4d ComposeTWV(const Kitti360Calib &calib, const Eigen::Matrix4d &T_W_I)
{
    const auto camera = calib.T_I_Ck.find("image_00");
    if (camera == calib.T_I_Ck.end())
    {
        throw IoError("calibration is missing image_00 in calib_cam_to_pose.txt", 4);
    }
    return Compose(Compose(T_W_I, camera->second), InvertRigid(calib.T_V_C0));
}

Eigen::Matrix4d ComposeTWCkRect(const Kitti360Calib &calib, const Eigen::Matrix4d &T_W_I, std::string_view cam)
{
    const std::string camera(cam);
    const auto pose = calib.T_I_Ck.find(camera);
    if (pose == calib.T_I_Ck.end())
    {
        throw IoError("calibration is missing " + camera + " in calib_cam_to_pose.txt", 4);
    }
    const auto rectification = calib.R_rect.find(camera);
    if (rectification == calib.R_rect.end())
    {
        throw IoError("calibration is missing R_rect for " + camera, 4);
    }
    return Compose(Compose(T_W_I, pose->second), InvertRigid(Rect4(rectification->second)));
}

Eigen::Matrix4d ComposeTCkRectV(const Kitti360Calib &calib, const Eigen::Matrix4d &T_W_I, std::string_view cam)
{
    const std::string camera(cam);
    const auto pose = calib.T_I_Ck.find(camera);
    if (pose == calib.T_I_Ck.end())
    {
        throw IoError("calibration is missing " + camera + " in calib_cam_to_pose.txt", 4);
    }
    const auto rectification = calib.R_rect.find(camera);
    if (rectification == calib.R_rect.end())
    {
        throw IoError("calibration is missing R_rect for " + camera, 4);
    }

    const Eigen::Matrix4d T_W_V = ComposeTWV(calib, T_W_I);
    const Eigen::Matrix4d T_W_Ck_rect = ComposeTWCkRect(calib, T_W_I, cam);
    const Eigen::Matrix4d from_chain = Compose(InvertRigid(T_W_Ck_rect), T_W_V);

    const auto camera00 = calib.T_I_Ck.find("image_00");
    if (camera00 == calib.T_I_Ck.end())
    {
        throw IoError("calibration is missing image_00 in calib_cam_to_pose.txt", 4);
    }
    const Eigen::Matrix4d T_C0_Ck = Compose(InvertRigid(camera00->second), pose->second);
    const Eigen::Matrix4d T_V_Ck = Compose(calib.T_V_C0, T_C0_Ck);
    const Eigen::Matrix4d from_raw = Compose(Rect4(rectification->second), InvertRigid(T_V_Ck));

    const double difference = (from_chain - from_raw).cwiseAbs().maxCoeff();
    if (!(difference <= 1e-9))
    {
        throw std::runtime_error("ComposeTCkRectV(" + camera + "): rectified and raw composition paths disagree by " +
                                 std::to_string(difference) + " > 1e-09");
    }
    return from_chain;
}

std::vector<int> FindCommonFrames(const std::filesystem::path &root, std::string_view sequence,
                                  const std::map<int, Eigen::Matrix4d> &poses,
                                  const std::map<int, Eigen::Matrix4d> &cam0ToWorld,
                                  const std::vector<std::string> &cameras, std::size_t requested)
{
    std::vector<int> common;
    common.reserve(poses.size());
    for (const auto &[frame, transform] : poses)
    {
        (void)transform;
        if (cam0ToWorld.find(frame) == cam0ToWorld.end())
        {
            continue;
        }
        bool complete = true;
        for (const std::string &camera : cameras)
        {
            if (!IsRegularFile(ImagePath(root, sequence, camera, frame)))
            {
                complete = false;
                break;
            }
        }
        if (!complete)
        {
            continue;
        }
        if (!IsRegularFile(LidarPath(root, sequence, frame)))
        {
            continue;
        }
        common.push_back(frame);
    }
    std::sort(common.begin(), common.end());
    common.erase(std::unique(common.begin(), common.end()), common.end());

    if (requested == 0U || common.empty())
    {
        return {};
    }

    // Find runs of consecutive frame IDs.
    std::size_t firstStart = std::numeric_limits<std::size_t>::max();
    std::size_t firstLength = 0;
    std::size_t bestStart = 0;
    std::size_t bestLength = 0;
    std::size_t index = 0;
    while (index < common.size())
    {
        std::size_t end = index;
        while (end + 1U < common.size() && common[end + 1U] == common[end] + 1)
        {
            ++end;
        }
        const std::size_t length = end - index + 1U;
        if (firstStart == std::numeric_limits<std::size_t>::max() && length >= requested)
        {
            firstStart = index;
            firstLength = length;
        }
        if (length > bestLength)
        {
            bestLength = length;
            bestStart = index;
        }
        index = end + 1U;
    }

    std::size_t start = 0;
    std::size_t length = 0;
    if (firstStart != std::numeric_limits<std::size_t>::max())
    {
        start = firstStart;
        length = firstLength;
    }
    else
    {
        start = bestStart;
        length = bestLength;
    }
    const std::size_t count = std::min(requested, length);
    std::vector<int> window;
    window.reserve(count);
    for (std::size_t offset = 0; offset < count; ++offset)
    {
        window.push_back(common[start + offset]);
    }
    return window;
}
} // namespace raylidar::calib
