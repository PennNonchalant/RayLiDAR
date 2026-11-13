#include "PlyIO.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace raylidar::simulation
{
namespace
{
struct PlyHeader
{
    struct Property
    {
        std::string Name;
        std::string Type;
        bool IsList = false;
        std::string CountType;
    };

    struct Element
    {
        std::string Name;
        std::size_t Count = 0;
        std::vector<Property> Properties;
    };

    bool HasSignature = false;
    std::string Version;
    std::vector<Element> Elements;
};

static std::size_t TypeSize(const std::string &type)
{
    if (type == "char" || type == "uchar" || type == "int8" || type == "uint8")
    {
        return 1;
    }
    if (type == "short" || type == "ushort" || type == "int16" || type == "uint16")
    {
        return 2;
    }
    if (type == "int" || type == "uint" || type == "int32" || type == "uint32" || type == "float" || type == "float32")
    {
        return 4;
    }
    if (type == "double" || type == "float64")
    {
        return 8;
    }
    throw std::runtime_error("Unsupported PLY property type: " + type);
}

template <typename T> static T Load(const char *data)
{
    T value;
    std::memcpy(&value, data, sizeof(T));
    return value;
}

static double LoadAsDouble(const std::string &type, const char *data)
{
    if (type == "char" || type == "int8")
    {
        return Load<std::int8_t>(data);
    }
    if (type == "uchar" || type == "uint8")
    {
        return Load<std::uint8_t>(data);
    }
    if (type == "short" || type == "int16")
    {
        return Load<std::int16_t>(data);
    }
    if (type == "ushort" || type == "uint16")
    {
        return Load<std::uint16_t>(data);
    }
    if (type == "int" || type == "int32")
    {
        return Load<std::int32_t>(data);
    }
    if (type == "uint" || type == "uint32")
    {
        return Load<std::uint32_t>(data);
    }
    if (type == "float" || type == "float32")
    {
        return Load<float>(data);
    }
    return Load<double>(data);
}

class Cursor
{
  public:
    explicit Cursor(const std::vector<char> &bytes) : m_bytes(bytes)
    {
    }

    const char *Take(std::size_t size)
    {
        if (m_offset + size > m_bytes.size())
        {
            throw std::runtime_error("PLY file is truncated.");
        }
        const char *data = m_bytes.data() + m_offset;
        m_offset += size;
        return data;
    }

  private:
    const std::vector<char> &m_bytes;
    std::size_t m_offset = 0;
};

PlyHeader ReadPlyHeader(std::istream &file, const std::filesystem::path &path)
{
    PlyHeader header;
    std::string line;
    std::getline(file, line);
    header.HasSignature = line == "ply" || line == "ply\r";
    bool isBinaryLittleEndian = false;

    do
    {
        std::istringstream tokens(line);
        std::string keyword;
        tokens >> keyword;
        if (keyword == "end_header")
        {
            if (!isBinaryLittleEndian)
            {
                throw std::runtime_error("Only binary_little_endian PLY is supported: " + path.string());
            }
            return header;
        }

        if (keyword == "format")
        {
            std::string encoding;
            tokens >> encoding >> header.Version;
            isBinaryLittleEndian = encoding == "binary_little_endian";
        }
        else if (keyword == "element")
        {
            PlyHeader::Element element;
            tokens >> element.Name >> element.Count;
            header.Elements.push_back(std::move(element));
        }
        else if (keyword == "property")
        {
            if (header.Elements.empty())
            {
                throw std::runtime_error("PLY property appears before any element.");
            }

            PlyHeader::Property property;
            tokens >> property.Type;
            if (property.Type == "list")
            {
                property.IsList = true;
                tokens >> property.CountType >> property.Type;
            }
            tokens >> property.Name;
            header.Elements.back().Properties.push_back(std::move(property));
        }
    } while (std::getline(file, line));

    throw std::runtime_error("PLY header is missing end_header: " + path.string());
}
} // namespace

MeshData PlyIO::LoadMesh(const std::filesystem::path &path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Failed to open PLY file: " + path.string());
    }

    const auto header = ReadPlyHeader(file, path);
    const std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    MeshData mesh;
    Cursor cursor(bytes);

    for (const auto &element : header.Elements)
    {
        const bool isVertex = element.Name == "vertex";
        const bool isFace = element.Name == "face";

        if (isVertex)
        {
            mesh.Positions.reserve(element.Count);
        }

        for (std::size_t item = 0; item < element.Count; ++item)
        {
            std::array<double, 3> position{};

            for (const auto &property : element.Properties)
            {
                if (property.IsList)
                {
                    const auto count = static_cast<std::size_t>(
                        LoadAsDouble(property.CountType, cursor.Take(TypeSize(property.CountType))));
                    const std::size_t itemSize = TypeSize(property.Type);

                    std::vector<std::uint32_t> polygon(count);

                    for (auto &index : polygon)
                    {
                        index = static_cast<std::uint32_t>(LoadAsDouble(property.Type, cursor.Take(itemSize)));
                    }

                    if (isFace && property.Name == "vertex_indices" && count >= 3)
                    {
                        for (std::size_t corner = 1; corner + 1 < count; ++corner)
                        {
                            mesh.Indices.insert(mesh.Indices.end(), {polygon[0], polygon[corner], polygon[corner + 1]});
                        }
                    }

                    continue;
                }

                const double value = LoadAsDouble(property.Type, cursor.Take(TypeSize(property.Type)));

                if (isVertex)
                {
                    if (property.Name == "x")
                    {
                        position[0] = value;
                    }
                    else if (property.Name == "y")
                    {
                        position[1] = value;
                    }
                    else if (property.Name == "z")
                    {
                        position[2] = value;
                    }
                }
            }

            if (isVertex)
            {
                mesh.Positions.push_back(position);
            }
        }
    }

    for (const auto index : mesh.Indices)
    {
        if (index >= mesh.Positions.size())
        {
            throw std::runtime_error("PLY face index is out of range: " + path.string());
        }
    }

    if (mesh.Positions.empty() || mesh.Indices.empty())
    {
        throw std::runtime_error("PLY file has no triangles: " + path.string());
    }

    return mesh;
}

std::vector<rt::Gaussian> PlyIO::LoadGaussians(const std::filesystem::path &path, const std::array<double, 3> &origin)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Failed to open Gaussian PLY: " + path.string());
    }

    const auto header = ReadPlyHeader(file, path);
    if (!header.HasSignature || header.Version != "1.0" || header.Elements.size() != 1 ||
        header.Elements.front().Name != "vertex")
    {
        throw std::runtime_error("Expected a 2D Gaussian PLY with one vertex element: " + path.string());
    }

    const auto &vertex = header.Elements.front();
    for (const auto &property : vertex.Properties)
    {
        if (property.IsList || property.Type != "float" || property.Name == "scale_2")
        {
            throw std::runtime_error("Unsupported 2D Gaussian PLY property " + property.Name + ": " + path.string());
        }
    }

    const auto propertyIndex = [&vertex, &path](std::string_view name) {
        const auto found = std::find_if(vertex.Properties.begin(), vertex.Properties.end(),
                                        [name](const auto &property) { return property.Name == name; });
        if (found == vertex.Properties.end())
        {
            throw std::runtime_error("Missing Gaussian PLY property " + std::string(name) + ": " + path.string());
        }
        return static_cast<std::size_t>(found - vertex.Properties.begin());
    };

    const std::array positionIndices{propertyIndex("x"), propertyIndex("y"), propertyIndex("z")};
    const auto opacityIndex = propertyIndex("opacity");
    const auto scaleUIndex = propertyIndex("scale_0");
    const auto scaleVIndex = propertyIndex("scale_1");
    const auto [wIndex, xIndex, yIndex, zIndex] =
        std::array{propertyIndex("rot_0"), propertyIndex("rot_1"), propertyIndex("rot_2"), propertyIndex("rot_3")};

    std::vector<float> record(vertex.Properties.size());
    const auto recordSize = static_cast<std::streamsize>(record.size() * sizeof(float));

    std::vector<rt::Gaussian> gaussians;
    gaussians.reserve(vertex.Count);
    for (std::size_t vertexIndex = 0; vertexIndex < vertex.Count; ++vertexIndex)
    {
        if (!file.read(reinterpret_cast<char *>(record.data()), recordSize))
        {
            throw std::runtime_error("Failed to read Gaussian PLY vertex data: " + path.string());
        }

        rt::Gaussian gaussian;
        for (std::size_t axis = 0; axis < gaussian.Center.size(); ++axis)
        {
            const double worldPosition = record[positionIndices[axis]];
            gaussian.Center[axis] = static_cast<float>(worldPosition - origin[axis]);
        }

        gaussian.Opacity = 1.0f / (1.0f + std::exp(-record[opacityIndex]));
        gaussian.SigmaU = std::exp(record[scaleUIndex]);
        gaussian.SigmaV = std::exp(record[scaleVIndex]);

        const double rawW = record[wIndex];
        const double rawX = record[xIndex];
        const double rawY = record[yIndex];
        const double rawZ = record[zIndex];
        const double inverseNorm = 1.0 / std::sqrt(rawW * rawW + rawX * rawX + rawY * rawY + rawZ * rawZ);
        const auto w = static_cast<float>(rawW * inverseNorm);
        const auto x = static_cast<float>(rawX * inverseNorm);
        const auto y = static_cast<float>(rawY * inverseNorm);
        const auto z = static_cast<float>(rawZ * inverseNorm);

        gaussian.AxisU = {1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y + w * z), 2.0f * (x * z - w * y)};
        gaussian.AxisV = {2.0f * (x * y - w * z), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z + w * x)};
        gaussians.push_back(gaussian);
    }

    return gaussians;
}

void PlyIO::WritePointCloud(const std::filesystem::path &path, std::span<const std::array<float, 3>> points)
{
    static_assert(std::endian::native == std::endian::little, "PLY writer assumes a little-endian host.");

    std::ofstream file(path, std::ios::binary);

    if (!file)
    {
        throw std::runtime_error("Failed to open PLY file for writing: " + path.string());
    }

    file << "ply\n"
         << "format binary_little_endian 1.0\n"
         << "element vertex " << points.size() << '\n'
         << "property float x\n"
         << "property float y\n"
         << "property float z\n"
         << "end_header\n";

    const auto bytes = std::as_bytes(points);
    file.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));

    if (!file)
    {
        throw std::runtime_error("Failed to write PLY file: " + path.string());
    }
}
} // namespace raylidar::simulation