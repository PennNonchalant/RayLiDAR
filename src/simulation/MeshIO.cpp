#include "MeshIO.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace raylidar::simulation
{
namespace
{
struct PlyProperty
{
    std::string Name;
    std::string Type;
    bool IsList = false;
    std::string CountType;
};

struct PlyElement
{
    std::string Name;
    std::size_t Count = 0;
    std::vector<PlyProperty> Properties;
};

std::size_t TypeSize(const std::string &type)
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

template <typename T> T Load(const char *data)
{
    T value;
    std::memcpy(&value, data, sizeof(T));
    return value;
}

double LoadAsDouble(const std::string &type, const char *data)
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
    Cursor(const std::vector<char> &bytes, std::size_t offset) : m_bytes(bytes), m_offset(offset)
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
} // namespace

MeshData LoadPlyMesh(const std::filesystem::path &path)
{
    std::ifstream file(path, std::ios::binary);

    if (!file)
    {
        throw std::runtime_error("Failed to open PLY file: " + path.string());
    }

    const std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    const std::string headerEnd = "end_header\n";
    const auto headerEndPosition = std::string_view(bytes.data(), bytes.size()).find(headerEnd);

    if (headerEndPosition == std::string_view::npos)
    {
        throw std::runtime_error("PLY header is missing end_header: " + path.string());
    }

    std::istringstream header(std::string(bytes.data(), headerEndPosition));
    std::vector<PlyElement> elements;
    std::string line;
    bool isBinaryLittleEndian = false;

    while (std::getline(header, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }

        std::istringstream tokens(line);
        std::string keyword;
        tokens >> keyword;

        if (keyword == "format")
        {
            std::string format;
            tokens >> format;
            isBinaryLittleEndian = format == "binary_little_endian";
        }
        else if (keyword == "element")
        {
            PlyElement element;
            tokens >> element.Name >> element.Count;
            elements.push_back(element);
        }
        else if (keyword == "property")
        {
            if (elements.empty())
            {
                throw std::runtime_error("PLY property appears before any element.");
            }

            PlyProperty property;
            tokens >> property.Type;

            if (property.Type == "list")
            {
                property.IsList = true;
                tokens >> property.CountType >> property.Type;
            }

            tokens >> property.Name;
            elements.back().Properties.push_back(property);
        }
    }

    if (!isBinaryLittleEndian)
    {
        throw std::runtime_error("Only binary_little_endian PLY is supported: " + path.string());
    }

    MeshData mesh;
    Cursor cursor(bytes, headerEndPosition + headerEnd.size());

    for (const auto &element : elements)
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

std::array<double, 3> BoundingBoxCenter(std::span<const std::array<double, 3>> positions)
{
    constexpr double largest = std::numeric_limits<double>::max();
    constexpr double lowest = std::numeric_limits<double>::lowest();

    std::array<double, 3> minimum{largest, largest, largest};
    std::array<double, 3> maximum{lowest, lowest, lowest};

    for (const auto &position : positions)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            minimum[axis] = std::min(minimum[axis], position[axis]);
            maximum[axis] = std::max(maximum[axis], position[axis]);
        }
    }

    return {
        0.5 * (minimum[0] + maximum[0]),
        0.5 * (minimum[1] + maximum[1]),
        0.5 * (minimum[2] + maximum[2]),
    };
}

std::vector<std::array<float, 3>> ToLocalFloat(std::span<const std::array<double, 3>> positions,
                                               const std::array<double, 3> &origin)
{
    std::vector<std::array<float, 3>> local;
    local.reserve(positions.size());

    for (const auto &position : positions)
    {
        local.push_back({
            static_cast<float>(position[0] - origin[0]),
            static_cast<float>(position[1] - origin[1]),
            static_cast<float>(position[2] - origin[2]),
        });
    }

    return local;
}
} // namespace raylidar::simulation
