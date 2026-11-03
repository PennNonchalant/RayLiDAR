#pragma once

#include "ASBuilder.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace raylidar::rt
{
struct MeshDescription
{
    std::span<const std::array<float, 3>> Positions{};
    std::span<const std::uint32_t> Indices{};
};

struct MeshInstanceDescription
{
    std::uint32_t MeshIndex = 0;

    vk::TransformMatrixKHR Transform{std::array<std::array<float, 4>, 3>{{
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
    }}};

    std::uint8_t Mask = 0xFF;
};

struct MeshSceneDescription
{
    std::span<const MeshDescription> Meshes{};
    std::span<const MeshInstanceDescription> Instances{};
};

class MeshScene
{
  public:
    MeshScene(const vulkan::VulkanContext &context, const MeshSceneDescription &description);
    ~MeshScene() = default;

    MeshScene(const MeshScene &) = delete;
    MeshScene &operator=(const MeshScene &) = delete;

    MeshScene(MeshScene &&) noexcept = default;
    MeshScene &operator=(MeshScene &&) = delete;

    void RecordBuild(const vk::raii::CommandBuffer &commandBuffer, ASBuilder &builder);

    void ReleaseUploadResources();

    [[nodiscard]]
    const AccelerationStructure &Tlas() const
    {
        return m_tlas.value();
    }

    [[nodiscard]]
    std::size_t MeshCount() const noexcept
    {
        return m_meshes.size();
    }

    [[nodiscard]]
    TriangleGeometryView Geometry(std::size_t meshIndex) const;

    [[nodiscard]]
    std::span<const MeshInstanceDescription> Instances() const noexcept
    {
        return m_instances;
    }

  private:
    struct MeshResources
    {
        vulkan::Buffer VertexBuffer;
        vulkan::Buffer IndexBuffer;

        std::uint32_t VertexCount = 0;
        std::uint32_t IndexCount = 0;
    };

    std::vector<MeshResources> m_meshes;
    std::vector<MeshInstanceDescription> m_instances;

    std::vector<vulkan::Buffer> m_uploadBuffers;

    std::vector<AccelerationStructure> m_blas;
    std::optional<AccelerationStructure> m_tlas;

    bool m_buildStarted = false;
};
} // namespace raylidar::rt
