#pragma once

#include "ASBuilder.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace raylidar::rt
{
struct Gaussian
{
    std::array<float, 3> Center{};
    float Opacity = 1.0f;
    std::array<float, 3> AxisU{1.0f, 0.0f, 0.0f};
    float SigmaU = 1.0f;
    std::array<float, 3> AxisV{0.0f, 1.0f, 0.0f};
    float SigmaV = 1.0f;
};

struct GaussianSceneDescription
{
    std::span<const Gaussian> Gaussians{};
    float SigmaCutoff = 3.0f;
};

struct GaussianProxyBatch
{
    std::size_t StartIndex = 0;
    std::uint32_t Count = 0;
};

struct GaussianProxyGeometry
{
    std::vector<std::array<float, 3>> Vertices;
    std::vector<std::uint32_t> Indices;
    std::vector<GaussianProxyBatch> Batches;

    static GaussianProxyGeometry Create(const GaussianSceneDescription &description, std::uint32_t gaussiansPerBatch);
};

class GaussianScene
{
  public:
    GaussianScene(const vulkan::VulkanContext &context, const GaussianSceneDescription &description);
    ~GaussianScene() = default;

    GaussianScene(const GaussianScene &) = delete;
    GaussianScene &operator=(const GaussianScene &) = delete;
    GaussianScene(GaussianScene &&) noexcept = default;
    GaussianScene &operator=(GaussianScene &&) = delete;

    void RecordBuild(const vk::raii::CommandBuffer &commandBuffer, ASBuilder &builder);
    void ReleaseUploadResources();

    [[nodiscard]]
    const AccelerationStructure &Tlas() const noexcept
    {
        return *m_tlas;
    }

    [[nodiscard]]
    const vulkan::Buffer &GaussianBuffer() const noexcept
    {
        return m_gaussianBuffer;
    }

    [[nodiscard]]
    float SigmaCutoff() const noexcept
    {
        return m_sigmaCutoff;
    }

    [[nodiscard]]
    std::uint32_t GaussiansPerBatch() const noexcept
    {
        return m_gaussiansPerBatch;
    }

  private:
    struct BuildResources
    {
        vulkan::Buffer VertexBuffer;
        vulkan::Buffer IndexBuffer;
        vulkan::Buffer GaussianUpload;
        vulkan::Buffer VertexUpload;
        vulkan::Buffer IndexUpload;
        std::vector<GaussianProxyBatch> Batches;
    };

    float m_sigmaCutoff = 0.0f;
    std::uint32_t m_gaussiansPerBatch = 0;
    vulkan::Buffer m_gaussianBuffer;
    std::optional<BuildResources> m_buildResources;
    std::vector<AccelerationStructure> m_blas;
    std::optional<AccelerationStructure> m_tlas;
};
} // namespace raylidar::rt