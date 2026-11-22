#include "GaussianScene.h"

#include "SceneBuffers.h"
#include "vulkan/Synchronization.h"
#include "vulkan/VulkanContext.h"

#include <algorithm>
#include <utility>

namespace raylidar::rt
{
namespace
{
using Position = std::array<float, 3>;

constexpr std::uint32_t kVerticesPerGaussian = 4;
constexpr std::uint32_t kTrianglesPerGaussian = 2;
constexpr std::uint32_t kIndicesPerGaussian = 3 * kTrianglesPerGaussian;
constexpr std::uint32_t kMortonAxisBits = 10;

std::uint32_t MortonCode(const Position &center, const Position &minimum, const Position &scale)
{
    std::uint32_t code = 0;
    for (std::uint32_t axis = 0; axis < center.size(); ++axis)
    {
        const auto coordinate = static_cast<std::uint32_t>((center[axis] - minimum[axis]) * scale[axis]);
        for (std::uint32_t bit = 0; bit < kMortonAxisBits; ++bit)
        {
            code |= ((coordinate >> bit) & 1u) << (3 * bit + axis);
        }
    }
    return code;
}

void SortByMortonCode(std::span<const Gaussian> gaussians, std::span<std::uint32_t> indices)
{
    Position minimum = gaussians.front().Center;
    Position maximum = minimum;
    for (const auto &gaussian : gaussians)
    {
        for (std::size_t axis = 0; axis < minimum.size(); ++axis)
        {
            minimum[axis] = std::min(minimum[axis], gaussian.Center[axis]);
            maximum[axis] = std::max(maximum[axis], gaussian.Center[axis]);
        }
    }

    Position scale{};
    for (std::size_t axis = 0; axis < scale.size(); ++axis)
    {
        const float extent = maximum[axis] - minimum[axis];
        scale[axis] = extent > 0.0f ? static_cast<float>((1u << kMortonAxisBits) - 1u) / extent : 0.0f;
    }

    std::vector<std::pair<std::uint32_t, std::uint32_t>> keys;
    keys.reserve(gaussians.size());
    for (std::uint32_t index = 0; index < gaussians.size(); ++index)
    {
        keys.emplace_back(MortonCode(gaussians[index].Center, minimum, scale), index);
    }
    std::sort(keys.begin(), keys.end());
    std::transform(keys.begin(), keys.end(), indices.begin(), [](const auto &key) { return key.second; });
}

void RecordCopy(const vk::raii::CommandBuffer &commandBuffer, const vulkan::Buffer &source,
                const vulkan::Buffer &destination)
{
    commandBuffer.copyBuffer(source.Handle(), destination.Handle(), vk::BufferCopy(0, 0, destination.Size()));
}
} // namespace

GaussianProxyGeometry GaussianProxyGeometry::Create(const GaussianSceneDescription &description,
                                                    std::uint32_t gaussiansPerBatch)
{
    GaussianProxyGeometry geometry;
    const auto gaussians = description.Gaussians;
    geometry.Vertices.reserve(gaussians.size() * kVerticesPerGaussian);
    geometry.Indices.reserve(gaussians.size() * kIndicesPerGaussian);
    geometry.Batches.reserve((gaussians.size() + gaussiansPerBatch - 1) / gaussiansPerBatch);
    const bool useMortonOrder = description.BlasGroupSizeLog2 != 0;
    if (useMortonOrder)
    {
        geometry.GaussianIndices.resize(gaussians.size());
        SortByMortonCode(gaussians, geometry.GaussianIndices);
    }

    for (std::size_t first = 0; first < gaussians.size(); first += gaussiansPerBatch)
    {
        const auto count =
            static_cast<std::uint32_t>(std::min<std::size_t>(gaussians.size() - first, gaussiansPerBatch));
        geometry.Batches.push_back({first, count});

        for (std::uint32_t index = 0; index < count; ++index)
        {
            const auto slot = first + index;
            const auto &gaussian = gaussians[useMortonOrder ? geometry.GaussianIndices[slot] : slot];
            const float extentU = description.SigmaCutoff * gaussian.SigmaU;
            const float extentV = description.SigmaCutoff * gaussian.SigmaV;
            std::array<Position, kVerticesPerGaussian> corners{};

            for (std::size_t axis = 0; axis < gaussian.Center.size(); ++axis)
            {
                const float center = gaussian.Center[axis];
                const float u = extentU * gaussian.AxisU[axis];
                const float v = extentV * gaussian.AxisV[axis];
                corners[0][axis] = center - u - v;
                corners[1][axis] = center + u - v;
                corners[2][axis] = center + u + v;
                corners[3][axis] = center - u + v;
            }

            const auto base = index * kVerticesPerGaussian;
            geometry.Vertices.insert(geometry.Vertices.end(), corners.begin(), corners.end());
            geometry.Indices.insert(geometry.Indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
        }
    }

    return geometry;
}

GaussianScene::GaussianScene(const vulkan::VulkanContext &context, const GaussianSceneDescription &description)
    : m_sigmaCutoff(description.SigmaCutoff),
      m_gaussiansPerBatch(description.BlasGroupSizeLog2 == 0 ? static_cast<std::uint32_t>(description.Gaussians.size())
                                                             : 1u << description.BlasGroupSizeLog2),
      m_gaussianBuffer(context, {description.Gaussians.size_bytes(),
                                 vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer})
{
    auto geometry = GaussianProxyGeometry::Create(description, m_gaussiansPerBatch);
    const auto vertices = std::as_bytes(std::span(geometry.Vertices));
    const auto indices = std::as_bytes(std::span(geometry.Indices));

    m_buildResources.emplace(BuildResources{
        MakeTriangleBuildInputBuffer(context, vertices.size()),
        MakeTriangleBuildInputBuffer(context, indices.size()),
        MakeStagingBuffer(context, std::as_bytes(description.Gaussians)),
        std::nullopt,
        MakeStagingBuffer(context, vertices),
        MakeStagingBuffer(context, indices),
        std::move(geometry.Batches),
    });

    if (!geometry.GaussianIndices.empty())
    {
        const auto gaussianIndices = std::as_bytes(std::span(geometry.GaussianIndices));
        m_gaussianIndexBuffer.emplace(
            context, vulkan::BufferDescription{gaussianIndices.size(), vk::BufferUsageFlagBits::eTransferDst |
                                                                           vk::BufferUsageFlagBits::eStorageBuffer});
        m_buildResources->GaussianIndexUpload.emplace(MakeStagingBuffer(context, gaussianIndices));
    }
}

void GaussianScene::RecordBuild(const vk::raii::CommandBuffer &commandBuffer, ASBuilder &builder)
{
    const auto &resources = *m_buildResources;
    RecordCopy(commandBuffer, resources.GaussianUpload, m_gaussianBuffer);
    if (resources.GaussianIndexUpload)
    {
        RecordCopy(commandBuffer, *resources.GaussianIndexUpload, *m_gaussianIndexBuffer);
    }
    RecordCopy(commandBuffer, resources.VertexUpload, resources.VertexBuffer);
    RecordCopy(commandBuffer, resources.IndexUpload, resources.IndexBuffer);

    vulkan::RecordMemoryBarrier(commandBuffer, vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite,
                                vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
                                vk::AccessFlagBits2::eShaderRead);

    const vk::TransformMatrixKHR transform{std::array<std::array<float, 4>, 3>{{
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
    }}};
    const auto flags =
        vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable | vk::GeometryInstanceFlagBitsKHR::eForceNoOpaque;

    std::vector<vk::AccelerationStructureInstanceKHR> instances(resources.Batches.size());
    m_blas.reserve(resources.Batches.size());

    for (std::size_t batchIndex = 0; batchIndex < resources.Batches.size(); ++batchIndex)
    {
        const auto &batch = resources.Batches[batchIndex];
        const auto first = static_cast<vk::DeviceSize>(batch.StartIndex);
        TriangleGeometryView geometry{resources.VertexBuffer, resources.IndexBuffer};
        geometry.VertexCount = batch.Count * kVerticesPerGaussian;
        geometry.IndexCount = batch.Count * kIndicesPerGaussian;
        geometry.VertexOffset = first * kVerticesPerGaussian * sizeof(Position);
        geometry.IndexOffset = first * kIndicesPerGaussian * sizeof(std::uint32_t);
        geometry.VertexStride = sizeof(Position);

        const auto &blas = m_blas.emplace_back(builder.RecordBlas(commandBuffer, geometry));
        instances[batchIndex]
            .setTransform(transform)
            .setInstanceCustomIndex(static_cast<std::uint32_t>(batchIndex))
            .setMask(0xFF)
            .setInstanceShaderBindingTableRecordOffset(0)
            .setFlags(flags)
            .setAccelerationStructureReference(blas.DeviceAddress());
    }

    vulkan::RecordMemoryBarrier(commandBuffer, vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
                                vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
                                vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
                                vk::AccessFlagBits2::eAccelerationStructureReadKHR);

    m_tlas.emplace(builder.RecordTlas(commandBuffer, instances));

    vulkan::RecordMemoryBarrier(commandBuffer, vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
                                vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
                                vk::PipelineStageFlagBits2::eComputeShader,
                                vk::AccessFlagBits2::eAccelerationStructureReadKHR);
    vulkan::RecordMemoryBarrier(commandBuffer, vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite,
                                vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead);
}

void GaussianScene::ReleaseUploadResources()
{
    m_buildResources.reset();
}
} // namespace raylidar::rt
