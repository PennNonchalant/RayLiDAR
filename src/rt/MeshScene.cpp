#include "MeshScene.h"

#include "SceneBuffers.h"
#include "vulkan/Synchronization.h"
#include "vulkan/VulkanContext.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace raylidar::rt
{
namespace
{
void ValidateDescription(const MeshSceneDescription &description)
{
    if (description.Meshes.empty() || description.Instances.empty())
    {
        throw std::invalid_argument("Mesh scene requires meshes and instances.");
    }

    for (const auto &mesh : description.Meshes)
    {
        if (mesh.Positions.empty() || mesh.Indices.empty())
        {
            throw std::invalid_argument("Mesh vertex and index counts must be nonzero.");
        }

        if (std::ranges::any_of(mesh.Indices, [&mesh](std::uint32_t index) { return index >= mesh.Positions.size(); }))
        {
            throw std::out_of_range("Mesh index references a vertex outside the mesh.");
        }
    }

    if (std::ranges::any_of(description.Instances, [&description](const MeshInstanceDescription &instance) {
            return instance.MeshIndex >= description.Meshes.size();
        }))
    {
        throw std::out_of_range("Instance references a mesh outside the scene.");
    }
}
} // namespace

MeshScene::MeshScene(const vulkan::VulkanContext &context, const MeshSceneDescription &description)
{
    ValidateDescription(description);

    m_instances.assign(description.Instances.begin(), description.Instances.end());

    m_meshes.reserve(description.Meshes.size());
    m_uploadBuffers.reserve(description.Meshes.size() * 2);
    m_blas.reserve(description.Meshes.size());

    for (const auto &mesh : description.Meshes)
    {
        const auto vertexBytes = std::as_bytes(mesh.Positions);
        const auto indexBytes = std::as_bytes(mesh.Indices);

        m_meshes.push_back(MeshResources{
            MakeTriangleBuildInputBuffer(context, vertexBytes.size()),
            MakeTriangleBuildInputBuffer(context, indexBytes.size()),
            static_cast<std::uint32_t>(mesh.Positions.size()),
            static_cast<std::uint32_t>(mesh.Indices.size()),
        });

        m_uploadBuffers.push_back(MakeStagingBuffer(context, vertexBytes));
        m_uploadBuffers.push_back(MakeStagingBuffer(context, indexBytes));
    }
}

void MeshScene::RecordBuild(const vk::raii::CommandBuffer &commandBuffer, ASBuilder &builder)
{
    if (m_buildStarted)
    {
        throw std::logic_error("Mesh scene build can only be recorded once.");
    }

    m_buildStarted = true;

    std::vector<vk::AccelerationStructureInstanceKHR> instances(m_instances.size());

    for (std::size_t meshIndex = 0; meshIndex < m_meshes.size(); ++meshIndex)
    {
        const auto &mesh = m_meshes[meshIndex];
        const auto &vertexUpload = m_uploadBuffers[meshIndex * 2];
        const auto &indexUpload = m_uploadBuffers[meshIndex * 2 + 1];

        commandBuffer.copyBuffer(vertexUpload.Handle(), mesh.VertexBuffer.Handle(),
                                 vk::BufferCopy(0, 0, mesh.VertexBuffer.Size()));

        commandBuffer.copyBuffer(indexUpload.Handle(), mesh.IndexBuffer.Handle(),
                                 vk::BufferCopy(0, 0, mesh.IndexBuffer.Size()));
    }

    vulkan::RecordMemoryBarrier(commandBuffer, vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite,
                                vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
                                vk::AccessFlagBits2::eShaderRead);

    for (std::size_t meshIndex = 0; meshIndex < m_meshes.size(); ++meshIndex)
    {
        m_blas.push_back(builder.RecordBlas(commandBuffer, Geometry(meshIndex)));
    }

    vulkan::RecordMemoryBarrier(commandBuffer, vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
                                vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
                                vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
                                vk::AccessFlagBits2::eAccelerationStructureReadKHR);

    for (std::size_t instanceIndex = 0; instanceIndex < m_instances.size(); ++instanceIndex)
    {
        const auto &description = m_instances[instanceIndex];

        instances[instanceIndex]
            .setTransform(description.Transform)
            .setInstanceCustomIndex(static_cast<std::uint32_t>(instanceIndex))
            .setMask(description.Mask)
            .setInstanceShaderBindingTableRecordOffset(0)
            .setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable)
            .setAccelerationStructureReference(m_blas[description.MeshIndex].DeviceAddress());
    }

    m_tlas.emplace(builder.RecordTlas(commandBuffer, instances));

    vulkan::RecordMemoryBarrier(commandBuffer, vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
                                vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
                                vk::PipelineStageFlagBits2::eComputeShader,
                                vk::AccessFlagBits2::eAccelerationStructureReadKHR);
}

void MeshScene::ReleaseUploadResources()
{
    m_uploadBuffers.clear();
}

TriangleGeometryView MeshScene::Geometry(std::size_t meshIndex) const
{
    const auto &mesh = m_meshes.at(meshIndex);

    TriangleGeometryView geometry{mesh.VertexBuffer, mesh.IndexBuffer};
    geometry.VertexCount = mesh.VertexCount;
    geometry.IndexCount = mesh.IndexCount;
    geometry.VertexStride = sizeof(std::array<float, 3>);

    return geometry;
}
} // namespace raylidar::rt
