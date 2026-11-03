#include "ASBuilder.h"

#include "vulkan/VulkanContext.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace raylidar::rt
{
namespace
{
vk::DeviceAddress GetBuildInputAddress(const vulkan::Buffer &buffer, vk::DeviceSize offset, vk::DeviceSize byteCount)
{
    if (!(buffer.Usage() & vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR))
    {
        throw std::invalid_argument("Geometry buffer requires acceleration structure build input usage.");
    }

    if (offset > buffer.Size() || byteCount > buffer.Size() - offset)
    {
        throw std::out_of_range("Geometry range exceeds the buffer size.");
    }

    const vk::DeviceAddress address = buffer.DeviceAddress() + offset;

    if (address % 4 != 0)
    {
        throw std::invalid_argument("Geometry input address must be aligned to four bytes.");
    }

    return address;
}
} // namespace

ASBuilder::ASBuilder(const vulkan::VulkanContext &context) : m_context(context)
{
}

AccelerationStructure ASBuilder::RecordBlas(const vk::raii::CommandBuffer &commandBuffer,
                                            const TriangleGeometryView &geometry)
{
    if (geometry.VertexCount == 0 || geometry.IndexCount == 0 || geometry.IndexCount % 3 != 0)
    {
        throw std::invalid_argument(
            "Triangle geometry requires vertices and a nonzero index count divisible by three.");
    }

    constexpr vk::DeviceSize vertexSize = 3 * sizeof(float);

    if (geometry.VertexStride < vertexSize || geometry.VertexStride % sizeof(float) != 0)
    {
        throw std::invalid_argument("Vertex stride must hold float3, and be aligned to four bytes.");
    }

    const std::uint32_t primitiveCount = geometry.IndexCount / 3;

    if (primitiveCount > m_context.AccelerationStructureProperties().maxPrimitiveCount)
    {
        throw std::invalid_argument("Triangle count exceeds the device acceleration structure limit.");
    }

    const vk::DeviceSize vertexByteCount =
        (static_cast<vk::DeviceSize>(geometry.VertexCount) - 1) * geometry.VertexStride + vertexSize;

    const vk::DeviceSize indexByteCount = static_cast<vk::DeviceSize>(geometry.IndexCount) * sizeof(std::uint32_t);

    const vk::DeviceAddress vertexAddress =
        GetBuildInputAddress(geometry.VertexBuffer, geometry.VertexOffset, vertexByteCount);

    const vk::DeviceAddress indexAddress =
        GetBuildInputAddress(geometry.IndexBuffer, geometry.IndexOffset, indexByteCount);

    vk::AccelerationStructureGeometryTrianglesDataKHR triangles;
    triangles.setVertexFormat(vk::Format::eR32G32B32Sfloat)
        .setVertexData(vertexAddress)
        .setVertexStride(geometry.VertexStride)
        .setMaxVertex(geometry.VertexCount - 1)
        .setIndexType(vk::IndexType::eUint32)
        .setIndexData(indexAddress);

    vk::AccelerationStructureGeometryKHR buildGeometry;
    buildGeometry.setGeometryType(vk::GeometryTypeKHR::eTriangles).setFlags(vk::GeometryFlagBitsKHR::eOpaque);
    buildGeometry.geometry.setTriangles(triangles);

    return RecordBuild(commandBuffer, vk::AccelerationStructureTypeKHR::eBottomLevel, buildGeometry, primitiveCount);
}

AccelerationStructure ASBuilder::RecordTlas(const vk::raii::CommandBuffer &commandBuffer,
                                            std::span<const vk::AccelerationStructureInstanceKHR> instances)
{
    if (instances.empty() || instances.size() > m_context.AccelerationStructureProperties().maxInstanceCount)
    {
        throw std::invalid_argument("Instance count must be nonzero and within the device limits.");
    }

    vulkan::BufferDescription instanceBufferDescription;
    instanceBufferDescription.Size = static_cast<vk::DeviceSize>(instances.size_bytes());
    instanceBufferDescription.Usage = vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
                                      vk::BufferUsageFlagBits::eShaderDeviceAddress;
    instanceBufferDescription.MemoryUsage = vulkan::BufferMemoryUsage::Upload;
    instanceBufferDescription.MinimumAlignment = 16;

    vulkan::Buffer instanceBuffer(m_context, instanceBufferDescription);
    instanceBuffer.Write(std::as_bytes(instances));

    const vk::DeviceAddress instanceAddress = instanceBuffer.DeviceAddress();

    m_buildBuffers.push_back(std::move(instanceBuffer));

    vk::AccelerationStructureGeometryInstancesDataKHR instanceData;
    instanceData.setArrayOfPointers(false).setData(instanceAddress);

    vk::AccelerationStructureGeometryKHR buildGeometry;
    buildGeometry.setGeometryType(vk::GeometryTypeKHR::eInstances);
    buildGeometry.geometry.setInstances(instanceData);

    return RecordBuild(commandBuffer, vk::AccelerationStructureTypeKHR::eTopLevel, buildGeometry,
                       static_cast<std::uint32_t>(instances.size()));
}

AccelerationStructure ASBuilder::RecordBuild(const vk::raii::CommandBuffer &commandBuffer,
                                             vk::AccelerationStructureTypeKHR type,
                                             const vk::AccelerationStructureGeometryKHR &geometry,
                                             std::uint32_t primitiveCount)
{
    vk::AccelerationStructureBuildGeometryInfoKHR buildInfo;
    buildInfo.setType(type)
        .setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace)
        .setMode(vk::BuildAccelerationStructureModeKHR::eBuild)
        .setGeometries(geometry);

    const auto sizes = m_context.Device().getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo, primitiveCount);

    AccelerationStructureDescription description;
    description.Type = type;
    description.Size = sizes.accelerationStructureSize;

    AccelerationStructure accelerationStructure(m_context, description);

    vulkan::BufferDescription scratchDescription;

    scratchDescription.Size = std::max(sizes.buildScratchSize, vk::DeviceSize{1});
    scratchDescription.Usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;
    scratchDescription.MinimumAlignment =
        m_context.AccelerationStructureProperties().minAccelerationStructureScratchOffsetAlignment;

    const auto &scratchBuffer = m_buildBuffers.emplace_back(m_context, scratchDescription);

    const vk::DeviceAddress scratchAddress = scratchBuffer.DeviceAddress();

    buildInfo.setDstAccelerationStructure(accelerationStructure.Handle())
        .setScratchData(scratchAddress);

    vk::AccelerationStructureBuildRangeInfoKHR rangeInfo;
    rangeInfo.setPrimitiveCount(primitiveCount);

    commandBuffer.buildAccelerationStructuresKHR(buildInfo, &rangeInfo);

    return accelerationStructure;
}
} // namespace raylidar::rt