#include "rt/ASBuilder.h"
#include "rt/AccelerationStructure.h"

#include "vulkan/Buffer.h"
#include "vulkan/CommandContext.h"
#include "vulkan/ComputePipeline.h"
#include "vulkan/DescriptorBindings.h"
#include "vulkan/Shader.h"
#include "vulkan/VulkanContext.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

int main(int argc, char *argv[])
{
    try
    {
        if (argc != 2)
        {
            throw std::invalid_argument("Usage: raylidar_smoke <shader.spv>");
        }

        {
            raylidar::vulkan::ContextDescription description;
            description.ApplicationName = "RayLiDAR Vulkan Smoke";
            description.EnableValidationLayers = true;

            const raylidar::vulkan::VulkanContext context(description);

            const auto &properties = context.PhysicalDeviceProperties();

            std::cout << "Device: " << properties.deviceName.data() << '\n'
                      << "Vulkan: " << VK_API_VERSION_MAJOR(properties.apiVersion) << '.'
                      << VK_API_VERSION_MINOR(properties.apiVersion) << '.'
                      << VK_API_VERSION_PATCH(properties.apiVersion) << '\n'
                      << "Compute queue family: " << context.ComputeQueueFamilyIndex() << '\n'
                      << "AS scratch alignment: "
                      << context.AccelerationStructureProperties().minAccelerationStructureScratchOffsetAlignment
                      << '\n';

            {
                raylidar::rt::AccelerationStructureDescription blasDescription;
                blasDescription.Type = vk::AccelerationStructureTypeKHR::eBottomLevel;
                blasDescription.Size = 4095;

                raylidar::rt::AccelerationStructure initialBlas(context, blasDescription);

                const vk::AccelerationStructureKHR originalHandle = initialBlas.Handle();
                const vk::DeviceAddress originalAddress = initialBlas.DeviceAddress();

                if (!originalHandle || originalAddress == 0 || initialBlas.Type() != blasDescription.Type ||
                    initialBlas.Size() != blasDescription.Size)
                {
                    throw std::runtime_error("Acceleration structure initialization failed.");
                }

                raylidar::rt::AccelerationStructure movedBlas(std::move(initialBlas));

                if (movedBlas.Handle() != originalHandle || movedBlas.DeviceAddress() != originalAddress ||
                    movedBlas.Type() != blasDescription.Type || movedBlas.Size() != blasDescription.Size)
                {
                    throw std::runtime_error("Acceleration structure move construction failed.");
                }

                raylidar::rt::AccelerationStructureDescription targetDescription;
                targetDescription.Type = vk::AccelerationStructureTypeKHR::eTopLevel;
                targetDescription.Size = 8192;

                raylidar::rt::AccelerationStructure target(context, targetDescription);
                target = std::move(movedBlas);

                if (target.Handle() != originalHandle || target.DeviceAddress() != originalAddress ||
                    target.Type() != blasDescription.Type || target.Size() != blasDescription.Size)
                {
                    throw std::runtime_error("Acceleration structure move assignment failed.");
                }
            }

            std::cout << "Acceleration structure resource smoke passed.\n";

            {
                const std::array<std::array<float, 3>, 3> vertices = {{
                    {-1.0f, -1.0f, 0.0f},
                    {1.0f, -1.0f, 0.0f},
                    {0.0f, 1.0f, 0.0f},
                }};

                const std::array<std::uint32_t, 3> indices = {0, 1, 2};

                const auto vertexBytes = std::as_bytes(std::span(vertices));
                const auto indexBytes = std::as_bytes(std::span(indices));

                raylidar::vulkan::BufferDescription geometryBufferDescription;
                geometryBufferDescription.Size = static_cast<vk::DeviceSize>(vertexBytes.size());
                geometryBufferDescription.Usage = vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
                                                  vk::BufferUsageFlagBits::eShaderDeviceAddress;
                geometryBufferDescription.MemoryUsage = raylidar::vulkan::BufferMemoryUsage::Upload;
                geometryBufferDescription.MinimumAlignment = 4;

                raylidar::vulkan::Buffer vertexBuffer(context, geometryBufferDescription);
                vertexBuffer.Write(vertexBytes);

                geometryBufferDescription.Size = static_cast<vk::DeviceSize>(indexBytes.size());

                raylidar::vulkan::Buffer indexBuffer(context, geometryBufferDescription);
                indexBuffer.Write(indexBytes);

                raylidar::rt::TriangleGeometryView geometry{vertexBuffer, indexBuffer};
                geometry.VertexCount = static_cast<std::uint32_t>(vertices.size());
                geometry.IndexCount = static_cast<std::uint32_t>(indices.size());

                raylidar::rt::ASBuilder builder(context);

                raylidar::vulkan::CommandContext buildCommands(context);
                const auto &commandBuffer = buildCommands.Begin();

                raylidar::rt::AccelerationStructure blas(builder.RecordBlas(commandBuffer, geometry));

                vk::MemoryBarrier2 buildBarrier;
                buildBarrier.setSrcStageMask(vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR)
                    .setSrcAccessMask(vk::AccessFlagBits2::eAccelerationStructureWriteKHR)
                    .setDstStageMask(vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR)
                    .setDstAccessMask(vk::AccessFlagBits2::eAccelerationStructureReadKHR);

                vk::DependencyInfo buildDependency;
                buildDependency.setMemoryBarriers(buildBarrier);

                commandBuffer.pipelineBarrier2(buildDependency);

                vk::TransformMatrixKHR transform{};
                transform.matrix[0][0] = 1.0f;
                transform.matrix[1][1] = 1.0f;
                transform.matrix[2][2] = 1.0f;

                vk::AccelerationStructureInstanceKHR instance;
                instance.setTransform(transform)
                    .setInstanceCustomIndex(0)
                    .setMask(0xFF)
                    .setInstanceShaderBindingTableRecordOffset(0)
                    .setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable)
                    .setAccelerationStructureReference(blas.DeviceAddress());

                raylidar::rt::AccelerationStructure tlas(builder.RecordTlas(
                    commandBuffer, std::span<const vk::AccelerationStructureInstanceKHR>(&instance, 1)));

                buildCommands.End();
                buildCommands.SubmitAndWait();

                if (blas.Type() != vk::AccelerationStructureTypeKHR::eBottomLevel ||
                    tlas.Type() != vk::AccelerationStructureTypeKHR::eTopLevel || blas.Size() == 0 ||
                    tlas.Size() == 0 || blas.DeviceAddress() == 0 || tlas.DeviceAddress() == 0)
                {
                    throw std::runtime_error("Acceleration structure build smoke failed.");
                }

                std::cout << "BLAS storage: " << blas.Size() << " bytes\n"
                          << "TLAS storage: " << tlas.Size() << " bytes\n";
            }

            std::cout << "Acceleration structure build smoke passed.\n";

            const raylidar::vulkan::Shader shader(context, argv[1]);

            raylidar::vulkan::ComputePipelineDescription pipelineDescription;
            pipelineDescription.DescriptorSets.resize(2);

            for (auto &setDescription : pipelineDescription.DescriptorSets)
            {
                setDescription.Bindings.emplace_back(0, vk::DescriptorType::eStorageBuffer, 1,
                                                     vk::ShaderStageFlagBits::eCompute);
            }

            const raylidar::vulkan::ComputePipeline pipeline(context, shader, pipelineDescription);

            const std::array<std::array<std::uint32_t, 4>, 2> inputs = {{
                {1, 2, 3, 4},
                {10, 20, 30, 40},
            }};

            const std::array<std::array<std::uint32_t, 4>, 2> expectedResults = {{
                {3, 5, 7, 9},
                {21, 41, 61, 81},
            }};

            const vk::DeviceSize byteCount = static_cast<vk::DeviceSize>(std::as_bytes(std::span(inputs[0])).size());

            raylidar::vulkan::BufferDescription uploadDescription;
            uploadDescription.Size = byteCount;
            uploadDescription.Usage = vk::BufferUsageFlagBits::eTransferSrc;
            uploadDescription.MemoryUsage = raylidar::vulkan::BufferMemoryUsage::Upload;

            raylidar::vulkan::Buffer initialUploadBuffer(context, uploadDescription);
            const vk::Buffer originalHandle = initialUploadBuffer.Handle();

            raylidar::vulkan::Buffer movedUploadBuffer(std::move(initialUploadBuffer));

            raylidar::vulkan::Buffer uploadBuffer(context, uploadDescription);
            uploadBuffer = std::move(movedUploadBuffer);

            if (uploadBuffer.Handle() != originalHandle || uploadBuffer.Size() != byteCount)
            {
                throw std::runtime_error("Buffer move failed.");
            }

            raylidar::vulkan::BufferDescription inputDescription;
            inputDescription.Size = byteCount;
            inputDescription.Usage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer;

            const raylidar::vulkan::Buffer inputBuffer(context, inputDescription);

            raylidar::vulkan::BufferDescription outputDescription;
            outputDescription.Size = byteCount;
            outputDescription.Usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc;

            const raylidar::vulkan::Buffer outputBuffer(context, outputDescription);

            raylidar::vulkan::BufferDescription readbackDescription;
            readbackDescription.Size = byteCount;
            readbackDescription.Usage = vk::BufferUsageFlagBits::eTransferDst;
            readbackDescription.MemoryUsage = raylidar::vulkan::BufferMemoryUsage::Readback;

            const raylidar::vulkan::Buffer readbackBuffer(context, readbackDescription);

            raylidar::vulkan::DescriptorBindings bindings(context, pipeline);

            if (bindings.Handles().size() != 2 || !bindings.Handle(0) || !bindings.Handle(1))
            {
                throw std::runtime_error("Descriptor set initialization failed.");
            }

            bindings.WriteBuffer(0, 0, inputBuffer);
            bindings.WriteBuffer(1, 0, outputBuffer);

            raylidar::vulkan::CommandContext commands(context);

            for (std::size_t round = 0; round < inputs.size(); ++round)
            {
                const auto &input = inputs[round];
                uploadBuffer.Write(std::as_bytes(std::span(input)));

                const auto &commandBuffer = commands.Begin();

                const auto barrier =
                    [&commandBuffer](vk::PipelineStageFlags2 sourceStage, vk::AccessFlags2 sourceAccess,
                                     vk::PipelineStageFlags2 destinationStage, vk::AccessFlags2 destinationAccess) {
                        vk::MemoryBarrier2 memoryBarrier;
                        memoryBarrier.setSrcStageMask(sourceStage)
                            .setSrcAccessMask(sourceAccess)
                            .setDstStageMask(destinationStage)
                            .setDstAccessMask(destinationAccess);

                        vk::DependencyInfo dependencyInformation;
                        dependencyInformation.setMemoryBarriers(memoryBarrier);

                        commandBuffer.pipelineBarrier2(dependencyInformation);
                    };

                const vk::BufferCopy copyRegion(0, 0, byteCount);

                commandBuffer.copyBuffer(uploadBuffer.Handle(), inputBuffer.Handle(), copyRegion);

                barrier(vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite,
                        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead);

                commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.Handle());
                commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline.Layout(), 0,
                                                 bindings.Handles(), {});
                commandBuffer.dispatch(static_cast<std::uint32_t>(input.size()), 1, 1);

                barrier(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
                        vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead);

                commandBuffer.copyBuffer(outputBuffer.Handle(), readbackBuffer.Handle(), copyRegion);

                barrier(vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite,
                        vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostRead);

                commands.End();
                commands.SubmitAndWait();

                std::array<std::uint32_t, 4> result{};
                readbackBuffer.Read(std::as_writable_bytes(std::span(result)));

                if (result != expectedResults[round])
                {
                    throw std::runtime_error("Compute result mismatch in round " + std::to_string(round + 1) + ".");
                }

                std::cout << "Round " << round + 1 << " passed.\n";
            }
        }

        std::cout << "Vulkan smoke completed; all resources destroyed.\n";
        return 0;
    }
    catch (const std::exception &exception)
    {
        std::cerr << "Vulkan smoke failed: " << exception.what() << '\n';
        return 1;
    }
}