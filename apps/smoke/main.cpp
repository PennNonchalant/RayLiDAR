#include "rt/ASBuilder.h"
#include "rt/GaussianScene.h"
#include "rt/MeshScene.h"
#include "rt/RayQueryTracer.h"
#include "rt/TraceTypes.h"

#include "vulkan/Buffer.h"
#include "vulkan/CommandContext.h"
#include "vulkan/ComputePipeline.h"
#include "vulkan/DescriptorBindings.h"
#include "vulkan/Shader.h"
#include "vulkan/Synchronization.h"
#include "vulkan/VulkanContext.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{
void RunMeshRayQuerySmoke(const raylidar::vulkan::VulkanContext &context, const std::filesystem::path &shaderPath)
{
    const std::array<std::array<float, 3>, 3> vertices = {{
        {-1.0f, -1.0f, 0.0f},
        {1.0f, -1.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
    }};

    const std::array<std::uint32_t, 3> indices = {0, 1, 2};

    const raylidar::rt::MeshDescription mesh{vertices, indices};
    const raylidar::rt::MeshInstanceDescription instance{};

    raylidar::rt::MeshSceneDescription sceneDescription;
    sceneDescription.Meshes = std::span(&mesh, 1);
    sceneDescription.Instances = std::span(&instance, 1);

    raylidar::rt::MeshScene scene(context, sceneDescription);
    raylidar::rt::ASBuilder builder(context);

    const raylidar::vulkan::Shader traceShader(context, shaderPath);
    raylidar::rt::RayQueryTracer tracer(context, traceShader);

    constexpr std::size_t rayCount = 65;
    std::array<raylidar::rt::Ray, rayCount> rays{};

    for (std::size_t index = 0; index < rays.size(); ++index)
    {
        rays[index].Origin = {index % 2 == 0 ? 0.0f : 3.0f, 0.0f, -1.0f};
        rays[index].TMax = 100.0f + static_cast<float>(index);
    }

    const auto rayBytes = std::as_bytes(std::span(rays));
    const vk::DeviceSize hitByteCount = rayCount * sizeof(raylidar::rt::Hit);

    raylidar::vulkan::BufferDescription rayDescription;
    rayDescription.Size = static_cast<vk::DeviceSize>(rayBytes.size());
    rayDescription.Usage = vk::BufferUsageFlagBits::eStorageBuffer;
    rayDescription.MemoryUsage = raylidar::vulkan::BufferMemoryUsage::Upload;

    raylidar::vulkan::Buffer rayBuffer(context, rayDescription);
    rayBuffer.Write(rayBytes);

    raylidar::vulkan::BufferDescription hitDescription;
    hitDescription.Size = hitByteCount;
    hitDescription.Usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc;

    const raylidar::vulkan::Buffer hitBuffer(context, hitDescription);

    raylidar::vulkan::BufferDescription readbackDescription;
    readbackDescription.Size = hitByteCount;
    readbackDescription.Usage = vk::BufferUsageFlagBits::eTransferDst;
    readbackDescription.MemoryUsage = raylidar::vulkan::BufferMemoryUsage::Readback;

    const raylidar::vulkan::Buffer readbackBuffer(context, readbackDescription);

    raylidar::vulkan::CommandContext traceCommands(context);
    const auto &commandBuffer = traceCommands.Begin();

    scene.RecordBuild(commandBuffer, builder);

    tracer.SetResources(scene, rayBuffer, hitBuffer);

    raylidar::rt::MeshTraceParameters parameters;
    parameters.RayCount = static_cast<std::uint32_t>(rays.size());

    tracer.RecordTrace(commandBuffer, parameters);

    raylidar::vulkan::RecordMemoryBarrier(commandBuffer, vk::PipelineStageFlagBits2::eComputeShader,
                                          vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eCopy,
                                          vk::AccessFlagBits2::eTransferRead);

    commandBuffer.copyBuffer(hitBuffer.Handle(), readbackBuffer.Handle(), vk::BufferCopy(0, 0, hitByteCount));

    raylidar::vulkan::RecordMemoryBarrier(commandBuffer, vk::PipelineStageFlagBits2::eCopy,
                                          vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eHost,
                                          vk::AccessFlagBits2::eHostRead);

    traceCommands.End();
    traceCommands.SubmitAndWait();

    scene.ReleaseUploadResources();

    std::array<raylidar::rt::Hit, rayCount> hits{};
    readbackBuffer.Read(std::as_writable_bytes(std::span(hits)));

    constexpr std::uint32_t invalidId = std::numeric_limits<std::uint32_t>::max();
    const std::array<float, 2> zeroBarycentrics{};

    for (std::size_t index = 0; index < hits.size(); ++index)
    {
        const auto &hit = hits[index];

        if (hit.T != rays[index].TMax || hit.Valid != 0 || hit.InstanceId != invalidId ||
            hit.PrimitiveId != invalidId || hit.Barycentrics != zeroBarycentrics)
        {
            throw std::runtime_error("Ray query I/O smoke failed at ray " + std::to_string(index) + ".");
        }
    }

    std::cout << "Ray query I/O smoke passed.\n";
}

void RunComputeSmoke(const raylidar::vulkan::VulkanContext &context, const std::filesystem::path &shaderPath)
{
    const raylidar::vulkan::Shader shader(context, shaderPath);

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

        const vk::BufferCopy copyRegion(0, 0, byteCount);

        commandBuffer.copyBuffer(uploadBuffer.Handle(), inputBuffer.Handle(), copyRegion);

        raylidar::vulkan::RecordMemoryBarrier(
            commandBuffer, vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead);

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.Handle());
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline.Layout(), 0, bindings.Handles(), {});
        commandBuffer.dispatch(static_cast<std::uint32_t>(input.size()), 1, 1);

        raylidar::vulkan::RecordMemoryBarrier(commandBuffer, vk::PipelineStageFlagBits2::eComputeShader,
                                              vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eCopy,
                                              vk::AccessFlagBits2::eTransferRead);

        commandBuffer.copyBuffer(outputBuffer.Handle(), readbackBuffer.Handle(), copyRegion);

        raylidar::vulkan::RecordMemoryBarrier(commandBuffer, vk::PipelineStageFlagBits2::eCopy,
                                              vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eHost,
                                              vk::AccessFlagBits2::eHostRead);

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

void RunGaussianRayQuerySmoke(const raylidar::vulkan::VulkanContext &context, const std::filesystem::path &shaderPath)
{
    constexpr std::array<raylidar::rt::Gaussian, 6> gaussians{{
        {{0.0f, 0.0f, 2.0f}, 0.3f},
        {{8.0f, 0.0f, 2.0f}, 0.3f},
        {{8.125f, 0.0f, 2.0f}, 0.3f},
        {{16.0f, 0.0f, 4.0f}, 0.3f},
        {{16.0f, 0.0f, 2.0f}, 0.3f},
        {{24.0f, 0.0f, 2.0f}, 0.5f},
    }};

    constexpr float tMax = 10.0f;
    constexpr float alphaThreshold = 0.5f;
    constexpr std::array<raylidar::rt::Ray, 4> rays{{
        {.Origin = {0.0f, 0.0f, 0.0f}, .TMax = tMax},
        {.Origin = {8.0625f, 0.0f, 0.0f}, .TMax = tMax},
        {.Origin = {16.0f, 0.0f, 0.0f}, .TMax = tMax},
        {.Origin = {24.0f, 0.0f, 0.0f}, .TMax = tMax},
    }};
    constexpr std::array<raylidar::rt::Hit, rays.size()> expectedHits{{
        {tMax, 0},
        {2.0f, 1},
        {4.0f, 1},
        {2.0f, 1},
    }};

    raylidar::rt::GaussianScene scene(context, {gaussians});
    raylidar::rt::ASBuilder builder(context);
    const raylidar::vulkan::Shader shader(context, shaderPath);
    raylidar::rt::RayQueryTracer tracer(context, shader, raylidar::rt::RayQueryMode::Gaussian);

    const auto rayBytes = std::as_bytes(std::span(rays));
    const auto storageUsage = vk::BufferUsageFlagBits::eStorageBuffer;
    raylidar::vulkan::Buffer rayBuffer(context,
                                       {rayBytes.size(), storageUsage, raylidar::vulkan::BufferMemoryUsage::Upload});
    rayBuffer.Write(rayBytes);

    std::array<raylidar::rt::Hit, rays.size()> hits{};
    const auto hitBytes = std::as_writable_bytes(std::span(hits));
    const raylidar::vulkan::Buffer hitBuffer(
        context, {hitBytes.size(), storageUsage, raylidar::vulkan::BufferMemoryUsage::Readback});

    raylidar::vulkan::CommandContext commands(context);
    const auto &commandBuffer = commands.Begin();
    scene.RecordBuild(commandBuffer, builder);
    tracer.SetResources(scene, rayBuffer, hitBuffer);

    const raylidar::rt::GaussianTraceParameters parameters{
        static_cast<std::uint32_t>(rays.size()), scene.GaussiansPerBatch(), scene.SigmaCutoff(), alphaThreshold};
    tracer.RecordTrace(commandBuffer, parameters);
    raylidar::vulkan::RecordMemoryBarrier(commandBuffer, vk::PipelineStageFlagBits2::eComputeShader,
                                          vk::AccessFlagBits2::eShaderStorageWrite, vk::PipelineStageFlagBits2::eHost,
                                          vk::AccessFlagBits2::eHostRead);

    commands.End();
    commands.SubmitAndWait();
    scene.ReleaseUploadResources();

    hitBuffer.Read(hitBytes);
    for (std::size_t index = 0; index < hits.size(); ++index)
    {
        if (hits[index].Valid != expectedHits[index].Valid || hits[index].T != expectedHits[index].T)
        {
            throw std::runtime_error("Gaussian ray query smoke failed at ray " + std::to_string(index) + ".");
        }
    }

    std::cout << "Gaussian ray query smoke passed.\n";
}
} // namespace

int main(int argc, char *argv[])
{
    try
    {
        if (argc != 4)
        {
            throw std::invalid_argument("Usage: raylidar_smoke <compute.spv> <ray-query.spv> <gaussian-trace.spv>");
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

            RunMeshRayQuerySmoke(context, argv[2]);
            RunComputeSmoke(context, argv[1]);
            RunGaussianRayQuerySmoke(context, argv[3]);
        }

        std::cout << "Vulkan smoke completed\n";
        return 0;
    }
    catch (const std::exception &exception)
    {
        std::cerr << "Vulkan smoke failed: " << exception.what() << '\n';
        return 1;
    }
}