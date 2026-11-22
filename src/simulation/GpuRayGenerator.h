#pragma once

#include "LidarSensor.h"
#include "vulkan/Buffer.h"
#include "vulkan/ComputePipeline.h"
#include "vulkan/DescriptorBindings.h"
#include "vulkan/Shader.h"

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <filesystem>

namespace raylidar::vulkan
{
class VulkanContext;
} // namespace raylidar::vulkan

namespace raylidar::simulation
{
// Fills a ray buffer on the GPU instead of building rays on the CPU and uploading them.
class GpuRayGenerator
{
  public:
    GpuRayGenerator(const vulkan::VulkanContext &context, const std::filesystem::path &rayGenShaderPath,
                    const LidarSensor &sensor);

    GpuRayGenerator(const GpuRayGenerator &) = delete;
    GpuRayGenerator &operator=(const GpuRayGenerator &) = delete;

    GpuRayGenerator(GpuRayGenerator &&) = delete;
    GpuRayGenerator &operator=(GpuRayGenerator &&) = delete;

    ~GpuRayGenerator() = default;

    void SetBuffers(const vulkan::Buffer &rays, const vulkan::Buffer &rayIndices);

    // Records the dispatch only; the caller adds a barrier before anything reads the rays.
    void Record(const vk::raii::CommandBuffer &commandBuffer, const LidarSensorDescription &sensor,
                const Transform &sensorToWorld, std::uint32_t firstRay, std::uint32_t rayCount,
                bool useRayIndices = false) const;

  private:
    vulkan::Shader m_shader;
    vulkan::ComputePipeline m_pipeline;
    vulkan::Buffer m_localDirections;
    vulkan::DescriptorBindings m_bindings;

    std::uint32_t m_rayCapacity = 0;
};
} // namespace raylidar::simulation
