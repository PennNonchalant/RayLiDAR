#pragma once

#include "LidarSensor.h"
#include "vulkan/ComputePipeline.h"
#include "vulkan/DescriptorBindings.h"
#include "vulkan/Shader.h"

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <filesystem>

namespace raylidar::vulkan
{
class Buffer;
class VulkanContext;
} // namespace raylidar::vulkan

namespace raylidar::simulation
{
// Fills a ray buffer on the GPU instead of building rays on the CPU and uploading them.
class GpuRayGenerator
{
  public:
    GpuRayGenerator(const vulkan::VulkanContext &context, const std::filesystem::path &rayGenShaderPath);

    GpuRayGenerator(const GpuRayGenerator &) = delete;
    GpuRayGenerator &operator=(const GpuRayGenerator &) = delete;

    GpuRayGenerator(GpuRayGenerator &&) = delete;
    GpuRayGenerator &operator=(GpuRayGenerator &&) = delete;

    ~GpuRayGenerator() = default;

    // rays: a storage buffer with room for BeamCount * AzimuthCount rays.
    void SetOutput(const vulkan::Buffer &rays);

    // Records the dispatch only; the caller adds a barrier before anything reads the rays.
    void Record(const vk::raii::CommandBuffer &commandBuffer, const LidarSensorDescription &sensor,
                const Transform &sensorToWorld) const;

  private:
    vulkan::Shader m_shader;
    vulkan::ComputePipeline m_pipeline;
    vulkan::DescriptorBindings m_bindings;

    std::uint32_t m_rayCapacity = 0;
};
} // namespace raylidar::simulation
