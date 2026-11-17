#include "GpuRayGenerator.h"

#include "rt/TraceTypes.h"
#include "vulkan/Buffer.h"
#include "vulkan/VulkanContext.h"

#include <array>
#include <numbers>
#include <stdexcept>

namespace raylidar::simulation
{
namespace
{
constexpr std::uint32_t kWorkgroupSize = 64;

// Must match RayGenParameters in shaders/LidarRayGen.slang.
struct RayGenParameters
{
    std::array<std::array<float, 4>, 3> Rows{};

    std::uint32_t BeamCount = 0;
    std::uint32_t AzimuthCount = 0;
    float ElevationMin = 0.0f;
    float ElevationMax = 0.0f;

    float RangeMin = 0.0f;
    float RangeMax = 0.0f;
    std::uint32_t Padding0 = 0;
    std::uint32_t Padding1 = 0;
};

static_assert(sizeof(RayGenParameters) == 80, "RayGenParameters must match the shader layout.");
static_assert(sizeof(rt::Ray) == 32, "rt::Ray must match the shader layout.");

vulkan::ComputePipelineDescription MakePipelineDescription()
{
    vulkan::ComputePipelineDescription description;
    description.PushConstantSize = sizeof(RayGenParameters);
    description.DescriptorSets.resize(1);
    description.DescriptorSets[0].Bindings = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
    };
    return description;
}
} // namespace

GpuRayGenerator::GpuRayGenerator(const vulkan::VulkanContext &context, const std::filesystem::path &rayGenShaderPath)
    : m_shader(context, rayGenShaderPath), m_pipeline(context, m_shader, MakePipelineDescription()),
      m_bindings(context, m_pipeline)
{
}

void GpuRayGenerator::SetOutput(const vulkan::Buffer &rays)
{
    if (!(rays.Usage() & vk::BufferUsageFlagBits::eStorageBuffer))
    {
        throw std::invalid_argument("GpuRayGenerator output requires storage buffer usage.");
    }

    m_bindings.WriteBuffer(0, 0, rays);
    m_rayCapacity = static_cast<std::uint32_t>(rays.Size() / sizeof(rt::Ray));
}

void GpuRayGenerator::Record(const vk::raii::CommandBuffer &commandBuffer, const LidarSensorDescription &sensor,
                             const Transform &sensorToWorld) const
{
    const std::uint32_t rayCount = sensor.BeamCount * sensor.AzimuthCount;

    if (rayCount == 0 || rayCount > m_rayCapacity)
    {
        throw std::out_of_range("Ray count is zero or exceeds the output buffer. Call SetOutput first.");
    }

    // Same float math as LidarSensor so both paths produce the same rays.
    constexpr float degreesToRadians = std::numbers::pi_v<float> / 180.0f;

    RayGenParameters parameters;
    parameters.Rows = sensorToWorld;
    parameters.BeamCount = sensor.BeamCount;
    parameters.AzimuthCount = sensor.AzimuthCount;
    parameters.ElevationMin = sensor.ElevationMinDegrees * degreesToRadians;
    parameters.ElevationMax = sensor.ElevationMaxDegrees * degreesToRadians;
    parameters.RangeMin = sensor.RangeMin;
    parameters.RangeMax = sensor.RangeMax;

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline.Handle());
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, m_pipeline.Layout(), 0, m_bindings.Handles(), {});
    commandBuffer.pushConstants<RayGenParameters>(m_pipeline.Layout(), vk::ShaderStageFlagBits::eCompute, 0,
                                                  parameters);
    commandBuffer.dispatch((rayCount + kWorkgroupSize - 1) / kWorkgroupSize, 1, 1);
}
} // namespace raylidar::simulation
