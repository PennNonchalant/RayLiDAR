#include "GpuRayGenerator.h"

#include "rt/TraceTypes.h"
#include "vulkan/VulkanContext.h"

#include <array>
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

    float RangeMin = 0.0f;
    float RangeMax = 0.0f;
    std::uint32_t FirstRay = 0;
    std::uint32_t RayCount = 0;
    std::uint32_t UseRayIndices = 0;
};

static_assert(sizeof(RayGenParameters) == 68, "RayGenParameters must match the shader layout.");
static_assert(sizeof(rt::Ray) == 32, "rt::Ray must match the shader layout.");

vulkan::ComputePipelineDescription MakePipelineDescription()
{
    vulkan::ComputePipelineDescription description;
    description.PushConstantSize = sizeof(RayGenParameters);
    description.DescriptorSets.resize(1);
    description.DescriptorSets[0].Bindings = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
    };
    return description;
}
} // namespace

GpuRayGenerator::GpuRayGenerator(const vulkan::VulkanContext &context, const std::filesystem::path &rayGenShaderPath,
                                 const LidarSensor &sensor)
    : m_shader(context, rayGenShaderPath), m_pipeline(context, m_shader, MakePipelineDescription()),
      m_localDirections(context, {.Size = sensor.LocalDirections().size_bytes(),
                                  .Usage = vk::BufferUsageFlagBits::eStorageBuffer,
                                  .MemoryUsage = vulkan::BufferMemoryUsage::Upload}),
      m_bindings(context, m_pipeline)
{
    m_localDirections.Write(std::as_bytes(sensor.LocalDirections()));
    m_bindings.WriteBuffer(0, 2, m_localDirections);
}

void GpuRayGenerator::SetBuffers(const vulkan::Buffer &rays, const vulkan::Buffer &rayIndices)
{
    if (!(rays.Usage() & vk::BufferUsageFlagBits::eStorageBuffer))
    {
        throw std::invalid_argument("GpuRayGenerator output requires storage buffer usage.");
    }

    m_bindings.WriteBuffer(0, 0, rays);
    m_bindings.WriteBuffer(0, 1, rayIndices);
    m_rayCapacity = static_cast<std::uint32_t>(rays.Size() / sizeof(rt::Ray));
}

void GpuRayGenerator::Record(const vk::raii::CommandBuffer &commandBuffer, const LidarSensorDescription &sensor,
                             const Transform &sensorToWorld, std::uint32_t firstRay, std::uint32_t rayCount,
                             bool useRayIndices) const
{
    if (rayCount == 0 || rayCount > m_rayCapacity)
    {
        throw std::out_of_range("Ray count is zero or exceeds the output buffer. Call SetBuffers first.");
    }

    RayGenParameters parameters;
    parameters.Rows = sensorToWorld;
    parameters.RangeMin = sensor.RangeMin;
    parameters.RangeMax = sensor.RangeMax;
    parameters.FirstRay = firstRay;
    parameters.RayCount = rayCount;
    parameters.UseRayIndices = useRayIndices;

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline.Handle());
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, m_pipeline.Layout(), 0, m_bindings.Handles(), {});
    commandBuffer.pushConstants<RayGenParameters>(m_pipeline.Layout(), vk::ShaderStageFlagBits::eCompute, 0,
                                                  parameters);
    commandBuffer.dispatch((rayCount + kWorkgroupSize - 1) / kWorkgroupSize, 1, 1);
}
} // namespace raylidar::simulation
