#include "Shader.h"

#include "VulkanContext.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace raylidar::vulkan
{
namespace
{
std::vector<std::uint32_t> ReadShaderCode(const std::filesystem::path &filePath)
{
    std::ifstream stream(filePath, std::ios::binary | std::ios::ate);

    if (!stream)
    {
        throw std::runtime_error("Cannot open shader file: " + filePath.string());
    }

    const std::streamoff byteCount = stream.tellg();
    constexpr std::streamoff wordSize = sizeof(std::uint32_t);

    if (byteCount <= 0 || byteCount % wordSize != 0)
    {
        throw std::runtime_error("Invalid SPIR-V file size: " + filePath.string());
    }

    std::vector<std::uint32_t> code(static_cast<std::size_t>(byteCount / wordSize));

    stream.seekg(0, std::ios::beg);

    if (!stream.read(reinterpret_cast<char *>(code.data()), static_cast<std::streamsize>(byteCount)))
    {
        throw std::runtime_error("Cannot read shader file: " + filePath.string());
    }

    return code;
}
} // namespace

Shader::Shader(const VulkanContext &context, const std::filesystem::path &filePath)
{
    const auto code = ReadShaderCode(filePath);

    vk::ShaderModuleCreateInfo shaderModuleInfo;
    shaderModuleInfo.setCode(code);

    m_shaderModule = vk::raii::ShaderModule(context.Device(), shaderModuleInfo);
}
} // namespace raylidar::vulkan