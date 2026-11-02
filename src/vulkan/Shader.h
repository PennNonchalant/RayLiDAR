#pragma once

#include <vulkan/vulkan_raii.hpp>

#include <cassert>
#include <filesystem>

namespace raylidar::vulkan
{
class VulkanContext;

class Shader
{
  public:
    Shader(const VulkanContext &context, const std::filesystem::path &filePath);
    ~Shader() = default;

    Shader(const Shader &) = delete;
    Shader &operator=(const Shader &) = delete;

    Shader(Shader &&) noexcept = default;
    Shader &operator=(Shader &&) noexcept = default;

    [[nodiscard]]
    vk::ShaderModule Handle() const noexcept
    {
        return *m_shaderModule;
    }

  private:
    vk::raii::ShaderModule m_shaderModule = nullptr;
};
} // namespace raylidar::vulkan
