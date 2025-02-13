#pragma once
#include <se.core.hpp>
#include <se.rhi.hpp>
#include <filesystem>

// ===========================================================================
// Base class definition

namespace se::image {
// host texture
struct SIByL_API Texture {
  /** virtual destructor */
  virtual ~Texture() = default;
  /** virtual constructor */
  Texture() = default;
  Texture(Texture&&) = default;
  Texture(Texture const&) = delete;
  auto operator=(Texture&&) ->Texture & = default;
  auto operator=(Texture const&) ->Texture & = delete;
  /** member functions */
  auto getDescriptor() noexcept -> rhi::TextureDescriptor;
  auto getData() noexcept -> char const*;

  rhi::Extend3D extend;
  rhi::TextureFormat format;
  rhi::TextureDimension dimension;
  se::buffer buffer;
  uint32_t mip_levels = 0;
  uint32_t array_layers = 0;
  uint32_t data_offset = 0;
  uint32_t data_size = 0;

  struct SubResource {
    uint32_t mip;
    uint32_t level;
    uint32_t offset;
    uint32_t size;
    uint32_t width;
    uint32_t height;
  };
  std::vector<SubResource> subResources;
};

struct SIByL_API PNG {
  static auto writePNG(std::filesystem::path const& path, uint32_t width,
    uint32_t height, uint32_t channel, float* data) noexcept -> void;
  static auto fromPNG(std::filesystem::path const& path) noexcept -> std::unique_ptr<Texture>;
};

struct SIByL_API JPEG {
  static auto writeJPEG(std::filesystem::path const& path, uint32_t width,
    uint32_t height, uint32_t channel, float* data) noexcept -> void;
  static auto fromJPEG(std::filesystem::path const& path) noexcept -> std::unique_ptr<Texture>;
};

struct SIByL_API EXR {
  static auto writeEXR(std::filesystem::path const& path, uint32_t width,
    uint32_t height, uint32_t channel, float* data) noexcept -> void;
  static auto fromEXR(std::filesystem::path const& path) noexcept -> std::unique_ptr<Texture>;
};

struct SIByL_API Binary {
  static auto fromBinary(int width, int height, int channel, int bits,
    const char* path) noexcept -> std::unique_ptr<Texture>;
};

auto SIByL_API load_image(std::filesystem::path const& path) noexcept -> std::unique_ptr<Texture>;
}