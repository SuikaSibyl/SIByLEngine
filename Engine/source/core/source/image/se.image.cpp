#define DLIB_EXPORT
#include <se.image.hpp>
#undef DLIB_EXPORT
#include <tinyexr.h>
#define STB_IMAGE_IMPLEMENTATION
#include "ex.stb.image.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#pragma warning(disable:4996)
#include "ex.stb.image-write.hpp"

namespace se::image {
auto Texture::getDescriptor() noexcept -> rhi::TextureDescriptor {
  return rhi::TextureDescriptor{
    extend, mip_levels,
    array_layers, 1,
    dimension, format,
    (uint32_t)rhi::TextureUsageBit::COPY_DST |
    (uint32_t)rhi::TextureUsageBit::TEXTURE_BINDING,
    {format}};
}

auto Texture::getData() noexcept -> char const* {
  return &(static_cast<char const*>(buffer.data)[data_offset]);
}

auto PNG::writePNG(std::filesystem::path const& path, uint32_t width,
  uint32_t height, uint32_t channel, float* data) noexcept -> void {
  stbi_write_png(path.string().c_str(), width, height, channel, data,
    width * channel);
}

auto PNG::fromPNG(std::filesystem::path const& path) noexcept -> std::unique_ptr<Texture> {
  int texWidth, texHeight, texChannels;
  stbi_uc* pixels = stbi_load(path.string().c_str(), &texWidth, &texHeight,
                              &texChannels, STBI_rgb_alpha);
  if (!pixels) {
    root::print::error("Image :: failed to load texture image!");
    return nullptr;
  }
  std::unique_ptr<Texture> image = std::make_unique<Texture>();
  image->buffer = se::buffer(texWidth * texHeight * sizeof(uint8_t) * 4);
  memcpy(image->buffer.data, pixels, texWidth * texHeight * sizeof(uint8_t) * 4);
  image->extend = rhi::Extend3D{ (uint32_t)texWidth, (uint32_t)texHeight, 1 };
  image->format = rhi::TextureFormat::RGBA8_UNORM_SRGB;
  image->dimension = rhi::TextureDimension::TEX2D;
  image->data_size = image->buffer.size;
  image->mip_levels = 1;
  image->array_layers = 1;
  image->subResources.push_back(Texture::SubResource{ 0, 0, 0,
    uint32_t(image->buffer.size), uint32_t(texWidth), uint32_t(texHeight) });
  stbi_image_free(pixels);
  return image;
}

auto JPEG::writeJPEG(std::filesystem::path const& path, uint32_t width,
  uint32_t height, uint32_t channel, float* data) noexcept -> void {
  stbi_write_jpg(path.string().c_str(), width, height, channel, data,
    width * channel);
}

auto JPEG::fromJPEG(std::filesystem::path const& path) noexcept -> std::unique_ptr<Texture> {
  int texWidth, texHeight, texChannels;
  stbi_uc* pixels = stbi_load(path.string().c_str(), &texWidth, &texHeight,
    &texChannels, STBI_rgb_alpha);
  if (!pixels) {
    root::print::error("Image :: failed to load texture image!");
    return nullptr;
  }
  std::unique_ptr<Texture> image = std::make_unique<Texture>();
  image->buffer = se::buffer(texWidth * texHeight * sizeof(uint8_t) * 4);
  memcpy(image->buffer.data, pixels, texWidth * texHeight * sizeof(uint8_t) * 4);
  image->extend = rhi::Extend3D{ (uint32_t)texWidth, (uint32_t)texHeight, 1 };
  image->format = rhi::TextureFormat::RGBA8_UNORM_SRGB;
  image->dimension = rhi::TextureDimension::TEX2D;
  image->data_size = image->buffer.size;
  image->mip_levels = 1;
  image->array_layers = 1;
  image->subResources.push_back(Texture::SubResource{ 0, 0, 0,
    uint32_t(image->buffer.size), uint32_t(texWidth), uint32_t(texHeight) });
  stbi_image_free(pixels);
  return image;
}

auto EXR::writeEXR(std::filesystem::path const& path, uint32_t width,
  uint32_t height, uint32_t channel, float* data) noexcept -> void {
  EXRHeader header;
  InitEXRHeader(&header);
  EXRImage image;
  InitEXRImage(&image);
  if (channel == 3) {
      image.num_channels = 3;

      std::vector<float> images[3];
      images[0].resize(width * height);
      images[1].resize(width * height);
      images[2].resize(width * height);

      // Split RGBRGBRGB... into R, G and B layer
      for (int i = 0; i < width * height; i++) {
          images[0][i] = data[4 * i + 0];
          images[1][i] = data[4 * i + 1];
          images[2][i] = data[4 * i + 2];
      }

      float* image_ptr[3];
      image_ptr[0] = &(images[2].at(0));  // B
      image_ptr[1] = &(images[1].at(0));  // G
      image_ptr[2] = &(images[0].at(0));  // R

      image.images = (unsigned char**)image_ptr;
      image.width = width;
      image.height = height;

      header.num_channels = 3;
      header.channels =
          (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo) * header.num_channels);
      // Must be (A)BGR order, since most of EXR viewers expect this channel order.
      strncpy(header.channels[0].name, "B", 255);
      header.channels[0].name[strlen("B")] = '\0';
      strncpy(header.channels[1].name, "G", 255);
      header.channels[1].name[strlen("G")] = '\0';
      strncpy(header.channels[2].name, "R", 255);
      header.channels[2].name[strlen("R")] = '\0';

      header.pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
      header.requested_pixel_types =
          (int*)malloc(sizeof(int) * header.num_channels);
      for (int i = 0; i < header.num_channels; i++) {
          header.pixel_types[i] =
              TINYEXR_PIXELTYPE_FLOAT;  // pixel type of input image
          header.requested_pixel_types[i] =
              TINYEXR_PIXELTYPE_HALF;  // pixel type of output image to be stored in
          // .EXR
      }

      const char* err = NULL;  // or nullptr in C++11 or later.
      int ret = SaveEXRImageToFile(&image, &header, path.string().c_str(), &err);
      if (ret != TINYEXR_SUCCESS) {
          root::print::error("Save EXR err: " + std::string(err));
          FreeEXRErrorMessage(err);  // free's buffer for an error message
          return;
      }
  }
  else if (channel == 4) {
      image.num_channels = 4;

      std::vector<float> images[4];
      images[0].resize(width * height);
      images[1].resize(width * height);
      images[2].resize(width * height);
      images[3].resize(width * height);

      // Split RGBRGBRGB... into R, G and B layer
      for (int i = 0; i < width * height; i++) {
          images[0][i] = data[4 * i + 0];
          images[1][i] = data[4 * i + 1];
          images[2][i] = data[4 * i + 2];
          images[3][i] = data[4 * i + 3];
      }

      float* image_ptr[4];
      image_ptr[0] = &(images[3].at(0));  // A
      image_ptr[1] = &(images[2].at(0));  // B
      image_ptr[2] = &(images[1].at(0));  // G
      image_ptr[3] = &(images[0].at(0));  // R

      image.images = (unsigned char**)image_ptr;
      image.width = width;
      image.height = height;

      header.num_channels = 4;
      header.channels =
          (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo) * header.num_channels);
      // Must be (A)BGR order, since most of EXR viewers expect this channel order.
      strncpy(header.channels[0].name, "A", 255);
      header.channels[3].name[strlen("A")] = '\0';
      strncpy(header.channels[1].name, "B", 255);
      header.channels[0].name[strlen("B")] = '\0';
      strncpy(header.channels[2].name, "G", 255);
      header.channels[1].name[strlen("G")] = '\0';
      strncpy(header.channels[3].name, "R", 255);
      header.channels[2].name[strlen("R")] = '\0';

      header.pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
      header.requested_pixel_types =
          (int*)malloc(sizeof(int) * header.num_channels);
      for (int i = 0; i < header.num_channels; i++) {
          header.pixel_types[i] =
              TINYEXR_PIXELTYPE_FLOAT;  // pixel type of input image
          header.requested_pixel_types[i] =
              TINYEXR_PIXELTYPE_FLOAT;  // pixel type of output image to be stored in
          // .EXR
      }

      const char* err = NULL;  // or nullptr in C++11 or later.
      int ret = SaveEXRImageToFile(&image, &header, path.string().c_str(), &err);
      if (ret != TINYEXR_SUCCESS) {
          root::print::error("Save EXR err: " + std::string(err));
          FreeEXRErrorMessage(err);  // free's buffer for an error message
          return;
      }
  }
  
  free(header.channels);
  free(header.pixel_types);
  free(header.requested_pixel_types);
}

auto EXR::fromEXR(std::filesystem::path const& path) noexcept
-> std::unique_ptr<Texture> {
  std::string const& str_path = path.string();
  const char* input = str_path.c_str();
  float* out;  // width * height * RGBA
  int width;
  int height;
  const char* err = nullptr;

  int ret = LoadEXR(&out, &width, &height, input, &err);

  if (ret != TINYEXR_SUCCESS) {
    if (err) {
      fprintf(stderr, "ERR : %s\n", err);
      FreeEXRErrorMessage(err);  // release memory of error message.
    }
  } else {
    std::unique_ptr<Texture> image = std::make_unique<Texture>();
    image->buffer = se::buffer(width * height * sizeof(float) * 4);
    memcpy(image->buffer.data, out, width * height * sizeof(float) * 4);
    image->extend = rhi::Extend3D{ (uint32_t)width, (uint32_t)height, 1 };
    image->format = rhi::TextureFormat::RGBA32_FLOAT;
    image->dimension = rhi::TextureDimension::TEX2D;
    image->data_size = image->buffer.size;
    image->mip_levels = 1;
    image->array_layers = 1;
    image->subResources.push_back(Texture::SubResource{ 0, 0, 0, 
      uint32_t(image->buffer.size), uint32_t(width), uint32_t(height) });
    free(out);
    return image;
  }
  return nullptr;
}

auto Binary::fromBinary(int texWidth, int texHeight, int texChannels, int bits,
  const char* pixels) noexcept -> std::unique_ptr<Texture> {
  std::unique_ptr<Texture> image = std::make_unique<Texture>();
  image->buffer = se::buffer(texWidth * texHeight * sizeof(uint8_t) * 4);
  memcpy(image->buffer.data, pixels, texWidth * texHeight * sizeof(uint8_t) * 4);
  image->extend = rhi::Extend3D{ (uint32_t)texWidth, (uint32_t)texHeight, 1 };
  image->format = rhi::TextureFormat::RGBA8_UNORM_SRGB;
  image->dimension = rhi::TextureDimension::TEX2D;
  image->data_size = image->buffer.size;
  image->mip_levels = 1;
  image->array_layers = 1;
  image->subResources.push_back(Texture::SubResource{ 0, 0, 0,
    uint32_t(image->buffer.size), uint32_t(texWidth), uint32_t(texHeight) });
  return image;
}

auto load_image(std::filesystem::path const& path) noexcept -> std::unique_ptr<Texture> {
if (path.extension() == ".jpg" || path.extension() == ".JPG" ||
    path.extension() == ".JPEG") {
  return JPEG::fromJPEG(path);
}
else if (path.extension() == ".png" || path.extension() == ".PNG") {
  return PNG::fromPNG(path);
}
else if (path.extension() == ".exr") {
  return EXR::fromEXR(path);
}
else {
  root::print::error(
    std::format("Image :: Image Loader failed when loading {0}, \
    as format extension {1} not supported. ", path.string(), path.extension().string()));
}
return nullptr;
}

}