#include "se.gfx.hpp"
#include <tinyexr/includes/tinyexr.h>
#include "se.editor.hpp"
#include <imgui.h>
#include "se.gfx.scene-loader.hpp"
#include <span/span.hpp>

namespace se {
  namespace gfx {
    struct glTFLoaderEnv {
      std::string directory;
      std::vector<MediumHandle> mediums;
      std::unordered_map<int, Node> node2go;
      std::unordered_map<tinygltf::Texture const*, TextureHandle> textures;
      std::unordered_map<tinygltf::Material const*, MaterialHandle> materials;
    };

    auto loadGLTFMaterial(tinygltf::Material const* glmaterial, tinygltf::Model const* model,
      glTFLoaderEnv& env, gfx::Scene& gfxscene, MeshLoaderConfig meshConfig = {}) noexcept
      -> MaterialHandle {
      if (env.materials.find(glmaterial) != env.materials.end()) {
        return env.materials[glmaterial];
      }

      MaterialHandle mat = GFXContext::create_material_empty();
      std::string name = glmaterial->name;
      auto to_sampler = [&](int sampler_idx) {
        tinygltf::Sampler const& sampler = model->samplers[sampler_idx];
        rhi::SamplerDescriptor desc;
        // Min and Mipmap filter
        if (sampler.minFilter == TINYGLTF_TEXTURE_FILTER_NEAREST) {
          desc.minFilter = rhi::FilterMode::NEAREST;
        }
        else if (sampler.minFilter == TINYGLTF_TEXTURE_FILTER_LINEAR) {
          desc.minFilter = rhi::FilterMode::LINEAR;
        }
        else if (sampler.minFilter == TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST) {
          desc.minFilter = rhi::FilterMode::NEAREST;
          desc.mipmapFilter = rhi::MipmapFilterMode::NEAREST;
        }
        else if (sampler.minFilter == TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_NEAREST) {
          desc.minFilter = rhi::FilterMode::LINEAR;
          desc.mipmapFilter = rhi::MipmapFilterMode::NEAREST;
        }
        else if (sampler.minFilter == TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR) {
          desc.minFilter = rhi::FilterMode::NEAREST;
          desc.mipmapFilter = rhi::MipmapFilterMode::LINEAR;
        }
        else if (sampler.minFilter == TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR) {
          desc.minFilter = rhi::FilterMode::LINEAR;
          desc.mipmapFilter = rhi::MipmapFilterMode::LINEAR;
        }
        // Mag filter
        if (sampler.magFilter == TINYGLTF_TEXTURE_FILTER_NEAREST) {
          desc.magFilter = rhi::FilterMode::NEAREST;
        }
        else if (sampler.magFilter == TINYGLTF_TEXTURE_FILTER_LINEAR) {
          desc.magFilter = rhi::FilterMode::LINEAR;
        }
        // WarpS
        if (sampler.wrapS == TINYGLTF_TEXTURE_WRAP_REPEAT) {
          desc.addressModeU = rhi::AddressMode::REPEAT;
        }
        else if (sampler.wrapS == TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE) {
          desc.addressModeU = rhi::AddressMode::CLAMP_TO_EDGE;
        }
        else if (sampler.wrapS == TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT) {
          desc.addressModeU = rhi::AddressMode::MIRROR_REPEAT;
        }
        // WarpT
        if (sampler.wrapT == TINYGLTF_TEXTURE_WRAP_REPEAT) {
          desc.addressModeV = rhi::AddressMode::REPEAT;
        }
        else if (sampler.wrapT == TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE) {
          desc.addressModeV = rhi::AddressMode::CLAMP_TO_EDGE;
        }
        else if (sampler.wrapT == TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT) {
          desc.addressModeV = rhi::AddressMode::MIRROR_REPEAT;
        }
        return desc;
      };

      // load diffuse information
      { // load diffuse color
        if (glmaterial->pbrMetallicRoughness.baseColorFactor.size() > 0) {
          mat->m_packet.vec4Data0 = se::vec4{
            (float)glmaterial->pbrMetallicRoughness.baseColorFactor[0],
            (float)glmaterial->pbrMetallicRoughness.baseColorFactor[1],
            (float)glmaterial->pbrMetallicRoughness.baseColorFactor[2],
            (float)glmaterial->pbrMetallicRoughness.roughnessFactor,
          };
        }
        mat->m_packet.vec4Data1 = se::vec4{
            (float)glmaterial->emissiveFactor[0],
            (float)glmaterial->emissiveFactor[1],
            (float)glmaterial->emissiveFactor[2],
            (float)glmaterial->pbrMetallicRoughness.metallicFactor
        };
      }

      if (glmaterial->pbrMetallicRoughness.baseColorTexture.index != -1) {
        mat->m_basecolorTex = env.textures[&(model->textures[
          glmaterial->pbrMetallicRoughness.baseColorTexture.index
        ])];
      }

      auto& extras = glmaterial->extras;
      if (extras.Has("bxdf")) {
        mat->m_packet.bxdfType = extras.Get("bxdf").GetNumberAsInt();
      }
      if (extras.Has("custom_string")) {
        mat->m_customString = extras.Get("custom_string").Get<std::string>();
      }

      //if (extras.Has("additional_tex")) {
      //  int tex_id = extras.Get("additional_tex").GetNumberAsInt();
      //  mat->additionalTex = env.textures[&(model->textures[tex_id])];
      //}
      if (extras.Has("ext_vector_2")) {
        tinygltf::Value ext_vector_2 = extras.Get("ext_vector_2");
        mat->m_packet.vec4Data2[0] = (float)ext_vector_2.Get(0).GetNumberAsDouble();
        mat->m_packet.vec4Data2[1] = (float)ext_vector_2.Get(1).GetNumberAsDouble();
        mat->m_packet.vec4Data2[2] = (float)ext_vector_2.Get(2).GetNumberAsDouble();
        mat->m_packet.vec4Data2[3] = (float)ext_vector_2.Get(3).GetNumberAsDouble();
      }
      { // load diffuse texture
        //if (glmaterial->pbrMetallicRoughness.baseColorTexture.index != -1) {
        //  mat->basecolorTex = env.textures[&(model->textures[glmaterial->pbrMetallicRoughness.baseColorTexture.index])];
        //}
      }

      mat->m_name = glmaterial->name;
      env.materials[glmaterial] = mat;
      return mat;
    }

    template <typename T>
    struct ArrayAdapter {
      /**
       * Construct an array adapter.
       * @param ptr Pointer to the start of the data, with offset applied
       * @param count Number of elements in the array
       * @param byte_stride Stride betweens elements in the array
       */
      ArrayAdapter(const unsigned char* ptr, size_t count, size_t byte_stride)
        : dataPtr(ptr), elemCount(count), stride(byte_stride) {}

      /// Returns a *copy* of a single element. Can't be used to modify it.
      T operator[](size_t pos) const {
        if (pos >= elemCount)
          throw std::out_of_range(
            "Tried to access beyond the last element of an array adapter with "
            "count " +
            std::to_string(elemCount) + " while getting elemnet number " +
            std::to_string(pos));
        return *(reinterpret_cast<const T*>(dataPtr + pos * stride));
      }
      /** Pointer to the bytes */
      unsigned const char* dataPtr;
      /** Number of elements in the array */
      const size_t elemCount;
      /** Stride in bytes between two elements */
      const size_t stride;
    };

    static inline auto loadGLTFMesh(tinygltf::Mesh const& gltfmesh,
      Node& gfxNode, Scene& scene, int node_id, tinygltf::Model const* model,
      glTFLoaderEnv& env) noexcept -> MeshHandle {
      // check for extensions and extras
      tinygltf::Value mesh_extra = gltfmesh.extras;

      auto handle_material_medium = [&](
        tinygltf::Primitive const& meshPrimitive,
        Mesh::CustomPrimitive& sePrimitive
      ) {
        if (meshPrimitive.material != -1) {
          auto const& gltf_material = model->materials[meshPrimitive.material];
          sePrimitive.material = loadGLTFMaterial(&gltf_material, model, env, scene, defaultMeshLoadConfig);
          se::vec3 emission = sePrimitive.material->m_packet.vec4Data1.xyz();
          if (emission.r > 0 || emission.g > 0 || emission.b > 0) {
            auto& light_component = gfxNode.add_component<Light>();
            //light_component.primitives.push_back(0);
            light_component.light.light_type = LightTypeEnum::MESH_PRIMITIVE;
          }
        }
        else {
          sePrimitive.material = GFXContext::create_material_empty();
        }
        tinygltf::Value primitive_extra = meshPrimitive.extras;
        if (primitive_extra.Has("exterior")) {
          int exterior_index = primitive_extra.Get("exterior").GetNumberAsInt();
          if (exterior_index >= 0) sePrimitive.exterior = env.mediums[exterior_index];
        }
        if (primitive_extra.Has("interior")) {
          int interior_index = primitive_extra.Get("interior").GetNumberAsInt();
          if (interior_index >= 0) sePrimitive.interior = env.mediums[interior_index];
        }
      };

      if(mesh_extra.Has("custom_primitive")) {
        size_t primitive_count = gltfmesh.primitives.size();
        if (primitive_count != 1) {
          se::error("Custom primitive mesh must have exactly one primitive.");
          return MeshHandle{};
        }
        auto const& meshPrimitive = gltfmesh.primitives[0];

        if (meshPrimitive.extras.Has("primitive_type")) {
          // custom primitive handling
          std::string primitive_type = meshPrimitive.extras.Get("primitive_type").Get<std::string>();
          if (primitive_type == "sphere") {
            std::string engine_path = Configuration::string_property("engine_path");
            std::string obj_path = engine_path + "assets/meshes/sphere.obj";
            MeshHandle mesh = load_obj_mesh(obj_path, scene);
            mesh->m_customPrimitives.emplace_back();
            Mesh::CustomPrimitive& sphere_primitive = mesh->m_customPrimitives.back();
            sphere_primitive.primitiveType = 1;
            sphere_primitive.min = -vec3(1.f);
            sphere_primitive.max = vec3(1.f);
            handle_material_medium(meshPrimitive, sphere_primitive);
            return mesh;
          }
          else if (primitive_type == "cube") {
            std::string engine_path = Configuration::string_property("engine_path");
            std::string obj_path = engine_path + "assets/meshes/cube.obj";
            MeshHandle mesh = load_obj_mesh(obj_path, scene);
            mesh->m_customPrimitives.emplace_back();
            Mesh::CustomPrimitive& cube_primitive = mesh->m_customPrimitives.back();
            cube_primitive.primitiveType = 3;
            cube_primitive.min = -vec3(1.f);
            cube_primitive.max = vec3(1.f);
            handle_material_medium(meshPrimitive, cube_primitive);
            return mesh;
          }
          else if (primitive_type == "rectangle") {
            std::string engine_path = Configuration::string_property("engine_path");
            std::string obj_path = engine_path + "assets/meshes/rectangle.obj";
            MeshHandle mesh = load_obj_mesh(obj_path, scene);
            mesh->m_customPrimitives.emplace_back();
            Mesh::CustomPrimitive& rectangle_primitive = mesh->m_customPrimitives.back();
            rectangle_primitive.primitiveType = 2;
            rectangle_primitive.min = -vec3(1.f);
            rectangle_primitive.max = vec3(1.f);
            handle_material_medium(meshPrimitive, rectangle_primitive);
            return mesh;
          }
          else {
            se::error("Unsupported custom primitive type: " + primitive_type);
            return MeshHandle{};
          }
        }
        else {
          se::error("Custom primitive type not specified.");
          return MeshHandle{};
        }
      }

      // ----------------------------------------------------------------------
      // Load ordinary triangle meshes
      // Create empty mesh component
      MeshHandle mesh = GFXContext::create_mesh_empty();
      // Load meshes into Runtime resource managers.
      rhi::Device* device = GFXContext::device();
      std::vector<uint32_t> indexBuffer_uint = {};
      std::vector<float> vertexBuffer = {};
      std::vector<float> PositionBuffer = {};
      std::vector<uint64_t> JointIndexBuffer = {};
      std::vector<float> JointweightsBuffer = {};
      // Create GFX mesh, and add it to resource manager
      size_t submesh_index_offset = 0;
      size_t submesh_vertex_offset = 0;
      // For each primitive
      for (auto const& meshPrimitive : gltfmesh.primitives) {
        std::vector<uint32_t> indexArray_uint = {};
        std::vector<float> vertexBuffer_positionOnly = {};
        std::vector<float> vertexBuffer_normalOnly = {};
        std::vector<float> vertexBuffer_uvOnly = {};
        std::vector<float> vertexBuffer_tangentOnly = {};
        std::vector<uint64_t> vertexBuffer_joints = {};
        std::vector<float> vertexBuffer_weights = {};
        auto const& indicesAccessor = model->accessors[meshPrimitive.indices];
        auto const& bufferView = model->bufferViews[indicesAccessor.bufferView];
        auto const& buffer = model->buffers[bufferView.buffer];
        auto const dataAddress =
          buffer.data.data() + bufferView.byteOffset + indicesAccessor.byteOffset;
        auto const byteStride = indicesAccessor.ByteStride(bufferView);
        uint64_t const count = indicesAccessor.count;
        se::vec3 positionMax, positionMin;
        // first, get all indices
        switch (indicesAccessor.componentType) {
        case TINYGLTF_COMPONENT_TYPE_BYTE: {
          ArrayAdapter<char> originIndexArray(dataAddress, count, byteStride);
          for (size_t i = 0; i < count; ++i)
            indexArray_uint.emplace_back(uint32_t(originIndexArray[i]));
        } break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
          ArrayAdapter<unsigned char> originIndexArray(dataAddress, count,
            byteStride);
          for (size_t i = 0; i < count; ++i)
            indexArray_uint.emplace_back(uint32_t(originIndexArray[i]));
        } break;
        case TINYGLTF_COMPONENT_TYPE_SHORT: {
          ArrayAdapter<short> originIndexArray(dataAddress, count, byteStride);
          for (size_t i = 0; i < count; ++i)
            indexArray_uint.emplace_back(uint32_t(originIndexArray[i]));
        } break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
          ArrayAdapter<unsigned short> originIndexArray(dataAddress, count,
            byteStride);
          for (size_t i = 0; i < count; ++i)
            indexArray_uint.emplace_back(uint32_t(originIndexArray[i]));
        } break;
        case TINYGLTF_COMPONENT_TYPE_INT: {
          ArrayAdapter<int> originIndexArray(dataAddress, count, byteStride);
          for (size_t i = 0; i < count; ++i)
            indexArray_uint.emplace_back(uint32_t(originIndexArray[i]));
        } break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
          ArrayAdapter<unsigned int> originIndexArray(dataAddress, count,
            byteStride);
          for (size_t i = 0; i < count; ++i)
            indexArray_uint.emplace_back(uint32_t(originIndexArray[i]));
        } break;
        default:
          break;
        }
        // We re-arrange the indices so that it describe a simple list of
        // triangles
        switch (meshPrimitive.mode) {
          // case TINYGLTF_MODE_TRIANGLE_FAN: // TODO
          // case TINYGLTF_MODE_TRIANGLE_STRIP: // TODO
        case TINYGLTF_MODE_TRIANGLES:  // this is the simpliest case to handle
        {
          for (auto const& attribute : meshPrimitive.attributes) {
            auto const attribAccessor = model->accessors[attribute.second];
            auto const& bufferView =
              model->bufferViews[attribAccessor.bufferView];
            auto const& buffer = model->buffers[bufferView.buffer];
            auto const dataPtr = buffer.data.data() + bufferView.byteOffset +
              attribAccessor.byteOffset;
            auto const byte_stride = attribAccessor.ByteStride(bufferView);
            auto const count = attribAccessor.count;
            if (attribute.first == "POSITION") {
              switch (attribAccessor.type) {
              case TINYGLTF_TYPE_VEC3: {
                positionMax = { (float)attribAccessor.maxValues[0], (float)attribAccessor.maxValues[1], (float)attribAccessor.maxValues[2] };
                positionMin = { (float)attribAccessor.minValues[0], (float)attribAccessor.minValues[1], (float)attribAccessor.minValues[2] };
                switch (attribAccessor.componentType) {
                case TINYGLTF_COMPONENT_TYPE_FLOAT:
                  // 3D vector of float
                  ArrayAdapter<se::vec3> positions(dataPtr, count,
                    byte_stride);
                  for (size_t i = 0; i < count; ++i) {
                    se::vec3 p0 = positions[i];
                    // Put them in the array in the correct order
                    vertexBuffer_positionOnly.push_back(p0.x);
                    vertexBuffer_positionOnly.push_back(p0.y);
                    vertexBuffer_positionOnly.push_back(p0.z);
                  }
                }
                break;
              case TINYGLTF_COMPONENT_TYPE_DOUBLE: {
                switch (attribAccessor.type) {
                case TINYGLTF_TYPE_VEC3: {
                  ArrayAdapter<se::dvec3> positions(dataPtr, count,
                    byte_stride);
                  for (size_t i = 0; i < count; ++i) {
                    se::dvec3 p0 = positions[i];
                    // Put them in the array in the correct order
                    vertexBuffer_positionOnly.push_back(p0.x);
                    vertexBuffer_positionOnly.push_back(p0.y);
                    vertexBuffer_positionOnly.push_back(p0.z);
                  }
                } break;
                default:
                  // TODO Handle error
                  break;
                }
                break;
              default:
                break;
              }
              } break;
              }
            }
            if (attribute.first == "NORMAL") {
              switch (attribAccessor.type) {
              case TINYGLTF_TYPE_VEC3: {
                switch (attribAccessor.componentType) {
                case TINYGLTF_COMPONENT_TYPE_FLOAT: {
                  ArrayAdapter<se::vec3> normals(dataPtr, count,
                    byte_stride);
                  for (size_t i = 0; i < count; ++i) {
                    se::vec3 n0 = normals[i];
                    // Put them in the array in the correct order
                    vertexBuffer_normalOnly.push_back(n0.x);
                    vertexBuffer_normalOnly.push_back(n0.y);
                    vertexBuffer_normalOnly.push_back(n0.z);
                  }
                } break;
                case TINYGLTF_COMPONENT_TYPE_DOUBLE: {
                  ArrayAdapter<se::dvec3> normals(dataPtr, count,
                    byte_stride);
                  for (size_t i = 0; i < count; ++i) {
                    se::dvec3 n0 = normals[i];
                    // Put them in the array in the correct order
                    vertexBuffer_normalOnly.push_back(n0.x);
                    vertexBuffer_normalOnly.push_back(n0.y);
                    vertexBuffer_normalOnly.push_back(n0.z);
                  }
                } break;
                }
              }
              }
            }
            if (attribute.first == "TEXCOORD_0") {
              switch (attribAccessor.type) {
              case TINYGLTF_TYPE_VEC2: {
                switch (attribAccessor.componentType) {
                case TINYGLTF_COMPONENT_TYPE_FLOAT: {
                  ArrayAdapter<se::vec2> uvs(dataPtr, count, byte_stride);
                  for (size_t i = 0; i < count; ++i) {
                    se::vec2 uv0 = uvs[i];
                    // Put them in the array in the correct order
                    vertexBuffer_uvOnly.push_back(uv0.x);
                    vertexBuffer_uvOnly.push_back(uv0.y);
                  }
                } break;
                case TINYGLTF_COMPONENT_TYPE_DOUBLE: {
                  ArrayAdapter<se::dvec2> uvs(dataPtr, count,
                    byte_stride);
                  for (size_t i = 0; i < count; ++i) {
                    se::dvec2 uv0 = uvs[i];
                    // Put them in the array in the correct order
                    vertexBuffer_uvOnly.push_back(uv0.x);
                    vertexBuffer_uvOnly.push_back(uv0.y);
                  }
                } break;
                default:
                  se::error("GFX :: tinygltf :: unrecognized vector type for UV");
                }
              } break;
              default:
                se::error("GFX :: tinygltf :: unreconized componant type for UV");
              }
            }
            if (attribute.first == "TANGENT") {
              switch (attribAccessor.type) {
              case TINYGLTF_TYPE_VEC3: {
                switch (attribAccessor.componentType) {
                case TINYGLTF_COMPONENT_TYPE_FLOAT: {
                  ArrayAdapter<se::vec3> tangents(dataPtr, count,
                    byte_stride);
                  for (size_t i = 0; i < count; ++i) {
                    se::vec3 t0 = tangents[i];
                    // Put them in the array in the correct order
                    vertexBuffer_tangentOnly.push_back(t0.x);
                    vertexBuffer_tangentOnly.push_back(t0.y);
                    vertexBuffer_tangentOnly.push_back(t0.z);
                  }
                } break;
                case TINYGLTF_COMPONENT_TYPE_DOUBLE: {
                  ArrayAdapter<se::dvec3> tangents(dataPtr, count,
                    byte_stride);
                  // IMPORTANT: We need to reorder normals (and texture
                  // coordinates into "facevarying" order) for each face
                  // For each triangle :
                  for (size_t i = 0; i < count; ++i) {
                    se::dvec3 t0 = tangents[i];
                    // Put them in the array in the correct order
                    vertexBuffer_tangentOnly.push_back(t0.x);
                    vertexBuffer_tangentOnly.push_back(t0.y);
                    vertexBuffer_tangentOnly.push_back(t0.z);
                  }
                } break;
                }
              }
              }
            }
            if (attribute.first == "JOINTS_0") {
              switch (attribAccessor.type) {
              case TINYGLTF_TYPE_VEC4: {
                switch (attribAccessor.componentType) {
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                  ArrayAdapter<se::Vector4<uint16_t>> joints(dataPtr, count, byte_stride);
                  for (size_t i = 0; i < count; ++i) {
                    se::Vector4<uint16_t> j0 = joints[i];
                    // Put them in the array in the correct order
                    vertexBuffer_joints.push_back(j0.x);
                    vertexBuffer_joints.push_back(j0.y);
                    vertexBuffer_joints.push_back(j0.z);
                    vertexBuffer_joints.push_back(j0.w);
                  }
                } break;
                }
              }
              }
            }
            if (attribute.first == "WEIGHTS_0") {
              switch (attribAccessor.type) {
              case TINYGLTF_TYPE_VEC4: {
                switch (attribAccessor.componentType) {
                case TINYGLTF_COMPONENT_TYPE_FLOAT: {
                  ArrayAdapter<se::vec4> weights(dataPtr, count, byte_stride);
                  // For each triangle :
                  for (size_t i = 0; i < count; ++i) {
                    se::vec4 w0 = weights[i];
                    // Put them in the array in the correct order
                    vertexBuffer_weights.push_back(w0.x);
                    vertexBuffer_weights.push_back(w0.y);
                    vertexBuffer_weights.push_back(w0.z);
                    vertexBuffer_weights.push_back(w0.w);
                  }
                } break;
                case TINYGLTF_COMPONENT_TYPE_DOUBLE: {
                  ArrayAdapter<se::dvec4> weights(dataPtr, count, byte_stride);
                  // IMPORTANT: We need to reorder normals (and texture
                  // coordinates into "facevarying" order) for each face
                  // For each triangle :
                  for (size_t i = 0; i < count; ++i) {
                    se::dvec4 w0 = weights[i];
                    // Put them in the array in the correct order
                    vertexBuffer_weights.push_back(w0.x);
                    vertexBuffer_weights.push_back(w0.y);
                    vertexBuffer_weights.push_back(w0.z);
                    vertexBuffer_weights.push_back(w0.w);
                  }
                } break;
                }
              }
              }
            }
          }
          break;
        }
        default:
          se::error("GFX :: tinygltf :: primitive mode not implemented");
          break;
        }
        // Compute the tangent vector if no provided
        if (vertexBuffer_tangentOnly.size() == 0) {
        }
        // Assemble vertex buffer
        PositionBuffer.insert(PositionBuffer.end(), vertexBuffer_positionOnly.begin(), vertexBuffer_positionOnly.end());
        indexBuffer_uint.insert(indexBuffer_uint.end(), indexArray_uint.begin(), indexArray_uint.end());
        size_t submehsVertexNumber = vertexBuffer_positionOnly.size() / 3;
        for (size_t i = 0; i < submehsVertexNumber; ++i) {
          for (auto const& entry : defaultMeshLoadConfig.layout.layout) {
            if (entry.info == MeshDataLayout::VertexInfo::POSITION) {}
            else if (entry.info == MeshDataLayout::VertexInfo::NORMAL) {
              if (vertexBuffer_normalOnly.size() == 0) { // if normal is not provided
                vertexBuffer.push_back(0.f);
                vertexBuffer.push_back(0.f);
                vertexBuffer.push_back(0.f);
              }
              else {
                vertexBuffer.push_back(vertexBuffer_normalOnly[i * 3 + 0]);
                vertexBuffer.push_back(vertexBuffer_normalOnly[i * 3 + 1]);
                vertexBuffer.push_back(vertexBuffer_normalOnly[i * 3 + 2]);
              }
            }
            else if (entry.info == MeshDataLayout::VertexInfo::UV) {
              if (vertexBuffer_uvOnly.size() == 0) { // if uv is not provided
                vertexBuffer.push_back(0.f);
                vertexBuffer.push_back(0.f);
              }
              else {
                if (vertexBuffer_uvOnly[i * 2 + 0] > 1) vertexBuffer_uvOnly[i * 2 + 0] -= int(vertexBuffer_uvOnly[i * 2 + 0]);
                if (vertexBuffer_uvOnly[i * 2 + 1] > 1) vertexBuffer_uvOnly[i * 2 + 1] -= int(vertexBuffer_uvOnly[i * 2 + 1]);
                vertexBuffer.push_back(vertexBuffer_uvOnly[i * 2 + 0]);
                vertexBuffer.push_back(vertexBuffer_uvOnly[i * 2 + 1]);
              }
            }
            else if (entry.info == MeshDataLayout::VertexInfo::TANGENT) {
              vertexBuffer.push_back(0.f);
              vertexBuffer.push_back(0.f);
              vertexBuffer.push_back(0.f);
              //vertexBuffer.push_back(vertexBuffer_tangentOnly[i * 3 + 0]);
              //vertexBuffer.push_back(vertexBuffer_tangentOnly[i * 3 + 1]);
              //vertexBuffer.push_back(vertexBuffer_tangentOnly[i * 3 + 2]);
            }
            else if (entry.info == MeshDataLayout::VertexInfo::COLOR) {
              // Optional: vertex colors
              vertexBuffer.push_back(0);
              vertexBuffer.push_back(0);
              vertexBuffer.push_back(0);
            }
            else if (entry.info == MeshDataLayout::VertexInfo::CUSTOM) {}
          }
        }

        // Assemble skin buffer
        if (vertexBuffer_joints.size() != 0) {
          JointIndexBuffer.insert(JointIndexBuffer.end(), vertexBuffer_joints.begin(), vertexBuffer_joints.end());
          JointweightsBuffer.insert(JointweightsBuffer.end(), vertexBuffer_weights.begin(), vertexBuffer_weights.end());
        }

        // load Material
        Mesh::MeshPrimitive sePrimitive;
        sePrimitive.offset = submesh_index_offset;
        sePrimitive.size = indexArray_uint.size();
        sePrimitive.baseVertex = submesh_vertex_offset;
        sePrimitive.numVertex = PositionBuffer.size() / 3 - submesh_vertex_offset;
        sePrimitive.max = positionMax;
        sePrimitive.min = positionMin;
        if (meshPrimitive.material != -1) {
          auto const& gltf_material = model->materials[meshPrimitive.material];
          sePrimitive.material = loadGLTFMaterial(&gltf_material, model, env, scene, defaultMeshLoadConfig);
        }
        else {
          sePrimitive.material = GFXContext::create_material_empty();
        }
        tinygltf::Value primitive_extra = meshPrimitive.extras;
        if (primitive_extra.Has("exterior")) {
          int exterior_index = primitive_extra.Get("exterior").GetNumberAsInt();
          if (exterior_index >= 0) sePrimitive.exterior = env.mediums[exterior_index];
        }
        if (primitive_extra.Has("interior")) {
          int interior_index = primitive_extra.Get("interior").GetNumberAsInt();
          if (interior_index >= 0) sePrimitive.interior = env.mediums[interior_index];
        }

        mesh->m_primitives.emplace_back(std::move(sePrimitive));
        // todo:: add material
        submesh_index_offset = indexBuffer_uint.size();
        submesh_vertex_offset = PositionBuffer.size() / 3;
      }
      // create mesh resource
      { // register mesh
        MiniBuffer buffer;
        buffer.m_isReference = true;

        buffer.m_data = PositionBuffer.data();
        buffer.m_size = sizeof(float) * PositionBuffer.size();

        bool need_rt = bool(gfx::GFXContext::device()->from_which_adapter()->from_which_context()
          ->get_context_extensions_flags() & rhi::ContextExtensionEnum::RAY_TRACING);
        Flags<rhi::BufferUsageEnum> rt_usage = 0;
        if (need_rt) rt_usage |= rhi::BufferUsageEnum::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY;

        mesh->m_positionBuffer = gfx::GFXContext::create_buffer_host(buffer,
          rhi::BufferUsageEnum::STORAGE |
          rhi::BufferUsageEnum::SHADER_DEVICE_ADDRESS | rt_usage);
        mesh->m_positionBuffer->m_job = "Mesh position buffer";
        mesh->m_positionBuffer->m_host.resize(buffer.m_size);
        memcpy(mesh->m_positionBuffer->m_host.data(), buffer.m_data, buffer.m_size);

        buffer.m_data = indexBuffer_uint.data();
        buffer.m_size = sizeof(uint32_t) * indexBuffer_uint.size();
        mesh->m_indexBuffer = gfx::GFXContext::create_buffer_host(buffer,
          rhi::BufferUsageEnum::INDEX |
          rhi::BufferUsageEnum::SHADER_DEVICE_ADDRESS | rt_usage);
        mesh->m_indexBuffer->m_job = "Mesh index buffer";
        mesh->m_indexBuffer->m_host.resize(buffer.m_size);
        memcpy(mesh->m_indexBuffer->m_host.data(), buffer.m_data, buffer.m_size);

        buffer.m_data = vertexBuffer.data();
        buffer.m_size = sizeof(uint32_t) * vertexBuffer.size();
        mesh->m_vertexBuffer = gfx::GFXContext::create_buffer_host(buffer,
          rhi::BufferUsageEnum::STORAGE |
          rhi::BufferUsageEnum::SHADER_DEVICE_ADDRESS);
        mesh->m_vertexBuffer->m_job = "Mesh vertex buffer";
        mesh->m_vertexBuffer->m_host.resize(buffer.m_size);
        memcpy(mesh->m_vertexBuffer->m_host.data(), buffer.m_data, buffer.m_size);
      }
      return mesh;
    }

    static bool ends_with(const std::string& str, const std::string& suffix) {
        return str.size() >= suffix.size() && str.compare(str.size()-suffix.size(), suffix.size(), suffix) == 0;
    }

    auto Scene::load_gltf(std::string const& path) noexcept -> void {
      tinygltf::TinyGLTF loader;
      tinygltf::Model model;
      std::string err;
      std::string warn;

      // if is binary glTF
      if (ends_with(path, ".glb")) {
        bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, path);
        if (!warn.empty()) {
          se::error("Scene::deserialize warn::" + warn); return;
        } if (!err.empty()) {
          se::error("Scene::deserialize error::" + err); return;
        } if (!ret) {
          se::error("Failed to parse binary glTF"); return;
        }
      }
      else {
        bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, path);
        if (!warn.empty()) {
          se::error("Scene::deserialize warn::" + warn); return;
        } if (!err.empty()) {
          se::error("Scene::deserialize error::" + err); return;
        } if (!ret) {
          se::error("Failed to parse glTF"); return;
        }
      }

      glTFLoaderEnv env;
      env.directory = Filesys::get_parent_path(path);

      gfx::DeserializeData deserialize;
      deserialize.model = &model;
      deserialize.nodes.resize(model.nodes.size());

      for (int i = 0; i < model.textures.size(); ++i) {
        tinygltf::Texture const& texture_gltf = model.textures[i];
        if (texture_gltf.extras.Has("dparam")) {
          //tinygltf::Value dparam = texture_gltf.extras.Get("dparam");
          //rhi::TextureDescriptor desc;
          //desc.size.width = dparam.Get("dim_0").GetNumberAsInt();
          //desc.size.height = dparam.Get("dim_1").GetNumberAsInt();
          //desc.arrayLayerCount = dparam.Get("dim_2").GetNumberAsInt();
          //float default_value = float(dparam.Get("default_value").GetNumberAsDouble());
          //int replica_num = float(dparam.Get("dim_replica").GetNumberAsInt());
          //int num_aux = float(dparam.Get("num_aux").GetNumberAsInt());
          //TextureHandle texture = gfx::GFXContext::create_buf_texture_desc(desc, default_value, num_aux, replica_num);
          //env.textures[&texture_gltf] = texture;
        }
        else {
          auto const& image_gltf = model.images[texture_gltf.source];
          if (image_gltf.image.size() > 0) {
            TextureHandle texture = gfx::GFXContext::load_texture_binary(
              image_gltf.width, image_gltf.height, image_gltf.component, 1, (const char*)image_gltf.image.data());
            texture->m_resourcePath = image_gltf.uri;
            env.textures[&texture_gltf] = texture;
          }
          else {
            std::string file_path = env.directory + "/" + image_gltf.uri;
            TextureHandle texture = gfx::GFXContext::load_texture_file(file_path);
            env.textures[&texture_gltf] = texture;
          }
        }
      }

      // load medium if any
      if (model.extras.Has("medium")) {
        tinygltf::Value medium_extra = model.extras.Get("medium");
        int buffer_index = medium_extra.Get("buffer_id").GetNumberAsInt();
        tinygltf::Value medium_instances = medium_extra.Get("mediums");
        env.mediums.resize(medium_instances.ArrayLen());
        tcb::span<float> medium_buffer;
        if (buffer_index >= 0) {
          medium_buffer = tcb::span<float>{
            (float*)model.buffers[buffer_index].data.data(),
            model.buffers[buffer_index].data.size() / 4
          };
        }
        for (int medium_index = 0; medium_index < medium_instances.ArrayLen(); ++medium_index) {
          MediumHandle medium = GFXContext::create_medium_empty();
          tinygltf::Value instance = medium_instances.Get(medium_index);
          medium->packet.aniso = {
            (float)instance.Get("aniso_x").GetNumberAsDouble(),
            (float)instance.Get("aniso_y").GetNumberAsDouble(),
            (float)instance.Get("aniso_z").GetNumberAsDouble(),
          };
          medium->packet.boundMin = {
            (float)instance.Get("bound_min_x").GetNumberAsDouble(),
            (float)instance.Get("bound_min_y").GetNumberAsDouble(),
            (float)instance.Get("bound_min_z").GetNumberAsDouble(),
          };
          medium->packet.boundMax = {
            (float)instance.Get("bound_max_x").GetNumberAsDouble(),
            (float)instance.Get("bound_max_y").GetNumberAsDouble(),
            (float)instance.Get("bound_max_z").GetNumberAsDouble(),
          };
          medium->packet.sigmaA = {
            (float)instance.Get("sigma_a_x").GetNumberAsDouble(),
            (float)instance.Get("sigma_a_y").GetNumberAsDouble(),
            (float)instance.Get("sigma_a_z").GetNumberAsDouble(),
          };
          medium->packet.sigmaS = {
            (float)instance.Get("sigma_s_x").GetNumberAsDouble(),
            (float)instance.Get("sigma_s_y").GetNumberAsDouble(),
            (float)instance.Get("sigma_s_z").GetNumberAsDouble(),
          };
          ivec3 const grid_nxyz = {
            instance.Get("grid_nx").GetNumberAsInt(),
            instance.Get("grid_ny").GetNumberAsInt(),
            instance.Get("grid_nz").GetNumberAsInt(),
          };
          medium->packet.scale = (float)instance.Get("scale").GetNumberAsDouble();

          if (instance.Get("type").GetNumberAsInt() == 0) {
            medium->packet.type = Medium::MediumType::Homogeneous;
          }
          else if (instance.Get("type").GetNumberAsInt() == 1) {
            medium->packet.type = Medium::MediumType::GridMedium;
            tinygltf::Value o2w = instance.Get("o2w");
            medium->packet.geometryTransform.matrix[0][0] = (float)o2w.Get(0).GetNumberAsDouble();
            medium->packet.geometryTransform.matrix[0][1] = (float)o2w.Get(1).GetNumberAsDouble();
            medium->packet.geometryTransform.matrix[0][2] = (float)o2w.Get(2).GetNumberAsDouble();
            medium->packet.geometryTransform.matrix[0][3] = (float)o2w.Get(3).GetNumberAsDouble();
            medium->packet.geometryTransform.matrix[1][0] = (float)o2w.Get(4).GetNumberAsDouble();
            medium->packet.geometryTransform.matrix[1][1] = (float)o2w.Get(5).GetNumberAsDouble();
            medium->packet.geometryTransform.matrix[1][2] = (float)o2w.Get(6).GetNumberAsDouble();
            medium->packet.geometryTransform.matrix[1][3] = (float)o2w.Get(7).GetNumberAsDouble();
            medium->packet.geometryTransform.matrix[2][0] = (float)o2w.Get(8).GetNumberAsDouble();
            medium->packet.geometryTransform.matrix[2][1] = (float)o2w.Get(9).GetNumberAsDouble();
            medium->packet.geometryTransform.matrix[2][2] = (float)o2w.Get(10).GetNumberAsDouble();
            medium->packet.geometryTransform.matrix[2][3] = (float)o2w.Get(11).GetNumberAsDouble();
            tinygltf::Value w2o = instance.Get("w2o");
            medium->packet.geometryTransformInverse.matrix[0][0] = (float)w2o.Get(0).GetNumberAsDouble();
            medium->packet.geometryTransformInverse.matrix[0][1] = (float)w2o.Get(1).GetNumberAsDouble();
            medium->packet.geometryTransformInverse.matrix[0][2] = (float)w2o.Get(2).GetNumberAsDouble();
            medium->packet.geometryTransformInverse.matrix[0][3] = (float)w2o.Get(3).GetNumberAsDouble();
            medium->packet.geometryTransformInverse.matrix[1][0] = (float)w2o.Get(4).GetNumberAsDouble();
            medium->packet.geometryTransformInverse.matrix[1][1] = (float)w2o.Get(5).GetNumberAsDouble();
            medium->packet.geometryTransformInverse.matrix[1][2] = (float)w2o.Get(6).GetNumberAsDouble();
            medium->packet.geometryTransformInverse.matrix[1][3] = (float)w2o.Get(7).GetNumberAsDouble();
            medium->packet.geometryTransformInverse.matrix[2][0] = (float)w2o.Get(8).GetNumberAsDouble();
            medium->packet.geometryTransformInverse.matrix[2][1] = (float)w2o.Get(9).GetNumberAsDouble();
            medium->packet.geometryTransformInverse.matrix[2][2] = (float)w2o.Get(10).GetNumberAsDouble();
            medium->packet.geometryTransformInverse.matrix[2][3] = (float)w2o.Get(11).GetNumberAsDouble();

            bounds3 bounds = { medium->packet.boundMin,medium->packet.boundMax };

            int density_offset = instance.Get("density_offset").GetNumberAsInt();
            int density_size = instance.Get("density_size").GetNumberAsInt();
            medium->density = Medium::SampledGrid{
              grid_nxyz.x, grid_nxyz.y, grid_nxyz.z,
              std::vector<float>{medium_buffer.begin() + density_offset, medium_buffer.begin() + density_offset + density_size},
              bounds, 1
            };

            int temperature_offset = instance.Get("temperature_offset").GetNumberAsInt();
            int temperature_size = instance.Get("temperature_size").GetNumberAsInt();
            medium->temperatureGrid = Medium::SampledGrid{
              grid_nxyz.x, grid_nxyz.y, grid_nxyz.z,
              std::vector<float>{medium_buffer.begin() + temperature_offset, medium_buffer.begin() + temperature_offset + temperature_size},
              bounds, 1
            };

            // create majorant grid
            medium->majorantGrid = Medium::MajorantGrid();
            medium->majorantGrid->res = ivec3(16, 16, 16);
            medium->majorantGrid->bounds = { medium->packet.boundMin, medium->packet.boundMax };
            medium->majorantGrid->voxels.resize(16 * 16 * 16);
            // Initialize _majorantGrid_ for _GridMedium_
            for (int z = 0; z < medium->majorantGrid->res.z; ++z)
              for (int y = 0; y < medium->majorantGrid->res.y; ++y)
                for (int x = 0; x < medium->majorantGrid->res.x; ++x) {
                  bounds3 bounds = medium->majorantGrid->voxel_bounds(x, y, z);
                  medium->majorantGrid->set(x, y, z, medium->density->max_value(bounds));
                }
          }
          else if (instance.Get("type").GetNumberAsInt() == 2) {
            medium->packet.type = Medium::MediumType::RGBGridMedium;
            tinygltf::Value o2w = instance.Get("o2w");
            medium->packet.geometryTransform.matrix[0][0] = (float)o2w.Get(0).GetNumberAsDouble();
            medium->packet.geometryTransform.matrix[0][1] = (float)o2w.Get(1).GetNumberAsDouble();
            medium->packet.geometryTransform.matrix[0][2] = (float)o2w.Get(2).GetNumberAsDouble();
            medium->packet.geometryTransform.matrix[0][3] = (float)o2w.Get(3).GetNumberAsDouble();
            medium->packet.geometryTransform.matrix[1][0] = (float)o2w.Get(4).GetNumberAsDouble();
            medium->packet.geometryTransform.matrix[1][1] = (float)o2w.Get(5).GetNumberAsDouble();
            medium->packet.geometryTransform.matrix[1][2] = (float)o2w.Get(6).GetNumberAsDouble();
            medium->packet.geometryTransform.matrix[1][3] = (float)o2w.Get(7).GetNumberAsDouble();
            medium->packet.geometryTransform.matrix[2][0] = (float)o2w.Get(8).GetNumberAsDouble();
            medium->packet.geometryTransform.matrix[2][1] = (float)o2w.Get(9).GetNumberAsDouble();
            medium->packet.geometryTransform.matrix[2][2] = (float)o2w.Get(10).GetNumberAsDouble();
            medium->packet.geometryTransform.matrix[2][3] = (float)o2w.Get(11).GetNumberAsDouble();
            tinygltf::Value w2o = instance.Get("w2o");
            medium->packet.geometryTransformInverse.matrix[0][0] = (float)w2o.Get(0).GetNumberAsDouble();
            medium->packet.geometryTransformInverse.matrix[0][1] = (float)w2o.Get(1).GetNumberAsDouble();
            medium->packet.geometryTransformInverse.matrix[0][2] = (float)w2o.Get(2).GetNumberAsDouble();
            medium->packet.geometryTransformInverse.matrix[0][3] = (float)w2o.Get(3).GetNumberAsDouble();
            medium->packet.geometryTransformInverse.matrix[1][0] = (float)w2o.Get(4).GetNumberAsDouble();
            medium->packet.geometryTransformInverse.matrix[1][1] = (float)w2o.Get(5).GetNumberAsDouble();
            medium->packet.geometryTransformInverse.matrix[1][2] = (float)w2o.Get(6).GetNumberAsDouble();
            medium->packet.geometryTransformInverse.matrix[1][3] = (float)w2o.Get(7).GetNumberAsDouble();
            medium->packet.geometryTransformInverse.matrix[2][0] = (float)w2o.Get(8).GetNumberAsDouble();
            medium->packet.geometryTransformInverse.matrix[2][1] = (float)w2o.Get(9).GetNumberAsDouble();
            medium->packet.geometryTransformInverse.matrix[2][2] = (float)w2o.Get(10).GetNumberAsDouble();
            medium->packet.geometryTransformInverse.matrix[2][3] = (float)w2o.Get(11).GetNumberAsDouble();

            bounds3 bounds = { medium->packet.boundMin,medium->packet.boundMax };

            int sigma_a_offset = instance.Get("sigma_a_offset").GetNumberAsInt();
            int sigma_a_size = instance.Get("sigma_a_size").GetNumberAsInt();
            medium->density = Medium::SampledGrid{
              grid_nxyz.x, grid_nxyz.y, grid_nxyz.z,
              std::vector<float>{medium_buffer.begin() + sigma_a_offset, medium_buffer.begin() + sigma_a_offset + sigma_a_size},
              bounds, 3
            };

            int sigma_s_offset = instance.Get("sigma_s_offset").GetNumberAsInt();
            int sigma_s_size = instance.Get("sigma_s_size").GetNumberAsInt();
            medium->temperatureGrid = Medium::SampledGrid{
              grid_nxyz.x, grid_nxyz.y, grid_nxyz.z,
              std::vector<float>{medium_buffer.begin() + sigma_s_offset, medium_buffer.begin() + sigma_s_offset + sigma_s_size},
              bounds, 3
            };

            // create majorant grid
            medium->majorantGrid = Medium::MajorantGrid();
            medium->majorantGrid->res = ivec3(16, 16, 16);
            medium->majorantGrid->bounds = { medium->packet.boundMin, medium->packet.boundMax };
            medium->majorantGrid->voxels.resize(16 * 16 * 16);
            // Initialize _majorantGrid_ for _GridMedium_
            for (int z = 0; z < medium->majorantGrid->res.z; ++z)
              for (int y = 0; y < medium->majorantGrid->res.y; ++y)
                for (int x = 0; x < medium->majorantGrid->res.x; ++x) {
                  bounds3 bounds = medium->majorantGrid->voxel_bounds(x, y, z);
                  float maximum = (medium->density->max_value(bounds) + medium->temperatureGrid->max_value(bounds)) * medium->packet.scale;
                  medium->majorantGrid->set(x, y, z, maximum);
                }
          }

          env.mediums[medium_index] = medium;
        }
      }

      // register all the nodes first
      for (size_t i = 0; i < model.nodes.size(); ++i) {
        deserialize.nodes[i] = create_node(model.nodes[i].name);
      }
      // add the hierarchy information
      for (size_t i = 0; i < model.nodes.size(); ++i) {
        auto& children = m_registry.get<NodeProperty>(deserialize.nodes[i].m_entity).children;
        for (auto& child_id : model.nodes[i].children) {
          children.push_back(deserialize.nodes[child_id]);
        }
      }
      // register the scene root
      auto& scene = model.scenes[0];
      for (auto root : scene.nodes) {
        m_roots.emplace_back(deserialize.nodes[root]);
      }

      // load tag, transform, mesh
      for (size_t i = 0; i < model.nodes.size(); ++i) {
        auto const& gltfNode = model.nodes[i];
        auto& seNode = deserialize.nodes[i];
        // process the transform
        {
          auto& transform = m_registry.get<gfx::Transform>(seNode.m_entity);
          if (gltfNode.scale.size() == 3)
            transform.scale = { static_cast<float>(gltfNode.scale[0]),
              static_cast<float>(gltfNode.scale[1]), static_cast<float>(gltfNode.scale[2]) };
          if (gltfNode.translation.size() == 3)
            transform.translation = { static_cast<float>(gltfNode.translation[0]),
              static_cast<float>(gltfNode.translation[1]), static_cast<float>(gltfNode.translation[2]) };
          if (gltfNode.rotation.size() == 4) {
            transform.rotation = { float(gltfNode.rotation[0]), float(gltfNode.rotation[1]),
              float(gltfNode.rotation[2]), float(gltfNode.rotation[3]) };
          }
          if (gltfNode.matrix.size() == 16) {
            se::mat4 mat = se::mat4{
              (float)gltfNode.matrix[0], (float)gltfNode.matrix[1], (float)gltfNode.matrix[2],  (float)gltfNode.matrix[3],
              (float)gltfNode.matrix[4], (float)gltfNode.matrix[5], (float)gltfNode.matrix[6],  (float)gltfNode.matrix[7],
              (float)gltfNode.matrix[8], (float)gltfNode.matrix[9], (float)gltfNode.matrix[10], (float)gltfNode.matrix[11],
              (float)gltfNode.matrix[12],(float)gltfNode.matrix[13],(float)gltfNode.matrix[14], (float)gltfNode.matrix[15],
            };
            mat = se::transpose(mat);
            se::vec3 t, s; se::Quaternion quat;
            se::decompose(mat, &t, &quat, &s);
            transform.translation = t;
            transform.rotation = { quat.x, quat.y, quat.z, quat.w };
            transform.scale = s;
            transform.m_dirtyToFile = false;
            transform.m_dirtyToGPU = true;
          }
        }
        // process the mesh
        if (gltfNode.mesh != -1) {
          tinygltf::Mesh& mesh_gltf = model.meshes[gltfNode.mesh];
          MeshHandle mesh = loadGLTFMesh(mesh_gltf, seNode, *this, i, &model, env);
          auto& mesh_renderer = m_registry.emplace<MeshRenderer>(seNode.m_entity);
          mesh_renderer.m_mesh = mesh;
          mesh_renderer.m_dirtyToFile = false;
          mesh_renderer.m_dirtyToGPU = true;

          std::vector<int> emissive_primitives;
          for (int i = 0; i < mesh->m_primitives.size(); ++i) {
            if (mesh->m_primitives[i].material.get()) {
              MaterialHandle& mat = mesh->m_primitives[i].material;
              if (mat->m_packet.vec4Data1.r > 0 ||
                mat->m_packet.vec4Data1.g > 0 ||
                mat->m_packet.vec4Data1.b > 0) {
                emissive_primitives.push_back(i);
              }
            }
          }
          if (emissive_primitives.size() > 0) {
            auto& light = seNode.add_component<Light>();
            //light.primitives = emissive_primitives;
            light.light.light_type = LightTypeEnum::MESH_PRIMITIVE;
          }
        }
        // process the camera
        if (gltfNode.camera != -1) {
          auto& camera = m_registry.emplace<Camera>(seNode.m_entity);
          camera.m_dirtyToFile = false;
          camera.m_dirtyToGPU = false;
          auto& gltf_camera = model.cameras[gltfNode.camera];
          if (gltf_camera.type == "perspective") {
            camera.zfar = gltf_camera.perspective.zfar;
            camera.znear = gltf_camera.perspective.znear;
            camera.yfov = gltf_camera.perspective.yfov * 180 / se::M_FLOAT_PI;
            camera.aspectRatio = gltf_camera.perspective.aspectRatio;
          }
        }
      }

      // register all the nodes first
      for (auto& iter : Singleton<ComponentManager>::instance()->m_components) {
        iter.second.deserialize(deserialize);
      }
    }

    auto Scene::save(std::string const& path) noexcept -> void {
      tinygltf::Model m;
      tinygltf::Scene scene;
      tinygltf::Value::Object model_extra;

      SerializeData data;
      data.model = &m;
      data.gfx_scene = this;

      // register all the nodes first
      for (auto& iter : Singleton<ComponentManager>::instance()->m_components) {
        iter.second.serialize(data);
      }
      // Write root nodes to scene
      scene.nodes.reserve(m_roots.size());
      for (auto& node : m_roots)
        scene.nodes.emplace_back(data.nodes[node.m_entity]);

      m.extras = tinygltf::Value(model_extra);
      m.scenes.emplace_back(scene);
      tinygltf::TinyGLTF gltf;
      gltf.WriteGltfSceneToFile(&m, path,
        false,  // embedImages
        true,   // embedBuffers
        true,   // pretty print
        false); // write binary
    }

    auto SceneBatch::emplace_scene(SceneHandle scene) noexcept -> void {
      m_scenes.emplace_back(scene);
    }

    SceneLoader::result_type SceneLoader::operator()(SceneLoader::from_gltf_tag, std::string const& path) {
      SceneLoader::result_type scene = std::make_shared<Scene>();
      scene->load_gltf(path);
      return scene;
    }

    SceneLoader::result_type SceneLoader::operator()(SceneLoader::from_xml_tag, std::string const& path) {
      SceneLoader::result_type scene = std::make_shared<Scene>();
      scene->load_xml(path);
      return scene;
    }

    SceneLoader::result_type SceneLoader::operator()(SceneLoader::from_pbrt_tag, std::string const& path) {
      SceneLoader::result_type scene = std::make_shared<Scene>();
      scene->load_pbrt(path);
      return scene;
    }

    auto GFXContext::load_scene_gltf(std::string const& _path) noexcept -> SceneHandle {
      std::string path = Filesys::resolve_path(_path, {
        Configuration::string_property("engine_path"),
        Configuration::string_property("project_path"), });
      std::string name = Filesys::get_stem(path);
      UID const ruid = se::Resources::query_string_uid(path);
      auto ret = Singleton<GFXContext>::instance()->m_scenes.load(ruid, SceneLoader::from_gltf_tag{}, path);
      ret.first->second->m_name = name;
      ret.first->second->m_filepath = path;
      return SceneHandle{ ret.first->second };
    }
    
    auto GFXContext::load_scene_xml(std::string const& _path) noexcept -> SceneHandle {
      PROFILE_SCOPE_FUNCTION();
      std::string path = Filesys::resolve_path(_path, {
        Configuration::string_property("engine_path"),
        Configuration::string_property("project_path"), });
      std::string name = Filesys::get_stem(path);
      UID const ruid = se::Resources::query_string_uid(path);
      auto ret = Singleton<GFXContext>::instance()->m_scenes.load(ruid, SceneLoader::from_xml_tag{}, path);
      ret.first->second->m_name = name;
      ret.first->second->m_filepath = path;
      return SceneHandle{ ret.first->second };
    }

    auto GFXContext::load_scene_pbrt(std::string const& _path) noexcept -> SceneHandle {
      PROFILE_SCOPE_FUNCTION();
      std::string path = Filesys::resolve_path(_path, {
        Configuration::string_property("engine_path"),
        Configuration::string_property("project_path"), });
      std::string name = Filesys::get_stem(path);
      UID const ruid = se::Resources::query_string_uid(path);
      auto ret = Singleton<GFXContext>::instance()->m_scenes.load(ruid, SceneLoader::from_pbrt_tag{}, path);
      ret.first->second->m_name = name;
      ret.first->second->m_filepath = path;
      return SceneHandle{ ret.first->second };
    }
    
    auto GFXContext::create_scene_batch() noexcept -> SceneBatchHandle {
      SceneBatchHandle handle = std::make_shared<SceneBatch>();
      return handle;
    }

    auto GFXContext::frame_end() noexcept -> void {
      se::gfx::GFXContext::get_flights()->frame_end();
      se::gfx::GFXContext::clean_cache();

      auto& jobs = Singleton<GFXContext>::instance()->m_jobsFrameEnd;
      while (jobs.size() > 0) {
        auto job = jobs.front();
        job();
        jobs.pop_front();
      }
    }

    auto SerializeData::add_buffer(
      std::vector<std::byte> const& data,
      std::string const& name) noexcept -> int32_t {
      tinygltf::Buffer buffer;
      buffer.name = name;
      buffer.data.resize(data.size());
      memcpy(buffer.data.data(), data.data(), data.size());
      int32_t buffer_idx = model->buffers.size();
      model->buffers.emplace_back(buffer);
      return buffer_idx;
    }

    auto SerializeData::add_view_accessor(
      tinygltf::BufferView bufferView,
      tinygltf::Accessor accessor
    ) noexcept -> int32_t {
      int view_id = model->bufferViews.size();
      model->bufferViews.push_back(bufferView);
      int accessor_id = model->accessors.size();
      accessor.bufferView = view_id;
      model->accessors.push_back(accessor);
      return accessor_id;
    }

    auto SerializeData::add_accessor(
      tinygltf::Accessor accessor) noexcept -> int32_t {
      int accessor_id = model->accessors.size();
      model->accessors.push_back(accessor);
      return accessor_id;
    }

    auto SerializeData::add_material(Material* material) noexcept -> int32_t {
      auto iter = m_materials.find(material);
      if (iter == m_materials.end()) {
        int32_t index = m_materials.size();;
        m_materials[material] = index;

        tinygltf::Material gltf_material;
        gltf_material.pbrMetallicRoughness.baseColorFactor = {
            material->m_packet.vec4Data0.r, material->m_packet.vec4Data0.g,
            material->m_packet.vec4Data0.b, 1. };
        gltf_material.pbrMetallicRoughness.roughnessFactor = double(material->m_packet.vec4Data0.w);
        gltf_material.pbrMetallicRoughness.metallicFactor = double(material->m_packet.vec4Data1.w);
        gltf_material.emissiveFactor = {
            material->m_packet.vec4Data1.r, material->m_packet.vec4Data1.g,
            material->m_packet.vec4Data1.b
        };

        tinygltf::Value::Object material_extra;
        material_extra["bxdf"] = tinygltf::Value(material->m_packet.bxdfType);
        if (material->m_customString != "")
          material_extra["custom_string"] = tinygltf::Value(material->m_customString);

        //// check additionalTex
        //if (material->basecolorTex.get() != nullptr) {
        //  gltf_material.pbrMetallicRoughness.baseColorTexture.index = add_texture(material->basecolorTex.get());
        //}
        //// check additionalTex
        //if (material->additionalTex.get() != nullptr) {
        //  material_extra["additional_tex"] = tinygltf::Value(add_texture(material->additionalTex.get()));
        //}
        material_extra["ext_vector_2"] = tinygltf::Value(tinygltf::Value::Array{
          tinygltf::Value(material->m_packet.vec4Data2.x),
          tinygltf::Value(material->m_packet.vec4Data2.y),
          tinygltf::Value(material->m_packet.vec4Data2.z),
          tinygltf::Value(material->m_packet.vec4Data2.w),
          });

        gltf_material.name = material->m_name;
        gltf_material.extras = tinygltf::Value(material_extra);

        model->materials.push_back(gltf_material);
        return index;
      }
      else return iter->second;
    }

  }
}