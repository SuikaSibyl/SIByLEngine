#include "se.gfx.hpp"
#include "se.editor.hpp"
#include <imgui.h>

namespace se {
namespace gfx {
  auto Scene::update_scripts() noexcept -> void {
    m_timer.update();
    double const deltaTime = m_timer.delta_time();
    auto node_view = m_registry.view<Script>();
    for (auto [entity, _script] : node_view.each()) {
      Node node = { entity, &m_registry };
      _script.update(node, deltaTime);
    }
  }

  auto update_node_transform(Node& node, se::mat4 const& mat, bool in_dirty) noexcept->void {
    auto* _property = node.get_component<NodeProperty>();
    auto* _transform = node.get_component<Transform>();
    _transform->global = mat * _transform->local();
    bool dirty = in_dirty || _transform->is_dirty_to_gpu();
    if (dirty && _property->name != "Camera")
      _transform->m_dirtyToGPU = true;
    for (auto& child : _property->children) {
      update_node_transform(child, _transform->global, dirty);
    }
  }

  auto Scene::update_transform() noexcept -> void {
    se::mat4 identity;
    for (auto& node : m_roots) {
      update_node_transform(node, identity, false);
    }
  }

  auto Scene::update_gpu_scene() noexcept -> void {
    update_transform();
    update_gpu_meshes();
    update_gpu_camera();
    update_gpu_lights();
    update_gpu_medium();
    update_gpu_bvh();

    auto node_view = m_registry.view<Transform>();
    for (auto [entity, _transform] : node_view.each()) {
      _transform.m_dirtyToGPU = false;
    }

    m_gpuScene.geometryBuffer.m_buffer->host_to_device();
  }

  auto Scene::update_gpu_meshes() noexcept -> void {
    auto node_view = m_registry.view<Transform, MeshRenderer>();
    for (auto [entity, transform, mesh] : node_view.each()) {
      // If the mesh resource itself is dirty, we update the reference to the mesh
      if (mesh.m_mesh->m_dirtyToGPU) {
        uint64_t vertex_address = mesh.m_mesh->m_vertexBuffer->m_buffer->get_device_address();
        uint64_t pos_address = mesh.m_mesh->m_positionBuffer->m_buffer->get_device_address();
        uint64_t index_address = mesh.m_mesh->m_indexBuffer->m_buffer->get_device_address();

        auto iter = m_gpuScene.meshList.find(mesh.m_mesh.get());
        if (iter == m_gpuScene.meshList.end()) {
          int32_t index = m_gpuScene.positionBuffer.insert(pos_address);
          m_gpuScene.vertexBuffer.insert(vertex_address);
          m_gpuScene.indexBuffer.insert(index_address);
          m_gpuScene.meshList[mesh.m_mesh.get()] = IndexInfo{ index, 0 };
        }
        else {
          se::error("todo :: a mesh is dirty after first register");
        }
        mesh.m_mesh->m_dirtyToGPU = false;
      }

      // The mesh get a uniform ID in the mesh-list
      int16_t meshID = (int16_t)m_gpuScene.meshList[mesh.m_mesh.get()].assignedIndex;

      // After we have the mesh resource ready,
      // we take care of the material resource
      for (auto& primitive : mesh.m_mesh->m_customPrimitives) {
        MaterialHandle mat = primitive.material;
        auto iter = m_gpuScene.materialList.find(mat.get());
        if (iter == m_gpuScene.materialList.end()) {
          MaterialInterpreterManager::init(mat.get(), mat->m_packet.bxdfType);
          int32_t index = m_gpuScene.materialBuffer.insert(mat->m_packet);
          m_gpuScene.materialList[mat.get()] = IndexInfo{ index, 0 };
          mat->m_dirtyToGPU = false;
        }
        else if (mat->m_dirtyToGPU == true) {
          MaterialInterpreterManager::init(mat.get(), mat->m_packet.bxdfType);
          m_gpuScene.materialBuffer.update(iter->second.assignedIndex, mat->m_packet);
          mat->m_dirtyToGPU = false;
        }
      }
      for (auto& primitive : mesh.m_mesh->m_primitives) {
        MaterialHandle mat = primitive.material;
        if (!mat.get()) continue;
        auto iter = m_gpuScene.materialList.find(mat.get());
        if (iter == m_gpuScene.materialList.end()) {
          MaterialInterpreterManager::init(mat.get(), mat->m_packet.bxdfType);
          if (mat->m_basecolorTex.get()) {
            mat->m_packet.baseTex = m_gpuScene.imagePool.try_fetch_index(mat->m_basecolorTex);
          }
          else mat->m_packet.baseTex = -1;

          int32_t index = m_gpuScene.materialBuffer.insert(mat->m_packet);
          m_gpuScene.materialList[mat.get()] = IndexInfo{ index, 0 };
          mat->m_dirtyToGPU = false;
        }
        else if (mat->m_dirtyToGPU == true) {
          MaterialInterpreterManager::init(mat.get(), mat->m_packet.bxdfType);
          m_gpuScene.materialBuffer.update(iter->second.assignedIndex, mat->m_packet);
          mat->m_dirtyToGPU = false;
        }
      }

      // Then we update the geometry,
      if (transform.is_dirty_to_gpu() || mesh.is_dirty_to_gpu()) {

        auto iter = m_gpuScene.geometryList.find(entity);

        std::vector<IndexInfo> info_set;


        if (mesh.m_mesh->m_customPrimitives.size() > 0) {
          size_t index_subprimitive = 0;
          for (auto& primitive : mesh.m_mesh->m_customPrimitives) {
            GeometryDrawData geometry;
            geometry.vertexOffset = 0;
            geometry.indexOffset = 0;
            geometry.indexSize = 0;
            geometry.geometryTransform = transform.global;
            geometry.geometryTransformInverse = se::inverse(transform.global);
            geometry.oddNegativeScaling = transform.oddScaling;
            geometry.materialID = primitive.material.get()
              ? m_gpuScene.materialList[primitive.material.get()].assignedIndex : -1;            geometry.primitiveType = primitive.primitiveType;
            geometry.lightID = -1;
            geometry.mediumIDInterior = -1;
            geometry.mediumIDExterior = -1;
            if (primitive.exterior.get())
              geometry.mediumIDExterior = m_gpuScene.mediumPool.try_fetch_index(primitive.exterior);
            if (primitive.interior.get())
              geometry.mediumIDInterior = m_gpuScene.mediumPool.try_fetch_index(primitive.interior);

            if (iter == m_gpuScene.geometryList.end()) {
              IndexInfo info;
              info.assignedIndex = m_gpuScene.geometryBuffer.insert(geometry);
              info.heartBeat = 0;
              info_set.emplace_back(info);
            }
            else {
              m_gpuScene.geometryBuffer.update(
                iter->second[index_subprimitive].assignedIndex,
                geometry
              );
            }
            index_subprimitive++;
          }
        }
        else if (mesh.m_mesh->m_primitives.size() > 0) {
          size_t index_subprimitive = 0;
          for (auto& primitive : mesh.m_mesh->m_primitives) {
            GeometryDrawData geometry;
            geometry.vertexOffset = primitive.baseVertex;
            geometry.indexOffset = primitive.offset;
            geometry.indexSize = primitive.size;
            geometry.geometryTransform = transform.global;
            geometry.geometryTransformInverse = se::inverse(transform.global);
            geometry.oddNegativeScaling = transform.oddScaling;
            geometry.materialID = primitive.material.get()
              ? m_gpuScene.materialList[primitive.material.get()].assignedIndex : -1;
            geometry.primitiveType = 0;
            geometry.meshID = meshID;
            geometry.lightID = -1;
            geometry.mediumIDInterior = -1;
            geometry.mediumIDExterior = -1;
            if (primitive.exterior.get())
              geometry.mediumIDExterior = m_gpuScene.mediumPool.try_fetch_index(primitive.exterior);
            if (primitive.interior.get())
              geometry.mediumIDInterior = m_gpuScene.mediumPool.try_fetch_index(primitive.interior);

            if (iter == m_gpuScene.geometryList.end()) {
              IndexInfo info;
              info.assignedIndex = m_gpuScene.geometryBuffer.insert(geometry);
              info.heartBeat = 0;
              info_set.emplace_back(info);
            }
            else {
              m_gpuScene.geometryBuffer.update(
                iter->second[index_subprimitive].assignedIndex,
                geometry
              );
            }
            index_subprimitive++;
          }
        }

        if (iter == m_gpuScene.geometryList.end()) {
          m_gpuScene.geometryList[entity] = info_set;
        }

        mesh.m_dirtyToGPU = false;
      }

      //for (auto& sub : _property.m_mesh->m_primitives) {

      //}
      //m_gpuScene.meshList.find();

      //size_t node_index = data.nodes.size();
      //data.nodes.emplace(entity, node_index);
      //tinygltf::Node node;
      //node.name = _property.name;
      //data.model->nodes.emplace_back(node);
    }

    m_gpuScene.positionBuffer.m_buffer->host_to_device();
    m_gpuScene.indexBuffer.m_buffer->host_to_device();
    m_gpuScene.vertexBuffer.m_buffer->host_to_device();
    m_gpuScene.materialBuffer.m_buffer->host_to_device();
  }

  auto Scene::update_gpu_camera() noexcept -> void {
    auto node_view = m_registry.view<gfx::Transform, Camera>();
    for (auto [entity, transform, camera] : node_view.each()) {
      auto texture_displayed = Singleton<editor::EditorContext>::instance()->m_viewportTexture;
      if (texture_displayed.has_value()) {
        float aspect_ratio = float(texture_displayed.value()->m_texture->width())
          / texture_displayed.value()->m_texture->height();
        if (aspect_ratio != camera.aspectRatio) {
          camera.aspectRatio = aspect_ratio;
          transform.m_dirtyToGPU = true;
        }
      }

      if(m_viewportSize.has_value()) {
        float aspect_ratio = float(m_viewportSize.value().x) / float(m_viewportSize.value().y);
        if (aspect_ratio != camera.aspectRatio) {
          camera.aspectRatio = aspect_ratio;
        }
      }

      if (transform.is_dirty_to_gpu() || camera.is_dirty_to_gpu()) {
        CameraData camData = CameraData(camera, transform);

        if (camera.medium.get() != nullptr) {
          camData.mediumID = m_gpuScene.mediumPool.try_fetch_index(camera.medium);
        }

        // move to gpu buffer
        auto find = m_gpuScene.cameraList.find(entity);
        if (find == m_gpuScene.cameraList.end()) {
          int32_t index = m_gpuScene.cameraBuffer.insert(camData);
          m_gpuScene.cameraList[entity] = { index, 0 };
        }
        else m_gpuScene.cameraBuffer.m_buffer->copy_to_host(find->second.assignedIndex, camData);
        // camera is no longer dirty
        camera.m_dirtyToGPU = false;
      }
    }
    m_gpuScene.cameraBuffer.m_buffer->host_to_device();
  }

  auto Scene::update_gpu_medium() noexcept -> void {
    //for (auto& pair : m_gpuScene.mediumPool.medium_loc_index) {
    //  if (pair.second.second->isDirty) {
    //    pair.second.second->isDirty = false;
    //    Medium::MediumPacket pack = pair.second.second->packet;
    //    memcpy((float*)&(m_gpuScene.mediumBuffer->host[sizeof(Medium::MediumPacket) *
    //      pair.second.first]), &pack, sizeof(pack));
    //    gpuScene.medium_buffer->host_stamp++;
    //  }
    //}
    m_gpuScene.mediumPool.medium_buffer.m_buffer->host_to_device();
    m_gpuScene.mediumPool.grid_storage_buffer->host_to_device();
  }

  auto Scene::update_gpu_lights() noexcept -> void {
    auto node_light_view = m_registry.view<gfx::Transform, Light>();
    bool lights_dirty = false;

    for (auto [entity, transform, light] : node_light_view.each()) {

      if ((!transform.is_dirty_to_gpu()) && (!light.is_dirty_to_gpu())) continue;

      switch (light.light.light_type) {
      case LightTypeEnum::MESH_PRIMITIVE: {
        MeshHandle& mesh = m_registry.get<MeshRenderer>(entity).m_mesh;
        std::vector<IndexInfo>& indices = m_gpuScene.geometryList[entity];
        // custom mesh primitives
        if (mesh->m_customPrimitives.size() > 0) {
          for (size_t i = 0; i < mesh->m_customPrimitives.size(); ++i) {
            int32_t geometry_index = indices[i].assignedIndex;
            GeometryDrawData& geometry = m_gpuScene.geometryBuffer[geometry_index];

            LightData packet;
            const vec3 emissive = mesh->m_customPrimitives[i].material->m_packet.vec4Data1.xyz();
            const vec3 yuv = {
              0.299f * emissive.r + 0.587f * emissive.g + 0.114f * emissive.b,
              -0.14713f * emissive.r - 0.28886f * emissive.g + 0.436f * emissive.b,
              0.615f * emissive.r - 0.51499f * emissive.g - 0.10001f * emissive.b,
            };
            uint32_t type = mesh->m_customPrimitives[i].primitiveType;
            if (type == 1) {
              packet.light_type = LightTypeEnum::SPHERE;
              vec3 x1 = mul(mat4(geometry.geometryTransform), vec4{ 1, 0, 0, 1 }).xyz();
              vec3 x0 = mul(mat4(geometry.geometryTransform), vec4{ 0, 0, 0, 1 }).xyz();
              float radius = se::length(x1 - x0);
              packet.uintscalar_0 = 0;
              packet.uintscalar_1 = geometry_index;
              bounds3 bound;
              bound.pMin = x0 - vec3{ radius };
              bound.pMax = x0 + vec3{ radius };
              float area = 4 * M_FLOAT_PI * radius * radius;
              vec3 power = yuv * M_FLOAT_PI * area;
              packet.floatvec_0 = { power , 0 };
              packet.floatvec_1 = { bound.pMin, 0 };
              packet.floatvec_2 = { bound.pMax, 0 };
            }
            else if (type == 2) {
              packet.light_type = LightTypeEnum::RECTANGLE;
              packet.uintscalar_0 = 0;
              packet.uintscalar_1 = geometry_index;

              vec3 x0 = mul(mat4(geometry.geometryTransform), vec4{ 1, 1, 0, 1 }).xyz();
              vec3 x1 = mul(mat4(geometry.geometryTransform), vec4{ 1, -1, 0, 1 }).xyz();
              vec3 x2 = mul(mat4(geometry.geometryTransform), vec4{ -1, 1, 0, 1 }).xyz();
              vec3 x3 = mul(mat4(geometry.geometryTransform), vec4{ -1, -1, 0, 1 }).xyz();
              bounds3 bound;
              bound = unionPoint(bound, point3(x0));
              bound = unionPoint(bound, point3(x1));
              bound = unionPoint(bound, point3(x2));
              bound = unionPoint(bound, point3(x3));
              float area = length(x0 - x2) * length(x1 - x0);
              vec3 power = yuv * M_FLOAT_PI * area;
              packet.floatvec_0 = { power , 0 };
              packet.floatvec_1 = { bound.pMin, 0 };
              packet.floatvec_2 = { bound.pMax, 0 };
            }
            else if (type == 3) {

            }

            int light_index = m_gpuScene.lightBuffer.insert(packet);
            geometry.lightID = light_index;
            m_gpuScene.lightList[entity].push_back({ light_index });
          }
        }
        // triangle mesh primitives
        else {
          for (size_t i = 0; i < mesh->m_primitives.size(); ++i) {
            int32_t geometry_index = indices[i].assignedIndex;
            GeometryDrawData& geometry = m_gpuScene.geometryBuffer[geometry_index];
            std::vector<LightData> packets(geometry.indexSize / 3);
            const vec3 emissive = mesh->m_primitives[i].material->m_packet.vec4Data1.xyz();
            const vec3 yuv = {
              0.299f * emissive.r + 0.587f * emissive.g + 0.114f * emissive.b,
              -0.14713f * emissive.r - 0.28886f * emissive.g + 0.436f * emissive.b,
              0.615f * emissive.r - 0.51499f * emissive.g - 0.10001f * emissive.b,
            };
            for (int j = 0; j < geometry.indexSize / 3; j++) {
              packets[j].light_type = LightTypeEnum::MESH_PRIMITIVE;
              packets[j].uintscalar_0 = j;
              packets[j].uintscalar_1 = geometry_index;
              // todo (twoSided ? 2 : 1)
              uvec3 indices = mesh->m_indexBuffer->read_from_host<uvec3>(j);
              vec3 v0 = mesh->m_positionBuffer->read_from_host<vec3>(indices[0] + int(geometry.vertexOffset));
              vec3 v1 = mesh->m_positionBuffer->read_from_host<vec3>(indices[1] + int(geometry.vertexOffset));
              vec3 v2 = mesh->m_positionBuffer->read_from_host<vec3>(indices[2] + int(geometry.vertexOffset));
              v0 = mul(mat4(geometry.geometryTransform), { v0, 1 }).xyz();
              v1 = mul(mat4(geometry.geometryTransform), { v1, 1 }).xyz();
              v2 = mul(mat4(geometry.geometryTransform), { v2, 1 }).xyz();
              float area = 0.5f * length(cross(v1 - v0, v2 - v0));
              bounds3 bound;
              bound = unionPoint(bound, point3(v0));
              bound = unionPoint(bound, point3(v1));
              bound = unionPoint(bound, point3(v2));

              normal3 n = normalize(normal3(cross(v1 - v0, v2 - v0)));
              // Ensure correct orientation of geometric normal for normal bounds
              vec3 n0 = mesh->m_vertexBuffer->read_from_host<vec3>(indices[0] + int(geometry.vertexOffset), sizeof(float) * 8, 0);
              vec3 n1 = mesh->m_vertexBuffer->read_from_host<vec3>(indices[1] + int(geometry.vertexOffset), sizeof(float) * 8, 0);
              vec3 n2 = mesh->m_vertexBuffer->read_from_host<vec3>(indices[2] + int(geometry.vertexOffset), sizeof(float) * 8, 0);
              n0 = mul(mat4(geometry.geometryTransformInverse), { n0, 0 }).xyz();
              n1 = mul(mat4(geometry.geometryTransformInverse), { n1, 0 }).xyz();
              n2 = mul(mat4(geometry.geometryTransformInverse), { n2, 0 }).xyz();
              //normal3 ns = normalize(n0 + n1 + n2);
              //n = faceForward(n, ns);
              n *= geometry.oddNegativeScaling;

              vec3 power = yuv * M_FLOAT_PI * area;
              packets[j].floatvec_0 = { power , n.x };
              packets[j].floatvec_1 = { bound.pMin, n.y };
              packets[j].floatvec_2 = { bound.pMax, n.z };
            }

            int light_index = m_gpuScene.lightBuffer.insert_consecutive(packets);
            geometry.lightID = light_index;
            m_gpuScene.lightList[entity].push_back({ light_index, 0, int32_t(packets.size()) });
          }
        }
        break;
      }
      default: break;
      }
      
      lights_dirty = true;
      light.m_dirtyToGPU = false;
    }

    if (lights_dirty) {
      update_gpu_lightbvh();

      m_gpuScene.sceneInfo.data->nondistant_light_count = m_gpuScene.lightBuffer.m_size;
      m_gpuScene.sceneInfo.data->light_bounds_min = m_gpuScene.lightSampler.allLightBounds.pMin;
      m_gpuScene.sceneInfo.data->light_bounds_max = m_gpuScene.lightSampler.allLightBounds.pMax;
    }
    m_gpuScene.lightBuffer.m_buffer->host_to_device();
  }

  auto Scene::update_gpu_bvh() noexcept -> void {
    if (!(gfx::GFXContext::device()->from_which_adapter()->from_which_context()
      ->get_context_extensions_flags() & rhi::ContextExtensionEnum::RAY_TRACING))
      return;

    bool should_rebuilt_tlas = false;

    auto node_view = m_registry.view<Transform, MeshRenderer>();
    for (auto [entity, transform, mesh] : node_view.each()) {
      if (mesh.m_mesh->m_customPrimitives.size() > 0) {
        for (auto& primitive : mesh.m_mesh->m_customPrimitives) {
          // if BLAS not exist, create one
          if (primitive.primBlas == nullptr) {
            should_rebuilt_tlas = true;
            primitive.blasDesc.allowCompaction = true;
            primitive.blasDesc.customGeometries.push_back(rhi::BLASCustomGeometry{
              rhi::AffineTransformMatrix{},
              std::vector<se::bounds3>{se::bounds3{primitive.min, primitive.max}},
              (uint32_t)rhi::BLASGeometryEnum::NO_DUPLICATE_ANY_HIT_INVOCATION
              | (uint32_t)rhi::BLASGeometryEnum::OPAQUE_GEOMETRY,
              });
            primitive.primBlas = GFXContext::device()->create_blas(primitive.blasDesc);
          }
        }
      }
      else {
        for (auto& primitive : mesh.m_mesh->m_primitives) {
          // if BLAS not exist, create one
          if (primitive.primBlas == nullptr) {
            should_rebuilt_tlas = true;
            primitive.blasDesc.allowCompaction = true;
            primitive.blasDesc.triangleGeometries.push_back(rhi::BLASTriangleGeometry{
              mesh.m_mesh->m_positionBuffer->m_buffer.get(),
              mesh.m_mesh->m_indexBuffer->m_buffer.get(),
              rhi::IndexFormat::UINT32_T,
              uint32_t(primitive.numVertex - 1),
              uint32_t(primitive.baseVertex),
              uint32_t(primitive.size / 3),
              uint32_t(primitive.offset * sizeof(uint32_t)),
              rhi::AffineTransformMatrix{},
              (uint32_t)rhi::BLASGeometryEnum::NO_DUPLICATE_ANY_HIT_INVOCATION
              | (uint32_t)rhi::BLASGeometryEnum::OPAQUE_GEOMETRY,
              0 });
            primitive.primBlas = GFXContext::device()->create_blas(primitive.blasDesc);
          }
        }
      }

      auto iter = m_gpuScene.tlas.instanceList.find(entity);
      if (iter == m_gpuScene.tlas.instanceList.end()) {
        if (mesh.m_mesh->m_customPrimitives.size() > 0) {
          for (auto& primitive : mesh.m_mesh->m_customPrimitives) {
            should_rebuilt_tlas = true;
            // update the instance of the mesh resource
            rhi::BLASInstance instance;
            instance.blas = primitive.primBlas.get();
            instance.transform = transform.global;
            instance.instanceCustomIndex = primitive.primitiveType; // geometry_start
            instance.instanceShaderBindingTableRecordOffset = 0;

            int32_t index = m_gpuScene.tlas.desc.instances.size();
            m_gpuScene.tlas.desc.instances.push_back(instance);
            m_gpuScene.tlas.instanceList[entity].push_back(IndexInfo{ index });
          }
        }
        else {
          for (auto& primitive : mesh.m_mesh->m_primitives) {
            should_rebuilt_tlas = true;
            // update the instance of the mesh resource
            rhi::BLASInstance instance;
            instance.blas = primitive.primBlas.get();
            instance.transform = transform.global;
            instance.instanceCustomIndex = 0; // geometry_start
            instance.instanceShaderBindingTableRecordOffset = 0;

            int32_t index = m_gpuScene.tlas.desc.instances.size();
            m_gpuScene.tlas.desc.instances.push_back(instance);
            m_gpuScene.tlas.instanceList[entity].push_back(IndexInfo{ index });
          }
        }
      }
      else if (transform.is_dirty_to_gpu()) {
        size_t instance_offset = 0;
        if (mesh.m_mesh->m_customPrimitives.size() > 0) {
          for (auto& primitive : mesh.m_mesh->m_customPrimitives) {
            should_rebuilt_tlas = true;
            // update the instance of the mesh resource
            rhi::BLASInstance instance;
            instance.blas = primitive.primBlas.get();
            instance.transform = transform.global;
            instance.instanceCustomIndex = primitive.primitiveType; // geometry_start
            instance.instanceShaderBindingTableRecordOffset = 0;

            int32_t index = m_gpuScene.tlas.instanceList[entity][instance_offset++].assignedIndex;
            m_gpuScene.tlas.desc.instances[index] = instance;
          }
        }
        else {
          for (auto& primitive : mesh.m_mesh->m_primitives) {
            should_rebuilt_tlas = true;
            // update the instance of the mesh resource
            rhi::BLASInstance instance;
            instance.blas = primitive.primBlas.get();
            instance.transform = transform.global;
            instance.instanceCustomIndex = 0; // geometry_start
            instance.instanceShaderBindingTableRecordOffset = 0;

            int32_t index = m_gpuScene.tlas.instanceList[entity][instance_offset++].assignedIndex;
            m_gpuScene.tlas.desc.instances[index] = instance;
          }
        }


      }
    }

    if (should_rebuilt_tlas) {
      m_gpuScene.tlas.back = std::move(m_gpuScene.tlas.prim);
      m_gpuScene.tlas.prim = GFXContext::device()->create_tlas(m_gpuScene.tlas.desc);

    }
  }

  auto Scene::draw_meshes(rhi::RenderPassEncoder* encoder, int32_t geometryID_offset) noexcept -> void {
    for (auto& iter : m_gpuScene.geometryList) {
      for (auto& index_info : iter.second) {
        int geometryID = index_info.assignedIndex;
        GeometryDrawData& draw = m_gpuScene.geometryBuffer[geometryID];
        encoder->push_constants(&geometryID, se::rhi::ShaderStageEnum::VERTEX
          | se::rhi::ShaderStageEnum::FRAGMENT,
          geometryID_offset, sizeof(int32_t));
        encoder->draw(draw.indexSize, 1, 0, 0);
      }
    }
  }

  auto Scene::GPUScene::ImagePool::try_fetch_index(TextureHandle texture) noexcept -> int {
    auto iter = texture_loc_index.find(texture->m_uid);
    if (iter == texture_loc_index.end()) {
      int index = texture_loc_index.size();
      texture_loc_index[texture->m_uid] = { index, texture };
      prim_t.push_back(texture->get_srv(0, 1, 0, 1));
      SamplerHandle sampler = GFXContext::create_sampler_desc(rhi::SamplerDescriptor{});
      prim_s.push_back(sampler.get()->m_sampler.get());
      return index;
    }
    return iter->second.first;
  }

  auto Scene::GPUScene::MediumPool::try_fetch_index(MediumHandle handle) noexcept -> int {
    auto iter = medium_loc_index.find(handle->m_uid);
    if (iter == medium_loc_index.end()) {
      int index = medium_loc_index.size();
      medium_loc_index[handle->m_uid] = { index, handle };

      // upload density grid
      if (handle->density.has_value()) {
        handle->packet.boundMin = handle->density->bounds.pMin;
        handle->packet.boundMax = handle->density->bounds.pMax;
        handle->packet.densityNxyz = ivec3{ handle->density->nx, handle->density->ny, handle->density->nz };
        int size = handle->density->values.size();
        int offset = grid_storage_buffer->m_host.size() / sizeof(float);
        offset = int((offset + 63) / 64) * 64;
        handle->packet.densityOffset = offset;
        grid_storage_buffer->m_host.resize(sizeof(float) * (offset + size));
        memcpy(&(grid_storage_buffer->m_host[offset * sizeof(float)]),
          handle->density->values.data(), size * sizeof(float));
        grid_storage_buffer->m_hostStamp++;
      }

      // upload temperature grid
      if (handle->temperatureGrid.has_value()) {
        handle->packet.temperatureNxyz = ivec3{ handle->temperatureGrid->nx, handle->temperatureGrid->ny, handle->temperatureGrid->nz };
        int size = handle->temperatureGrid->values.size();
        int offset = grid_storage_buffer->m_host.size() / sizeof(float);
        offset = int((offset + 63) / 64) * 64;
        handle->packet.temperatureOffset = offset;
        handle->packet.temperatureBoundMin = handle->temperatureGrid->bounds.pMin;
        handle->packet.temperatureBoundMax = handle->temperatureGrid->bounds.pMax;
        grid_storage_buffer->m_host.resize(sizeof(float) * (offset + size));
        if (size > 0)
          memcpy(&(grid_storage_buffer->m_host[offset * sizeof(float)]),
            handle->temperatureGrid->values.data(), size * sizeof(float));
        grid_storage_buffer->m_hostStamp++;
      }

      // upload majorant grid
      if (handle->majorantGrid.has_value()) {
        handle->packet.majorantNxyz = handle->majorantGrid->res;
        int size = handle->majorantGrid->voxels.size();
        int offset = grid_storage_buffer->m_host.size() / sizeof(float);
        offset = int((offset + 63) / 64) * 64;
        handle->packet.majorantOffset = offset;
        grid_storage_buffer->m_host.resize(sizeof(float) * (offset + size));
        memcpy(&(grid_storage_buffer->m_host[offset * sizeof(float)]),
          handle->majorantGrid->voxels.data(), size * sizeof(float));
        grid_storage_buffer->m_hostStamp++;
      }

      // copy the packet to GPU
      Medium::MediumPacket pack = handle->packet;
      medium_buffer.insert(pack);
      return index;
    }
    return iter->second.first;
  }

  auto Scene::GPUScene::binding_resource_position() noexcept -> rhi::BindingResource {
    return rhi::BindingResource{ {positionBuffer.m_buffer->m_buffer.get(), 0, positionBuffer.m_buffer->m_buffer->size()} };
  }

  auto Scene::GPUScene::binding_resource_index() noexcept -> rhi::BindingResource {
    return rhi::BindingResource{ {indexBuffer.m_buffer->m_buffer.get(), 0, indexBuffer.m_buffer->m_buffer->size()} };
  }

  auto Scene::GPUScene::binding_resource_vertex() noexcept -> rhi::BindingResource {
    return rhi::BindingResource{ {vertexBuffer.m_buffer->m_buffer.get(), 0, vertexBuffer.m_buffer->m_buffer->size()} };
  }

  auto Scene::GPUScene::binding_resource_camera() noexcept -> rhi::BindingResource {
    return rhi::BindingResource{ {cameraBuffer.m_buffer->m_buffer.get(), 0, cameraBuffer.m_buffer->m_buffer->size()} };
  }

  auto Scene::GPUScene::binding_resource_geometry() noexcept -> rhi::BindingResource {
    return rhi::BindingResource{ {geometryBuffer.m_buffer->m_buffer.get(), 0, geometryBuffer.m_buffer->m_buffer->size()} };
  }

  auto Scene::GPUScene::binding_resource_material() noexcept -> rhi::BindingResource {
    return rhi::BindingResource{ {materialBuffer.m_buffer->m_buffer.get(), 0, materialBuffer.m_buffer->m_buffer->size()} };
  }

  auto Scene::GPUScene::binding_resource_textures() noexcept -> rhi::BindingResource {
    return rhi::BindingResource{ imagePool.prim_t, imagePool.prim_s };
  }

  auto Scene::GPUScene::binding_resource_light() noexcept -> rhi::BindingResource {
    return rhi::BindingResource{ {lightBuffer.m_buffer->m_buffer.get(), 0, lightBuffer.m_buffer->m_buffer->size()} };
  }

  auto Scene::GPUScene::binding_resource_sceneinfo() noexcept -> rhi::BindingResource {
    return rhi::BindingResource{ {sceneInfo.sceneBuffer->m_buffer.get(), 0, sceneInfo.sceneBuffer->m_buffer->size()} };
  }

  auto Scene::GPUScene::binding_resource_lightbvh_tree() noexcept -> rhi::BindingResource {
    if (lightSampler.treeBuffer->m_buffer.get() == nullptr) {
      lightSampler.treeBuffer->m_host.resize(64);
      lightSampler.treeBuffer->m_hostStamp++;
      lightSampler.treeBuffer->host_to_device();
    }
    return rhi::BindingResource{ {lightSampler.treeBuffer->m_buffer.get(), 0, lightSampler.treeBuffer->m_buffer->size()} };
  }

  auto Scene::GPUScene::binding_resource_lightbvh_trail() noexcept -> rhi::BindingResource {
    if (lightSampler.trailBuffer->m_buffer.get() == nullptr) {
      lightSampler.trailBuffer->m_host.resize(64);
      lightSampler.trailBuffer->m_hostStamp++;
      lightSampler.trailBuffer->host_to_device();
    }
    return rhi::BindingResource{ {lightSampler.trailBuffer->m_buffer.get(), 0, lightSampler.trailBuffer->m_buffer->size()} };
  }

  auto Scene::GPUScene::binding_resource_tlas() noexcept -> rhi::BindingResource {
    return rhi::BindingResource{ {tlas.prim.get()} };
  }

  auto Scene::GPUScene::binding_resource_medium() noexcept -> rhi::BindingResource {
    return rhi::BindingResource{ rhi::BufferBinding{
        mediumPool.medium_buffer.m_buffer->m_buffer.get(), 0,
        mediumPool.medium_buffer.m_buffer->m_buffer->size()} };
  }

  auto Scene::GPUScene::binding_resource_medium_grid() noexcept -> rhi::BindingResource {
    return rhi::BindingResource{ rhi::BufferBinding{
        mediumPool.grid_storage_buffer->m_buffer.get(), 0,
        mediumPool.grid_storage_buffer->m_buffer->size()} };
  }

}
}