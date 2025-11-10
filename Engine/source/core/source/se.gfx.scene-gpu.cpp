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
    
    update_gpu_meshes(&m_gpuScene);
    m_gpuScene.positionBuffer.m_buffer->host_to_device();
    m_gpuScene.indexBuffer.m_buffer->host_to_device();
    m_gpuScene.vertexBuffer.m_buffer->host_to_device();
    m_gpuScene.materialBuffer.m_buffer->host_to_device();
    
    update_gpu_camera(&m_gpuScene);
    m_gpuScene.cameraBuffer.m_buffer->host_to_device();

    update_gpu_lights(&m_gpuScene);
    m_gpuScene.lightBuffer.m_buffer->host_to_device();
    m_gpuScene.sceneDataList.m_buffer->host_to_device();
    m_gpuScene.lbvhTreeBuffer.m_buffer->host_to_device();
    m_gpuScene.lbvhTrailBuffer.m_buffer->host_to_device();
     
    update_gpu_medium(&m_gpuScene);
    m_gpuScene.mediumPool.medium_buffer.m_buffer->host_to_device();
    m_gpuScene.mediumPool.grid_storage_buffer->host_to_device();

    update_gpu_bvh(&m_gpuScene);

    // set all transform dirty flag to false
    auto node_view = m_registry.view<Transform>();
    for (auto [entity, _transform] : node_view.each()) {
      _transform.m_dirtyToGPU = false;
    }

    m_gpuScene.geometryBuffer.m_buffer->host_to_device();
  }

  auto Scene::update_gpu_meshes(GPUScene* gpu_scene) noexcept -> void {
    // Iterate through all nodes with Transform and MeshRenderer components
    auto node_view = m_registry.view<Transform, MeshRenderer>();
    for (auto [entity, transform, mesh] : node_view.each()) {
      // If the mesh resource itself is dirty, we update the reference to the mesh
      if (mesh.m_mesh->m_dirtyToGPU) {
        uint64_t vertex_address = mesh.m_mesh->m_vertexBuffer->m_buffer->get_device_address();
        uint64_t pos_address = mesh.m_mesh->m_positionBuffer->m_buffer->get_device_address();
        uint64_t index_address = mesh.m_mesh->m_indexBuffer->m_buffer->get_device_address();
        // Register the mesh if not already in the gpu scene
        auto iter = gpu_scene->meshList.find(mesh.m_mesh.get());
        if (iter == gpu_scene->meshList.end()) {
          int32_t index = gpu_scene->positionBuffer.insert(pos_address);
          gpu_scene->vertexBuffer.insert(vertex_address);
          gpu_scene->indexBuffer.insert(index_address);
          gpu_scene->meshList[mesh.m_mesh.get()] = IndexInfo{ index, 0 };
        }
        else {
          se::error("todo :: a mesh is dirty after first register");
        }
        mesh.m_mesh->m_dirtyToGPU = false;
      }

      // The mesh get a uniform ID in the mesh-list
      int16_t meshID = (int16_t)gpu_scene->meshList[mesh.m_mesh.get()].assignedIndex;

      // After we have the mesh resource ready,
      // we take care of the material resource
      for (auto& primitive : mesh.m_mesh->m_customPrimitives) {
        MaterialHandle mat = primitive.material;
        auto iter = gpu_scene->materialList.find(mat.get());
        if (iter == gpu_scene->materialList.end()) {
          MaterialInterpreterManager::init(mat.get(), mat->m_packet.bxdfType);
          int32_t index = gpu_scene->materialBuffer.insert(mat->m_packet);
          gpu_scene->materialList[mat.get()] = IndexInfo{ index, 0 };
          mat->m_dirtyToGPU = false;
        }
        else if (mat->m_dirtyToGPU == true) {
          MaterialInterpreterManager::init(mat.get(), mat->m_packet.bxdfType);
          gpu_scene->materialBuffer.update(iter->second.assignedIndex, mat->m_packet);
          mat->m_dirtyToGPU = false;
        }
      }
      for (auto& primitive : mesh.m_mesh->m_primitives) {
        MaterialHandle mat = primitive.material;
        if (!mat.get()) continue;
        auto iter = gpu_scene->materialList.find(mat.get());
        if (iter == gpu_scene->materialList.end()) {
          MaterialInterpreterManager::init(mat.get(), mat->m_packet.bxdfType);
          if (mat->m_basecolorTex.get()) {
            mat->m_packet.baseTex = gpu_scene->imagePool.try_fetch_index(mat->m_basecolorTex);
          }
          else mat->m_packet.baseTex = -1;

          int32_t index = gpu_scene->materialBuffer.insert(mat->m_packet);
          gpu_scene->materialList[mat.get()] = IndexInfo{ index, 0 };
          mat->m_dirtyToGPU = false;
        }
        else if (mat->m_dirtyToGPU == true) {
          MaterialInterpreterManager::init(mat.get(), mat->m_packet.bxdfType);
          gpu_scene->materialBuffer.update(iter->second.assignedIndex, mat->m_packet);
          mat->m_dirtyToGPU = false;
        }
      }

      // Then we update the geometry,
      if (transform.is_dirty_to_gpu() || mesh.is_dirty_to_gpu()) {

        auto iter = gpu_scene->geometryList.find(entity);

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
              ? gpu_scene->materialList[primitive.material.get()].assignedIndex : -1;            geometry.primitiveType = primitive.primitiveType;
            geometry.lightID = -1;
            geometry.mediumIDInterior = -1;
            geometry.mediumIDExterior = -1;
            if (primitive.exterior.get())
              geometry.mediumIDExterior = gpu_scene->mediumPool.try_fetch_index(primitive.exterior);
            if (primitive.interior.get())
              geometry.mediumIDInterior = gpu_scene->mediumPool.try_fetch_index(primitive.interior);

            if (iter == gpu_scene->geometryList.end()) {
              IndexInfo info;
              info.assignedIndex = gpu_scene->geometryBuffer.insert(geometry);
              info.heartBeat = 0;
              info_set.emplace_back(info);
            }
            else {
              gpu_scene->geometryBuffer.update(
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
              ? gpu_scene->materialList[primitive.material.get()].assignedIndex : -1;
            geometry.primitiveType = 0;
            geometry.meshID = meshID;
            geometry.lightID = -1;
            geometry.mediumIDInterior = -1;
            geometry.mediumIDExterior = -1;
            if (primitive.exterior.get())
              geometry.mediumIDExterior = gpu_scene->mediumPool.try_fetch_index(primitive.exterior);
            if (primitive.interior.get())
              geometry.mediumIDInterior = gpu_scene->mediumPool.try_fetch_index(primitive.interior);

            if (iter == gpu_scene->geometryList.end()) {
              IndexInfo info;
              info.assignedIndex = gpu_scene->geometryBuffer.insert(geometry);
              info.heartBeat = 0;
              info_set.emplace_back(info);
            }
            else {
              gpu_scene->geometryBuffer.update(
                iter->second[index_subprimitive].assignedIndex,
                geometry
              );
            }
            index_subprimitive++;
          }
        }

        if (iter == gpu_scene->geometryList.end()) {
          gpu_scene->geometryList[entity] = info_set;
        }

        mesh.m_dirtyToGPU = false;
      }

      //for (auto& sub : _property.m_mesh->m_primitives) {

      //}
      //gpu_scene->meshList.find();

      //size_t node_index = data.nodes.size();
      //data.nodes.emplace(entity, node_index);
      //tinygltf::Node node;
      //node.name = _property.name;
      //data.model->nodes.emplace_back(node);
    }
  }

  auto Scene::update_gpu_camera(GPUScene* gpu_scene) noexcept -> void {
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
          camData.mediumID = gpu_scene->mediumPool.try_fetch_index(camera.medium);
        }

        // move to gpu buffer
        SceneEntity key{ this, entity };
        auto find = gpu_scene->cameraList.find(key);
        if (find == gpu_scene->cameraList.end()) {
          int32_t index = gpu_scene->cameraBuffer.insert(camData);
          gpu_scene->cameraList[key] = { index, 0 };
        }
        else gpu_scene->cameraBuffer.m_buffer->copy_to_host(find->second.assignedIndex, camData);
        // camera is no longer dirty
        camera.m_dirtyToGPU = false;
      }
    }
  }

  auto Scene::update_gpu_medium(GPUScene* gpu_scene) noexcept -> void {
    //for (auto& pair : m_gpuScene.mediumPool.medium_loc_index) {
    //  if (pair.second.second->isDirty) {
    //    pair.second.second->isDirty = false;
    //    Medium::MediumPacket pack = pair.second.second->packet;
    //    memcpy((float*)&(m_gpuScene.mediumBuffer->host[sizeof(Medium::MediumPacket) *
    //      pair.second.first]), &pack, sizeof(pack));
    //    gpuScene.medium_buffer->host_stamp++;
    //  }
    //}
  }

  auto Scene::update_gpu_lights(GPUScene* gpu_scene) noexcept -> void {
    auto node_light_view = m_registry.view<gfx::Transform, Light>();
    bool lights_dirty = false;

    for (auto [entity, transform, light] : node_light_view.each()) {
      // Skip if neither transform nor light is dirty
      if ((!transform.is_dirty_to_gpu()) && (!light.is_dirty_to_gpu())) continue;

      switch (light.light.light_type) {
      case LightTypeEnum::MESH_PRIMITIVE: {
        MeshHandle& mesh = m_registry.get<MeshRenderer>(entity).m_mesh;
        std::vector<IndexInfo>& indices = gpu_scene->geometryList[entity];
        // custom mesh primitives
        if (mesh->m_customPrimitives.size() > 0) {
          for (size_t i = 0; i < mesh->m_customPrimitives.size(); ++i) {
            int32_t geometry_index = indices[i].assignedIndex;
            GeometryDrawData& geometry = gpu_scene->geometryBuffer[geometry_index];

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

            int light_index = gpu_scene->lightBuffer.insert(packet);
            geometry.lightID = light_index;
            gpu_scene->lightList[entity].push_back({ light_index });
          }
        }
        // triangle mesh primitives
        else {
          for (size_t i = 0; i < mesh->m_primitives.size(); ++i) {
            int32_t geometry_index = indices[i].assignedIndex;
            GeometryDrawData& geometry = gpu_scene->geometryBuffer[geometry_index];
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

            int light_index = gpu_scene->lightBuffer.insert_consecutive(packets);
            geometry.lightID = light_index;
            gpu_scene->lightList[entity].push_back({ light_index, 0, int32_t(packets.size()) });
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

      int32_t scene_index = -1;
      auto iter = gpu_scene->m_sceneIDMap.find(this);
      if (iter == gpu_scene->m_sceneIDMap.end()) {
        // upload light information to scene data
        Scene::GPUScene::SceneData data;
        data.nondistant_light_count = gpu_scene->lightBuffer.m_size;
        data.light_bounds_min = m_perSceneGPUInfo.lightSampler.allLightBounds.pMin;
        data.light_bounds_max = m_perSceneGPUInfo.lightSampler.allLightBounds.pMax;
        // push data to gpu buffer
        scene_index = gpu_scene->sceneDataList.insert(data);
        gpu_scene->m_sceneIDMap[this] = scene_index;
      }
      else {
        // fetch scene data and update
        scene_index = iter->second;
        Scene::GPUScene::SceneData& data = gpu_scene->sceneDataList[scene_index];
        data.nondistant_light_count = gpu_scene->lightBuffer.m_size;
        data.light_bounds_min = m_perSceneGPUInfo.lightSampler.allLightBounds.pMin;
        data.light_bounds_max = m_perSceneGPUInfo.lightSampler.allLightBounds.pMax;
      }

      // upload lbvh to gpu_scene
      uint64_t tree_address = 0;
      if (m_perSceneGPUInfo.lightSampler.trailBuffer->m_buffer.get())
        tree_address = m_perSceneGPUInfo.lightSampler.treeBuffer->m_buffer->get_device_address();
      uint64_t trail_address = 0;
      if (m_perSceneGPUInfo.lightSampler.trailBuffer->m_buffer.get())
        trail_address = m_perSceneGPUInfo.lightSampler.trailBuffer->m_buffer->get_device_address();
      
      gpu_scene->lbvhTreeBuffer.set_or_update(scene_index, tree_address);
      gpu_scene->lbvhTrailBuffer.set_or_update(scene_index, trail_address);
    }
  }

  auto Scene::update_gpu_bvh(GPUScene* gpu_scene) noexcept -> void {
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

      auto iter = m_perSceneGPUInfo.tlas.instanceList.find(entity);
      if (iter == m_perSceneGPUInfo.tlas.instanceList.end()) {
        if (mesh.m_mesh->m_customPrimitives.size() > 0) {
          for (auto& primitive : mesh.m_mesh->m_customPrimitives) {
            should_rebuilt_tlas = true;
            // update the instance of the mesh resource
            rhi::BLASInstance instance;
            instance.blas = primitive.primBlas.get();
            instance.transform = transform.global;
            instance.instanceCustomIndex = primitive.primitiveType; // geometry_start
            instance.instanceShaderBindingTableRecordOffset = 0;

            int32_t index = m_perSceneGPUInfo.tlas.desc.instances.size();
            m_perSceneGPUInfo.tlas.desc.instances.push_back(instance);
            m_perSceneGPUInfo.tlas.instanceList[entity].push_back(IndexInfo{ index });
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

            int32_t index = m_perSceneGPUInfo.tlas.desc.instances.size();
            m_perSceneGPUInfo.tlas.desc.instances.push_back(instance);
            m_perSceneGPUInfo.tlas.instanceList[entity].push_back(IndexInfo{ index });
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

            int32_t index = m_perSceneGPUInfo.tlas.instanceList[entity][instance_offset++].assignedIndex;
            m_perSceneGPUInfo.tlas.desc.instances[index] = instance;
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

            int32_t index = m_perSceneGPUInfo.tlas.instanceList[entity][instance_offset++].assignedIndex;
            m_perSceneGPUInfo.tlas.desc.instances[index] = instance;
          }
        }


      }
    }

    if (should_rebuilt_tlas) {
      m_perSceneGPUInfo.tlas.back = std::move(m_perSceneGPUInfo.tlas.prim);
      m_perSceneGPUInfo.tlas.prim = GFXContext::device()->create_tlas(m_perSceneGPUInfo.tlas.desc);
    }

    // get scene index
    int32_t scene_index = -1;
    auto iter = gpu_scene->m_sceneIDMap.find(this);
    if (iter == gpu_scene->m_sceneIDMap.end()) {
      scene_index = gpu_scene->m_sceneIDMap.size();
      gpu_scene->m_sceneIDMap[this] = scene_index;
    }
    else {
      scene_index = iter->second;
    }
    // upload tlas to gpu_scene
    if(gpu_scene->tlasList.size() <= scene_index)
      gpu_scene->tlasList.resize(scene_index + 1);
    gpu_scene->tlasList[scene_index] = m_perSceneGPUInfo.tlas.prim.get();
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

  auto Scene::PerSceneGPUInfo::reset() noexcept -> void {
    lightSampler.treeBuffer = GFXContext::create_buffer_empty();
    lightSampler.treeBuffer->m_job = "Scene light-bvh tree buffer";
    lightSampler.treeBuffer->m_usages = rhi::BufferUsageEnum::STORAGE
      | rhi::BufferUsageEnum::SHADER_DEVICE_ADDRESS;

    lightSampler.trailBuffer = GFXContext::create_buffer_empty();
    lightSampler.trailBuffer->m_job = "Scene light-bvh trail buffer";
    lightSampler.trailBuffer->m_usages = rhi::BufferUsageEnum::STORAGE
      | rhi::BufferUsageEnum::SHADER_DEVICE_ADDRESS;
  }
  
  auto Scene::GPUScene::reset() noexcept -> void {
    meshList.clear();
    cameraList.clear();
    geometryList.clear();

    positionBuffer = DynamicVectorBufferView<uint64_t>();
    positionBuffer.m_buffer = GFXContext::create_buffer_empty();
    positionBuffer.m_buffer->m_job = "Scene position buffer";
    positionBuffer.m_buffer->m_usages = rhi::BufferUsageEnum::STORAGE;

    indexBuffer = DynamicVectorBufferView<uint64_t>();
    indexBuffer.m_buffer = GFXContext::create_buffer_empty();
    indexBuffer.m_buffer->m_job = "Scene index buffer";
    indexBuffer.m_buffer->m_usages = rhi::BufferUsageEnum::STORAGE;

    vertexBuffer = DynamicVectorBufferView<uint64_t>();
    vertexBuffer.m_buffer = GFXContext::create_buffer_empty();
    vertexBuffer.m_buffer->m_job = "Scene vertex buffer";
    vertexBuffer.m_buffer->m_usages = rhi::BufferUsageEnum::STORAGE;

    cameraBuffer = DynamicVectorBufferView<CameraData>();
    cameraBuffer.m_buffer = GFXContext::create_buffer_empty();
    cameraBuffer.m_buffer->m_job = "Scene camera buffer";
    cameraBuffer.m_buffer->m_usages = rhi::BufferUsageEnum::STORAGE;
    cameraBuffer.m_buffer->m_memoryCopyMode = gfx::Buffer::MemoryCopyMode::COHERENT_MAPPING;

    geometryBuffer = DynamicVectorBufferView<GeometryDrawData>();
    geometryBuffer.m_buffer = GFXContext::create_buffer_empty();
    geometryBuffer.m_buffer->m_job = "Scene geometry buffer";
    geometryBuffer.m_buffer->m_usages = rhi::BufferUsageEnum::STORAGE;

    materialBuffer = DynamicVectorBufferView<Material::MaterialPacket>();
    materialBuffer.m_buffer = GFXContext::create_buffer_empty();
    materialBuffer.m_buffer->m_job = "Scene material buffer";
    materialBuffer.m_buffer->m_usages = rhi::BufferUsageEnum::STORAGE;

    lightBuffer = DynamicVectorBufferView<LightData>();
    lightBuffer.m_buffer = GFXContext::create_buffer_empty();
    lightBuffer.m_buffer->m_job = "Scene light buffer";
    lightBuffer.m_buffer->m_usages = rhi::BufferUsageEnum::STORAGE;

    mediumPool.medium_buffer = DynamicVectorBufferView<Medium::MediumPacket>();
    mediumPool.medium_buffer.m_buffer = GFXContext::create_buffer_empty();
    mediumPool.medium_buffer.m_buffer->m_job = "Scene medium desc buffer buffer";
    mediumPool.medium_buffer.m_buffer->m_usages = rhi::BufferUsageEnum::STORAGE;
    mediumPool.medium_buffer.m_buffer->m_memoryCopyMode = gfx::Buffer::MemoryCopyMode::COHERENT_MAPPING;

    mediumPool.grid_storage_buffer = GFXContext::create_buffer_empty();
    mediumPool.grid_storage_buffer->m_job = "Scene medium storage buffer";
    mediumPool.grid_storage_buffer->m_usages = rhi::BufferUsageEnum::STORAGE;

    lbvhTreeBuffer = DynamicVectorBufferView<uint64_t>();
    lbvhTreeBuffer.m_buffer = GFXContext::create_buffer_empty();
    lbvhTreeBuffer.m_buffer->m_job = "Scene lbvh tree buffer";
    lbvhTreeBuffer.m_buffer->m_usages = rhi::BufferUsageEnum::STORAGE;

    lbvhTrailBuffer = DynamicVectorBufferView<uint64_t>();
    lbvhTrailBuffer.m_buffer = GFXContext::create_buffer_empty();
    lbvhTrailBuffer.m_buffer->m_job = "Scene lbvh trail buffer";
    lbvhTrailBuffer.m_buffer->m_usages = rhi::BufferUsageEnum::STORAGE;

    sceneDataList = DynamicVectorBufferView<SceneData>();
    sceneDataList.m_buffer = GFXContext::create_buffer_empty();
    sceneDataList.m_buffer->m_job = "Scene info buffer";
    sceneDataList.m_buffer->m_usages = rhi::BufferUsageEnum::STORAGE;
    sceneDataList.m_buffer->m_memoryCopyMode = gfx::Buffer::MemoryCopyMode::COHERENT_MAPPING;
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
    return rhi::BindingResource{ {sceneDataList.m_buffer->m_buffer.get(), 0, sceneDataList.m_buffer->m_buffer->size()} };
  }

  auto Scene::GPUScene::binding_resource_lightbvh_tree() noexcept -> rhi::BindingResource {
    return rhi::BindingResource{ {lbvhTreeBuffer.m_buffer->m_buffer.get(), 0, lbvhTreeBuffer.m_buffer->m_buffer->size()} };
  }

  auto Scene::GPUScene::binding_resource_lightbvh_trail() noexcept -> rhi::BindingResource {
    return rhi::BindingResource{ {lbvhTrailBuffer.m_buffer->m_buffer.get(), 0, lbvhTrailBuffer.m_buffer->m_buffer->size()} };
  }

  auto Scene::GPUScene::binding_resource_tlas() noexcept -> rhi::BindingResource {
    // get all TLAS pointers
    std::vector<rhi::TLAS*> tlas_ptrs;
    for (auto& tlas : tlasList) {
      tlas_ptrs.push_back(tlas);
    }
    return rhi::BindingResource{ {tlas_ptrs} };
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

  auto SceneBatch::reset() noexcept -> void {
    m_gpuSceneBatch.reset();
  }

  auto SceneBatch::update_gpu_scene_batch() noexcept -> void {
    // update scene transforms
    for (auto& scene_handle : m_scenes) 
      scene_handle->update_transform();

    // update scene meshes
    for (auto& scene_handle : m_scenes) 
      scene_handle->update_gpu_meshes(&m_gpuSceneBatch);
    m_gpuSceneBatch.positionBuffer.m_buffer->host_to_device();
    m_gpuSceneBatch.indexBuffer.m_buffer->host_to_device();
    m_gpuSceneBatch.vertexBuffer.m_buffer->host_to_device();
    m_gpuSceneBatch.materialBuffer.m_buffer->host_to_device();

    // update scene cameras
    for (auto& scene_handle : m_scenes) 
      scene_handle->update_gpu_camera(&m_gpuSceneBatch);
    m_gpuSceneBatch.cameraBuffer.m_buffer->host_to_device();

    // update scene lights
    for (auto& scene_handle : m_scenes) 
      scene_handle->update_gpu_lights(&m_gpuSceneBatch);
    m_gpuSceneBatch.lightBuffer.m_buffer->host_to_device();
    m_gpuSceneBatch.sceneDataList.m_buffer->host_to_device();
    m_gpuSceneBatch.lbvhTreeBuffer.m_buffer->host_to_device();
    m_gpuSceneBatch.lbvhTrailBuffer.m_buffer->host_to_device();

    // update scene mediums
    for (auto& scene_handle : m_scenes) 
      scene_handle->update_gpu_medium(&m_gpuSceneBatch);
    m_gpuSceneBatch.mediumPool.medium_buffer.m_buffer->host_to_device();
    m_gpuSceneBatch.mediumPool.grid_storage_buffer->host_to_device();

    // update scene BVH
    for (auto& scene_handle : m_scenes) 
      scene_handle->update_gpu_bvh(&m_gpuSceneBatch);

    for (auto& scene_handle : m_scenes) {
      // set all transform dirty flag to false
      auto node_view = scene_handle->m_registry.view<Transform>();
      for (auto [entity, _transform] : node_view.each()) {
        _transform.m_dirtyToGPU = false;
      }
    }

    m_gpuSceneBatch.geometryBuffer.m_buffer->host_to_device();
  }
}
}