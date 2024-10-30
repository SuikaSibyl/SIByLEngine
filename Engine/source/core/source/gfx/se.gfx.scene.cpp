#define DLIB_EXPORT
#include <se.gfx.hpp>
#undef DLIB_EXPORT
#include <stack>
#include <locale>
#include <codecvt>
#include <tinyparser-mitsuba.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include "ex.tinyprbrtloader.hpp"
#include <nanovdb/NanoVDB.h>
#define NANOVDB_USE_ZIP 1
#include <nanovdb/util/IO.h>


namespace se::gfx {
static std::array<uint64_t, 24> primes = {3,  5,  7,  11, 13, 17, 19, 23,
    29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};

inline auto hash_vertex(std::vector<float> const& v) noexcept -> std::uint64_t {
  std::uint64_t hash = reinterpret_cast<std::uint32_t const&>(v[0]);
  for (size_t n = 1; n < std::min(primes.size(), v.size()); ++n) {
    std::uint64_t floated = reinterpret_cast<std::uint32_t const&>(v[n]);
    hash += primes[n] * floated;
  }
  return hash;
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
/** Mesh data layout */
struct MeshDataLayout {
  /** info types in vertex */
  enum struct VertexInfo {
    POSITION,
    NORMAL,
    TANGENT,
    UV,
    COLOR,
    CUSTOM,
  };
  /** an entry of the layout */
  struct Entry {
    rhi::VertexFormat format;
    VertexInfo info;
  };
  /* the list of vertex layout */
  std::vector<Entry> layout;
  /* index format */
  rhi::IndexFormat format;
};

MeshDataLayout defaultMeshDataLayout = {{
  {rhi::VertexFormat::FLOAT32X3,
   gfx::MeshDataLayout::VertexInfo::POSITION},
  {rhi::VertexFormat::FLOAT32X3, gfx::MeshDataLayout::VertexInfo::NORMAL},
  {rhi::VertexFormat::FLOAT32X3, gfx::MeshDataLayout::VertexInfo::TANGENT},
  {rhi::VertexFormat::FLOAT32X2, gfx::MeshDataLayout::VertexInfo::UV},
},
rhi::IndexFormat::UINT32_T};

/** A setting config to guide loading of mesh resource */
struct MeshLoaderConfig {
  MeshDataLayout layout = {};
  bool usePositionBuffer = true;
  bool residentOnHost = true;
  bool residentOnDevice = false;
  bool deduplication = false;
};

MeshLoaderConfig defaultMeshLoadConfig = { defaultMeshDataLayout, true, true, false, false };

Scene::Scene() {
  gpuScene.position_buffer = GFXContext::load_buffer_empty();
  gpuScene.index_buffer = GFXContext::load_buffer_empty();
  gpuScene.vertex_buffer = GFXContext::load_buffer_empty();
  gpuScene.texcoord_buffer = GFXContext::load_buffer_empty();
  gpuScene.material_buffer = GFXContext::load_buffer_empty();
  gpuScene.light_buffer = GFXContext::load_buffer_empty();
  gpuScene.geometry_buffer = GFXContext::load_buffer_empty();
  gpuScene.camera_buffer = GFXContext::load_buffer_empty();
  gpuScene.scene_desc_buffer = GFXContext::load_buffer_empty();
  gpuScene.medium_buffer = GFXContext::load_buffer_empty();
  gpuScene.grid_storage_buffer = GFXContext::load_buffer_empty();
  gpuScene.lbvh.light_bvh_buffer = GFXContext::load_buffer_empty();
  gpuScene.lbvh.light_trail_buffer = GFXContext::load_buffer_empty();
  gpuScene.camera_buffer->host.resize(sizeof(Scene::CameraData));
  gpuScene.scene_desc_buffer->host.resize(sizeof(Scene::SceneDescription));
  gpuScene.position_buffer->usages =
    (uint32_t)rhi::BufferUsageBit::VERTEX |
    (uint32_t)rhi::BufferUsageBit::SHADER_DEVICE_ADDRESS |
    (uint32_t)rhi::BufferUsageBit::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY |
    (uint32_t)rhi::BufferUsageBit::STORAGE;
  gpuScene.index_buffer->usages =
    (uint32_t)rhi::BufferUsageBit::INDEX |
    (uint32_t)rhi::BufferUsageBit::SHADER_DEVICE_ADDRESS |
    (uint32_t)rhi::BufferUsageBit::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY |
    (uint32_t)rhi::BufferUsageBit::STORAGE;
  gpuScene.vertex_buffer->usages =
    (uint32_t)rhi::BufferUsageBit::VERTEX |
    (uint32_t)rhi::BufferUsageBit::SHADER_DEVICE_ADDRESS |
      (uint32_t)rhi::BufferUsageBit::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY |
    (uint32_t)rhi::BufferUsageBit::STORAGE;
  gpuScene.texcoord_buffer->usages =
    (uint32_t)rhi::BufferUsageBit::VERTEX |
    (uint32_t)rhi::BufferUsageBit::SHADER_DEVICE_ADDRESS |
      (uint32_t)rhi::BufferUsageBit::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY |
    (uint32_t)rhi::BufferUsageBit::STORAGE;
  gpuScene.material_buffer->usages =
    (uint32_t)rhi::BufferUsageBit::SHADER_DEVICE_ADDRESS |
    (uint32_t)rhi::BufferUsageBit::STORAGE;
  gpuScene.medium_buffer->usages =
    (uint32_t)rhi::BufferUsageBit::SHADER_DEVICE_ADDRESS |
    (uint32_t)rhi::BufferUsageBit::STORAGE;
  gpuScene.light_buffer->usages =
    (uint32_t)rhi::BufferUsageBit::SHADER_DEVICE_ADDRESS |
    (uint32_t)rhi::BufferUsageBit::STORAGE;
  gpuScene.geometry_buffer->usages =
    (uint32_t)rhi::BufferUsageBit::SHADER_DEVICE_ADDRESS |
    (uint32_t)rhi::BufferUsageBit::STORAGE;
  gpuScene.camera_buffer->usages =
    (uint32_t)rhi::BufferUsageBit::SHADER_DEVICE_ADDRESS |
    (uint32_t)rhi::BufferUsageBit::STORAGE;
  gpuScene.grid_storage_buffer->usages =
    (uint32_t)rhi::BufferUsageBit::SHADER_DEVICE_ADDRESS |
    (uint32_t)rhi::BufferUsageBit::STORAGE;
  gpuScene.scene_desc_buffer->usages =
    (uint32_t)rhi::BufferUsageBit::SHADER_DEVICE_ADDRESS |
    (uint32_t)rhi::BufferUsageBit::STORAGE;
  gpuScene.lbvh.light_bvh_buffer->usages =
    (uint32_t)rhi::BufferUsageBit::SHADER_DEVICE_ADDRESS |
    (uint32_t)rhi::BufferUsageBit::STORAGE;
  gpuScene.lbvh.light_trail_buffer->usages =
    (uint32_t)rhi::BufferUsageBit::SHADER_DEVICE_ADDRESS |
    (uint32_t)rhi::BufferUsageBit::STORAGE;
}

Scene::~Scene() {
  float a = 1.f;
}

auto Scene::createNode(std::string const& name) noexcept -> Node {
  auto entity = registry.create();
  auto node = Node{ entity, &registry };
  registry.emplace<NodeProperty>(entity, name);
  registry.emplace<Transform>(entity);

  roots.push_back(node);
  return node;
}

auto Scene::createNode(Node parent, std::string const& name) noexcept -> Node {
  auto entity = registry.create();
  auto node = Node{ entity, &registry };
  registry.emplace<NodeProperty>(entity, name);
  registry.get<NodeProperty>(parent.entity).children.push_back(node);
  return node;
}

auto Scene::destroyNode(Node const& node) noexcept -> void {
  registry.destroy(node.entity);
}

auto Scene::serialize(std::string const& path) noexcept -> void {
  tinygltf::Model model = serialize();
  tinygltf::TinyGLTF gltf;
  gltf.WriteGltfSceneToFile(&model, path,
    false, // embedImages
    false, // embedBuffers
    true, // pretty print
    false); // write binary
}

auto Scene::deserialize(tinygltf::Model& model) noexcept -> void {
  // register all the nodes first
  DeserializeData data;
  data.model = &model;
  data.nodes.resize(model.nodes.size());
  for (size_t i = 0; i < model.nodes.size(); ++i)
    data.nodes[i] = createNode(model.nodes[i].name);
  // find root nodes
  roots.clear(); // not all created nodes are roots
  auto& scene = model.scenes[0];
  for (auto root : scene.nodes) {
    roots.emplace_back(data.nodes[root]);
  }
}

auto Scene::deserialize(std::string const& path) noexcept -> void {
  tinygltf::TinyGLTF loader;
  tinygltf::Model model;
  std::string err;
  std::string warn;
  bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, path);
  if (!warn.empty()) {
    root::print::error("Scene::deserialize warn::" + warn); return;
  } if (!err.empty()) {
    root::print::error("Scene::deserialize error::" + err); return;
  } if (!ret) {
    root::print::error("Failed to parse glTF"); return;
  }
  deserialize(model);
}

auto Scene::getName() const noexcept -> char const* {
  return name.c_str();
}

auto Scene::serialize() noexcept -> tinygltf::Model {
  tinygltf::Model m;
  SerializeData data;
  data.model = &m;
  tinygltf::Scene scene;
  // register all the nodes first
  auto node_view = registry.view<NodeProperty, Transform>();
  for (auto entity : node_view)
    data.nodes.emplace(entity, data.nodes.size());
  for (auto entity : node_view) {
    int const node_id = data.nodes[entity];
    auto [prop, transform] = node_view.get<NodeProperty, Transform>(entity);
    m.nodes.emplace_back(tinygltf::Node{});
    m.nodes[node_id].name = prop.name;
    for (auto& child : prop.children)
      m.nodes[node_id].children.push_back(data.nodes[child.entity]);
    m.nodes[node_id].translation = { transform.translation.x,transform.translation.y, transform.translation.z };
    m.nodes[node_id].scale = { transform.scale.x,transform.scale.y, transform.scale.z };
    m.nodes[node_id].rotation = { transform.rotation.x,transform.rotation.y, transform.rotation.z, transform.rotation.w };
  }
  // declare root nodes
  for (auto& node : roots) {
    int node_id = data.nodes[node.entity];
    scene.nodes.push_back(node_id);
  }
  m.buffers.resize(3);
  // store position buffer
  m.buffers[0].data = gpuScene.position_buffer.get()->host;
  m.buffers[1].data = gpuScene.index_buffer.get()->host;
  m.buffers[2].data = gpuScene.vertex_buffer.get()->host;
  if (gpuScene.texcoord_buffer.get()->host.size() > 0) {
    m.buffers.resize(4);
    m.buffers[3].data = gpuScene.texcoord_buffer.get()->getHost();
  }

  auto add_view_accessor = [&](
    tinygltf::BufferView bufferView,
    tinygltf::Accessor accessor
  ) -> int {
    int view_id = m.bufferViews.size();
    m.bufferViews.push_back(bufferView);
    int accessor_id = m.accessors.size();
    accessor.bufferView = view_id;
    m.accessors.push_back(accessor);
    return accessor_id;
  };
  auto add_accessor = [&](tinygltf::Accessor accessor) -> int {
    int accessor_id = m.accessors.size();
    m.accessors.push_back(accessor);
    return accessor_id;
  };

  std::unordered_map<Material*, int32_t> material_map;
  auto add_material = [&](Material* material) -> int {
    if (material == nullptr) return -1;
    auto iter = material_map.find(material);
    if (iter != material_map.end()) {
      return iter->second;
    }
    int material_id = m.materials.size();
    tinygltf::Material gltf_material;
    gltf_material.pbrMetallicRoughness.baseColorFactor = {
        material->baseOrDiffuseColor.r, material->baseOrDiffuseColor.g,
        material->baseOrDiffuseColor.b, 1. };
    gltf_material.pbrMetallicRoughness.roughnessFactor = double(material->roughnessFactor);
    gltf_material.pbrMetallicRoughness.metallicFactor = double(material->metallicFactor);
    gltf_material.emissiveFactor = {
        material->emissiveColor.r, material->emissiveColor.g,
        material->emissiveColor.b
    };
    m.materials.push_back(gltf_material);
    material_map[material] = material_id;
    return material_id;
  };

  std::unordered_map<RUID, std::pair<MediumHandle, int32_t>> mediums_map;
  auto add_medium = [&](MediumHandle medium) -> int {
    if (!medium.get()) return -1;
    auto iter = mediums_map.find(medium.ruid);
    if (iter != mediums_map.end()) {
      return iter->second.second;
    }
    int medium_id = mediums_map.size();
    mediums_map[medium.ruid] = { medium, medium_id };
    return medium_id;
  };

  // store index buffer
  auto mesh_view = registry.view<MeshRenderer>();
  for (auto entity : mesh_view) {
    int const node_id = data.nodes[entity];
    auto& se_mesh = mesh_view.get<MeshRenderer>(entity);
    int const mesh_id = m.meshes.size();
    m.meshes.emplace_back(tinygltf::Mesh{}); auto& gltf_mesh = m.meshes.back();
    for (auto& primitive : se_mesh.mesh.get()->primitives) {
      tinygltf::Primitive gltf_primitive;
      { // position buffer
        tinygltf::BufferView bufferView;
        bufferView.buffer = 0;
        bufferView.byteOffset = primitive.baseVertex * sizeof(float) * 3 + se_mesh.mesh.get()->vertex_offset;
        bufferView.byteLength = primitive.numVertex * sizeof(float) * 3;
        bufferView.target = TINYGLTF_TARGET_ARRAY_BUFFER;
        tinygltf::Accessor accessor;
        accessor.byteOffset = 0;
        accessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        accessor.count = primitive.numVertex;
        accessor.type = TINYGLTF_TYPE_VEC3;
        accessor.maxValues = { primitive.max.x, primitive.max.y, primitive.max.z };
        accessor.minValues = { primitive.min.x, primitive.min.y, primitive.min.z };
        gltf_primitive.attributes["POSITION"] = add_view_accessor(bufferView, accessor);
      }
      { // index buffer
        tinygltf::BufferView bufferView;
        bufferView.buffer = 1;
        bufferView.byteOffset = primitive.offset * sizeof(uint32_t) + se_mesh.mesh.get()->index_offset;
        bufferView.byteLength = primitive.size * sizeof(uint32_t);
        bufferView.target = TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER;
        tinygltf::Accessor accessor;
        accessor.byteOffset = 0;
        accessor.componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;
        accessor.count = primitive.size;
        accessor.type = TINYGLTF_TYPE_SCALAR;
        gltf_primitive.indices = add_view_accessor(bufferView, accessor);
      }
      { // vertex buffer
        tinygltf::BufferView bufferView;
        bufferView.buffer = 2;
        size_t vertexByteOffset = primitive.baseVertex * sizeof(float) * 3 + se_mesh.mesh.get()->vertex_offset;
        bufferView.byteOffset = vertexByteOffset / 3 * 8;
        bufferView.byteLength = primitive.numVertex * sizeof(float) * 8;
        bufferView.byteStride = 8 * sizeof(float);
        bufferView.target = TINYGLTF_TARGET_ARRAY_BUFFER;
        int view_id = m.bufferViews.size();
        m.bufferViews.push_back(bufferView);
        // normal
        tinygltf::Accessor accessor;
        accessor.bufferView = view_id;
        accessor.byteOffset = 0;
        accessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        accessor.count = primitive.numVertex;
        accessor.type = TINYGLTF_TYPE_VEC3;
        gltf_primitive.attributes["NORMAL"] = add_accessor(accessor);
        // tangent
        accessor.byteOffset = sizeof(float) * 3;
        gltf_primitive.attributes["TANGENT"] = add_accessor(accessor);
        // coord
        accessor.byteOffset = sizeof(float) * 6;
        accessor.type = TINYGLTF_TYPE_VEC2;
        gltf_primitive.attributes["TEXCOORD_0"] = add_accessor(accessor);
      }
      // texcoord1 buffer
      if(m.buffers.size() >= 4 && m.buffers[3].data.size() > 0) {
        tinygltf::BufferView bufferView;
        bufferView.buffer = 3;
        bufferView.byteOffset = (primitive.baseVertex * sizeof(float) * 3 + se_mesh.mesh.get()->vertex_offset) * 2 / 3;
        bufferView.byteLength = primitive.numVertex * sizeof(float) * 2;
        bufferView.target = TINYGLTF_TARGET_ARRAY_BUFFER;
        tinygltf::Accessor accessor;
        accessor.byteOffset = 0;
        accessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        accessor.count = primitive.numVertex;
        accessor.type = TINYGLTF_TYPE_VEC2;
        gltf_primitive.attributes["TEXCOORD_1"] = add_view_accessor(bufferView, accessor);
      }
      gltf_primitive.mode = TINYGLTF_MODE_TRIANGLES;
      gltf_primitive.material = add_material(primitive.material.get());

      tinygltf::Value::Object primitive_extra;
      primitive_extra["exterior"] = tinygltf::Value(add_medium(primitive.exterior));
      primitive_extra["interior"] = tinygltf::Value(add_medium(primitive.interior));
      gltf_primitive.extras = tinygltf::Value(primitive_extra);

      gltf_mesh.primitives.emplace_back(gltf_primitive);
    }
    m.nodes[node_id].mesh = mesh_id;
  }

  auto camera_view = registry.view<Camera>();
  for (auto entity : camera_view) {
    int const node_id = data.nodes[entity];
    auto& se_camera = camera_view.get<Camera>(entity);
    tinygltf::Camera gltf_camera = {};
    if (se_camera.projectType == Camera::ProjectType::PERSPECTIVE) {
      gltf_camera.type = "perspective";
      gltf_camera.perspective.aspectRatio = se_camera.aspectRatio;
      gltf_camera.perspective.yfov = se::radians(se_camera.yfov);
      gltf_camera.perspective.znear = se_camera.znear;
      gltf_camera.perspective.zfar = se_camera.zfar;
    }
    int const camera_id = m.cameras.size();
    m.cameras.emplace_back(gltf_camera);
    m.nodes[node_id].camera = camera_id;
  }

  tinygltf::Value::Object model_extra;


  // serialize the medium map
  if (mediums_map.size() > 0) {
    tinygltf::Value::Object medium_extra;
    int medium_buffer_index = -1;
    std::vector<unsigned char> medium_buffer;
    auto allocate_buffer = [&](int size) -> std::pair<float*, int> {
      int offset = medium_buffer.size();
      medium_buffer.resize(offset + size * sizeof(float));
      return { (float*)&medium_buffer[offset] , offset / sizeof(float) };
    };

    tinygltf::Value::Array mediums(mediums_map.size());
    for (auto& pair : mediums_map) {
      int index = pair.second.second;
      MediumHandle medium = pair.second.first;
      tinygltf::Value::Object media_extra;
      media_extra["type"] = tinygltf::Value(int(medium->packet.type));
      if (medium->packet.type == Medium::MediumType::RGBGridMedium) {
        media_extra["scale"] = tinygltf::Value(medium->packet.scale);
        media_extra["bound_min_x"] = tinygltf::Value(medium->packet.bound_min.x);
        media_extra["bound_min_y"] = tinygltf::Value(medium->packet.bound_min.y);
        media_extra["bound_min_z"] = tinygltf::Value(medium->packet.bound_min.z);
        media_extra["bound_max_x"] = tinygltf::Value(medium->packet.bound_max.x);
        media_extra["bound_max_y"] = tinygltf::Value(medium->packet.bound_max.y);
        media_extra["bound_max_z"] = tinygltf::Value(medium->packet.bound_max.z);
        media_extra["grid_nx"] = tinygltf::Value(medium->packet.density_nxyz.x);
        media_extra["grid_ny"] = tinygltf::Value(medium->packet.density_nxyz.y);
        media_extra["grid_nz"] = tinygltf::Value(medium->packet.density_nxyz.z);
        media_extra["aniso_x"] = tinygltf::Value(medium->packet.aniso.x);
        media_extra["aniso_y"] = tinygltf::Value(medium->packet.aniso.y);
        media_extra["aniso_z"] = tinygltf::Value(medium->packet.aniso.z);

        media_extra["o2w"] = tinygltf::Value(tinygltf::Value::Array{
          tinygltf::Value(medium->packet.geometryTransform.matrix[0][0]),
          tinygltf::Value(medium->packet.geometryTransform.matrix[0][1]),
          tinygltf::Value(medium->packet.geometryTransform.matrix[0][2]),
          tinygltf::Value(medium->packet.geometryTransform.matrix[0][3]),
          tinygltf::Value(medium->packet.geometryTransform.matrix[1][0]),
          tinygltf::Value(medium->packet.geometryTransform.matrix[1][1]),
          tinygltf::Value(medium->packet.geometryTransform.matrix[1][2]),
          tinygltf::Value(medium->packet.geometryTransform.matrix[1][3]),
          tinygltf::Value(medium->packet.geometryTransform.matrix[2][0]),
          tinygltf::Value(medium->packet.geometryTransform.matrix[2][1]),
          tinygltf::Value(medium->packet.geometryTransform.matrix[2][2]),
          tinygltf::Value(medium->packet.geometryTransform.matrix[2][3]),
        });
        media_extra["w2o"] = tinygltf::Value(tinygltf::Value::Array{
          tinygltf::Value(medium->packet.geometryTransformInverse.matrix[0][0]),
          tinygltf::Value(medium->packet.geometryTransformInverse.matrix[0][1]),
          tinygltf::Value(medium->packet.geometryTransformInverse.matrix[0][2]),
          tinygltf::Value(medium->packet.geometryTransformInverse.matrix[0][3]),
          tinygltf::Value(medium->packet.geometryTransformInverse.matrix[1][0]),
          tinygltf::Value(medium->packet.geometryTransformInverse.matrix[1][1]),
          tinygltf::Value(medium->packet.geometryTransformInverse.matrix[1][2]),
          tinygltf::Value(medium->packet.geometryTransformInverse.matrix[1][3]),
          tinygltf::Value(medium->packet.geometryTransformInverse.matrix[2][0]),
          tinygltf::Value(medium->packet.geometryTransformInverse.matrix[2][1]),
          tinygltf::Value(medium->packet.geometryTransformInverse.matrix[2][2]),
          tinygltf::Value(medium->packet.geometryTransformInverse.matrix[2][3]),
          });
        //medium_handle->packet.geometryTransform
        int sigma_a_size = int(medium->density->values.size());
        auto [sigma_a_tgt, offset] = allocate_buffer(sigma_a_size);
        memcpy(sigma_a_tgt, medium->density->values.data(), sizeof(float)* sigma_a_size);
        media_extra["sigma_a_size"] = tinygltf::Value(sigma_a_size);
        media_extra["sigma_a_offset"] = tinygltf::Value(offset);
        int sigma_s_size = int(medium->temperatureGrid->values.size());
        auto [sigma_s_tgt, offset_s] = allocate_buffer(sigma_s_size);
        memcpy(sigma_s_tgt, medium->temperatureGrid->values.data(), sizeof(float)* sigma_s_size);
        media_extra["sigma_s_offset"] = tinygltf::Value(offset_s);
        media_extra["sigma_s_size"] = tinygltf::Value(sigma_s_size);
      }
      mediums[index] = tinygltf::Value(media_extra);
    }
    // also upload buffer
    if (medium_buffer.size() > 0) {
      medium_buffer_index = m.buffers.size();
      m.buffers.emplace_back();
      m.buffers[medium_buffer_index].data = medium_buffer;
    }

    medium_extra["buffer_id"] = tinygltf::Value(medium_buffer_index);
    medium_extra["mediums"] = tinygltf::Value(mediums);
    model_extra["medium"] = tinygltf::Value(medium_extra);
  }

  m.extras = tinygltf::Value(model_extra);
  m.scenes.emplace_back(scene);
  return m;
}

auto Scene::updateTransform() noexcept -> void {
  std::stack<float> is_odd_scaling;
  std::stack<se::mat4> global_transfrom;
  is_odd_scaling.push(1.f);
  global_transfrom.push(se::mat4{});
  auto node_view = registry.view<NodeProperty, Transform>();

  std::function<void(ex::entity)> fn_process_per_node = [&](ex::entity entity) {
    auto [prop, transform] = node_view.get<NodeProperty, Transform>(entity);
    se::mat4 objectMat = global_transfrom.top();
    float oddScaling = is_odd_scaling.top();
    objectMat = objectMat * transform.local();
    oddScaling *= transform.scale.x * transform.scale.y * transform.scale.z;
    // update transform information
    if (oddScaling != 0) oddScaling / std::abs(oddScaling);
    transform.oddScaling = oddScaling >= 0;
    transform.global = objectMat;
    // process children nodes
    global_transfrom.push(objectMat);
    is_odd_scaling.push(oddScaling);
    for (auto child : prop.children)
      fn_process_per_node(child.entity);
    is_odd_scaling.pop();
    global_transfrom.pop();
  };

  for (auto& root : roots) {
    fn_process_per_node(root.entity);
  }
}

Scene::CameraData::CameraData(Camera const& camera, Transform const& transform) {
  nearZ = camera.znear;
  farZ = camera.zfar;
  posW = transform.translation;
  target = transform.translation + transform.forward();
  viewMat = se::transpose(se::lookAt(posW, target, se::vec3(0, 1, 0)).m);
  invViewMat = se::inverse(viewMat);
  projMat = se::transpose(camera.getProjectionMat());
  invProjMat = se::inverse(projMat);
  viewProjMat = viewMat * projMat;
  invViewProj = se::inverse(viewProjMat);
  // Ray tracing related vectors
  focalDistance = 1;
  aspectRatio = camera.aspectRatio;
  up = se::vec3(0, 1, 0);
  cameraW = se::normalize(target - posW) * focalDistance;
  cameraU = se::normalize(se::cross(cameraW, up));
  cameraV = se::normalize(se::cross(cameraU, cameraW));
  const float ulen = focalDistance * std::tan(se::radians(camera.yfov) * 0.5f) * aspectRatio;
  cameraU *= ulen;
  const float vlen = focalDistance * std::tan(se::radians(camera.yfov) * 0.5f);
  cameraV *= vlen;
  jitterX = 0;
  jitterY = 0;
  clipToWindowScale = se::vec2(0.5f * camera.width, -0.5f * camera.height);
  clipToWindowBias = se::vec2(0.f) + se::vec2(camera.width, camera.height) * 0.5f;
  rectArea = 4 * ulen * vlen / (focalDistance * focalDistance);
}

auto Scene::getSceneLightCounts() noexcept -> int {
  return gpuScene.light_buffer->host.size() / sizeof(Light::LightPacket);
}

auto Scene::useEditorCameraView(Transform* transfrom, Camera* camera) noexcept -> void {
  editorInfo.viewport_transfrom = transfrom;
  editorInfo.viewport_camera = camera;
}

auto Scene::getEditorActiveCameraIndex() noexcept -> int {
  return editorInfo.active_camera_index;
}

auto Scene::getDefaultResolution() noexcept -> ivec2 {
  return resolution;
}

auto Scene::updateGPUScene() noexcept -> void {
  // update camera information
  size_t camera_index = 0;
  if ((dirtyFlags & (uint64_t)DirtyFlagBit::Camera) != 0) {
    auto invalid_camera = [&](Transform& transform, Camera& camera) {
      CameraData camData(camera, transform);
      if (gpuScene.camera_buffer->host.size() < (camera_index + 1) * sizeof(CameraData)) {
        gpuScene.camera_buffer->host.resize((camera_index + 1) * sizeof(CameraData));
      }
      if (camera.medium.get() != nullptr) {
        camData.mediumID = gpuScene.try_fetch_medium_index(camera.medium);
      }
      memcpy(&gpuScene.camera_buffer->host[(camera_index++) * sizeof(CameraData)], &camData, sizeof(CameraData));
      gpuScene.camera_buffer->host_stamp++;
    };
    auto view_transform_camera = registry.view<Transform, Camera>();
    for (auto entity : view_transform_camera) {
      auto [transform, camera] = view_transform_camera.get<Transform, Camera>(entity);
      invalid_camera(transform, camera);
    }
    if (editorInfo.viewport_transfrom != nullptr && 
        editorInfo.viewport_camera != nullptr) {
      invalid_camera(*editorInfo.viewport_transfrom, *editorInfo.viewport_camera);
    }
  }

  // update geometry information
  auto node_mesh_view = registry.view<Transform, MeshRenderer>();
  if ((dirtyFlags & (uint64_t)DirtyFlagBit::Geometry) != 0) {
    size_t geometry_index = 0;
    for (auto entity : node_mesh_view) {
      auto [transform, mesh_renderer] = node_mesh_view.get<Transform, MeshRenderer>(entity);
      std::vector<int> indices(mesh_renderer.mesh->primitives.size()); int i = 0;
      if (mesh_renderer.mesh->custom_primitives.size() > 0) {
        for (auto& primitive : mesh_renderer.mesh->custom_primitives) {
          GeometryDrawData geometry;
          geometry.vertexOffset = 0;
          geometry.indexOffset = 0;
          geometry.indexSize = 0;
          geometry.geometryTransform = transform.global;
          geometry.geometryTransformInverse = se::inverse(transform.global);
          geometry.oddNegativeScaling = transform.oddScaling;
          if (primitive.material.get())
            geometry.materialID = gpuScene.try_fetch_material_index(primitive.material);
          geometry.primitiveType = primitive.primitive_type;
          geometry.lightID = -1;
          geometry.mediumIDInterior = -1;
          geometry.mediumIDExterior = -1;
          if (primitive.exterior.get())
            geometry.mediumIDExterior = gpuScene.try_fetch_medium_index(primitive.exterior);
          if (primitive.interior.get())
            geometry.mediumIDInterior = gpuScene.try_fetch_medium_index(primitive.interior);
          if (gpuScene.geometry_buffer->host.size() < (geometry_index + 1) * sizeof(GeometryDrawData)) {
            gpuScene.geometry_buffer->host.resize((geometry_index + 1) * sizeof(GeometryDrawData));
          }
          memcpy(&gpuScene.geometry_buffer->host[geometry_index * sizeof(GeometryDrawData)], &geometry, sizeof(GeometryDrawData));
          indices[i++] = geometry_index; geometry_index++;
        }
      }
      else {
        for (auto& primitive : mesh_renderer.mesh->primitives) {
          GeometryDrawData geometry;
          geometry.vertexOffset = primitive.baseVertex + mesh_renderer.mesh->vertex_offset / (sizeof(float) * 3);
          geometry.indexOffset = primitive.offset + mesh_renderer.mesh->index_offset / sizeof(uint32_t);
          geometry.indexSize = primitive.size;
          geometry.geometryTransform = transform.global;
          geometry.geometryTransformInverse = se::inverse(transform.global);
          geometry.oddNegativeScaling = transform.oddScaling;
          if (primitive.material.get())
            geometry.materialID = gpuScene.try_fetch_material_index(primitive.material);
          geometry.primitiveType = 0;
          geometry.lightID = -1;
          geometry.mediumIDInterior = -1;
          geometry.mediumIDExterior = -1;
          if (primitive.exterior.get())
            geometry.mediumIDExterior = gpuScene.try_fetch_medium_index(primitive.exterior);
          if (primitive.interior.get())
            geometry.mediumIDInterior = gpuScene.try_fetch_medium_index(primitive.interior);
          if (gpuScene.geometry_buffer->host.size() < (geometry_index + 1) * sizeof(GeometryDrawData)) {
            gpuScene.geometry_buffer->host.resize((geometry_index + 1) * sizeof(GeometryDrawData));
          }
          memcpy(&gpuScene.geometry_buffer->host[geometry_index * sizeof(GeometryDrawData)], &geometry, sizeof(GeometryDrawData));
          indices[i++] = geometry_index; geometry_index++;
        }
      }
      gpuScene.geometry_loc_index[mesh_renderer.mesh.ruid] = indices;
    }
  }
  std::span<GeometryDrawData> geometry_array = {
    (GeometryDrawData*)gpuScene.geometry_buffer->host.data(),
    gpuScene.geometry_buffer->host.size() / sizeof(GeometryDrawData),
  };

  auto node_light_view = registry.view<Transform, Light>();
  if ((dirtyFlags & (uint64_t)DirtyFlagBit::Light) != 0) {
    size_t light_index = 0;
    for (auto entity : node_light_view) {
      auto [transform, light] = node_light_view.get<Transform, Light>(entity);
      switch (light.type) {
      case Light::LightType::MESH_PRIMITIVE: {
        MeshHandle& mesh = registry.get<MeshRenderer>(entity).mesh;
        std::vector<int>& indices = gpuScene.geometry_loc_index[mesh.ruid];
        if (mesh->custom_primitives.size() > 0) {
          for (int i = 0; i < light.primitives.size(); ++i) {
            int primitive_index = light.primitives[i];
            int geometry_index = indices[primitive_index];
            GeometryDrawData& geometry = geometry_array[geometry_index];
            Light::LightPacket packet;
            const vec3 emissive = mesh->custom_primitives[primitive_index].material->emissiveColor;
            const vec3 yuv = {
              0.299f * emissive.r + 0.587f * emissive.g + 0.114f * emissive.b,
              -0.14713f * emissive.r - 0.28886f * emissive.g + 0.436f * emissive.b,
              0.615f * emissive.r - 0.51499f * emissive.g - 0.10001f * emissive.b,
            };
            int type = mesh->custom_primitives[primitive_index].primitive_type;
            if (type == 1) {
              packet.light_type = int(Light::LightType::SPHERE);
              vec3 x1 = mul(mat4(geometry.geometryTransform), vec4{ 1, 0, 0, 1 }).xyz();
              vec3 x0 = mul(mat4(geometry.geometryTransform), vec4{ 0, 0, 0, 1 }).xyz();
              float radius = se::length(x1 - x0);
              packet.uintscalar_0 = 0;
              packet.uintscalar_1 = geometry_index;
              bounds3 bound;
              bound.pMin = x0 - vec3{ radius };
              bound.pMax = x0 + vec3{ radius };
              float area = 4 * k_pi * radius * radius;
              vec3 power = yuv * k_pi * area;
              packet.floatvec_0 = { power , 0 };
              packet.floatvec_1 = { bound.pMin, 0 };
              packet.floatvec_2 = { bound.pMax, 0 };
            }
            else if (type == 2) {
              packet.light_type = int(Light::LightType::RECTANGLE);
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
              vec3 power = yuv * k_pi * area;
              packet.floatvec_0 = { power , 0 };
              packet.floatvec_1 = { bound.pMin, 0 };
              packet.floatvec_2 = { bound.pMax, 0 };
            }
            else if (type == 3) {

            }
            //if()


            int light_index = gpuScene.light_buffer->host.size() / sizeof(Light::LightPacket);
            geometry.lightID = light_index;
            if (gpuScene.light_buffer->host.size() < (light_index + 1) * sizeof(Light::LightPacket)) {
              gpuScene.light_buffer->host.resize((light_index + 1) * sizeof(Light::LightPacket));
            }
            memcpy(&gpuScene.light_buffer->host[light_index * sizeof(Light::LightPacket)],
              (float*)&packet, 1 * sizeof(Light::LightPacket));
          }
        }
        else {
          for (int i = 0; i < light.primitives.size(); ++i) {
            int primitive_index = light.primitives[i];
            int geometry_index = indices[primitive_index];
            GeometryDrawData& geometry = geometry_array[geometry_index];
            std::vector<Light::LightPacket> packets(geometry.indexSize / 3);
            const vec3 emissive = mesh->primitives[primitive_index].material->emissiveColor;
            const vec3 yuv = {
              0.299f * emissive.r + 0.587f * emissive.g + 0.114f * emissive.b,
              -0.14713f * emissive.r - 0.28886f * emissive.g + 0.436f * emissive.b,
              0.615f * emissive.r - 0.51499f * emissive.g - 0.10001f * emissive.b,
            };
            for (int j = 0; j < geometry.indexSize / 3; j++) {
              packets[j].light_type = int(Light::LightType::MESH_PRIMITIVE);
              packets[j].uintscalar_0 = j;
              packets[j].uintscalar_1 = geometry_index;
              // todo (twoSided ? 2 : 1)
              uvec3 indices = gpuScene.fetch_triangle_indices(geometry, j);
              vec3 v0 = gpuScene.fetch_vertex_position(indices[0] + int(geometry.vertexOffset));
              vec3 v1 = gpuScene.fetch_vertex_position(indices[1] + int(geometry.vertexOffset));
              vec3 v2 = gpuScene.fetch_vertex_position(indices[2] + int(geometry.vertexOffset));
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
              vec3 n0 = gpuScene.fetch_vertex_normal(indices[0] + int(geometry.vertexOffset));
              vec3 n1 = gpuScene.fetch_vertex_normal(indices[1] + int(geometry.vertexOffset));
              vec3 n2 = gpuScene.fetch_vertex_normal(indices[2] + int(geometry.vertexOffset));
              n0 = mul(mat4(geometry.geometryTransformInverse), { n0, 0 }).xyz();
              n1 = mul(mat4(geometry.geometryTransformInverse), { n1, 0 }).xyz();
              n2 = mul(mat4(geometry.geometryTransformInverse), { n2, 0 }).xyz();
              //normal3 ns = normalize(n0 + n1 + n2);
              //n = faceForward(n, ns);
              n *= geometry.oddNegativeScaling;

              vec3 power = yuv * k_pi * area;
              packets[j].floatvec_0 = { power , n.x };
              packets[j].floatvec_1 = { bound.pMin, n.y };
              packets[j].floatvec_2 = { bound.pMax, n.z };
            }
            int light_index = gpuScene.light_buffer->host.size() / sizeof(Light::LightPacket);
            geometry.lightID = light_index;
            if (gpuScene.light_buffer->host.size() < (light_index + packets.size()) * sizeof(Light::LightPacket)) {
              gpuScene.light_buffer->host.resize((light_index + packets.size()) * sizeof(Light::LightPacket));
            }
            memcpy(&gpuScene.light_buffer->host[light_index * sizeof(Light::LightPacket)],
              (float*)packets.data(), packets.size() * sizeof(Light::LightPacket));
          }
        }
        gpuScene.light_buffer->host_stamp++;
        break;
      }
      default: break;
      }
    }
    // build light bvh
    gpuScene.build_light_bvh();
  }

  if ((dirtyFlags & (uint64_t)DirtyFlagBit::Material) != 0) {
    for (auto& pair : gpuScene.material_loc_index) {
      if (pair.second.second->isDirty) {
          pair.second.second->isDirty = false;
          Material::MaterialPacket pack = pair.second.second->getDataPacket();
          // load material
          if (pair.second.second->basecolorTex.get())
            pack.base_tex = gpuScene.try_fetch_texture_index(pair.second.second->basecolorTex);
          memcpy((float*)&(gpuScene.material_buffer->host[sizeof(Material::MaterialPacket) * 
            pair.second.first]), &pack, sizeof(pack));
          gpuScene.material_buffer->host_stamp++;
      }
    }
  }

  if ((dirtyFlags & (uint64_t)DirtyFlagBit::Medium) != 0) {
    for (auto& pair : gpuScene.medium_loc_index) {
      if (pair.second.second->isDirty) {
        pair.second.second->isDirty = false;
        Medium::MediumPacket pack = pair.second.second->packet;
        memcpy((float*)&(gpuScene.medium_buffer->host[sizeof(Medium::MediumPacket) *
          pair.second.first]), &pack, sizeof(pack));
        gpuScene.medium_buffer->host_stamp++;
      }
    }
  }

  // always update ?
  {
    SceneDescription desc;
    desc.light_bound_min = gpuScene.lbvh.all_light_bounds.pMin;
    desc.light_bound_max = gpuScene.lbvh.all_light_bounds.pMax;
    desc.max_light_count = getSceneLightCounts();
    desc.active_camera_index = getEditorActiveCameraIndex();
    memcpy((SceneDescription*)gpuScene.scene_desc_buffer->host.data(), 
      &desc, sizeof(SceneDescription));
    gpuScene.scene_desc_buffer->host_stamp++;
  }

  gpuScene.position_buffer->hostToDevice();
  gpuScene.index_buffer->hostToDevice();
  gpuScene.vertex_buffer->hostToDevice();
  gpuScene.geometry_buffer->hostToDevice();
  gpuScene.camera_buffer->hostToDevice();
  gpuScene.texcoord_buffer->hostToDevice();
  gpuScene.material_buffer->hostToDevice();
  gpuScene.light_buffer->hostToDevice();
  gpuScene.medium_buffer->hostToDevice();
  gpuScene.grid_storage_buffer->hostToDevice();
  gpuScene.scene_desc_buffer->hostToDevice();

  // also update the light bvh
  gpuScene.lbvh.light_bvh_buffer->hostToDevice();
  gpuScene.lbvh.light_trail_buffer->hostToDevice();
    
  // also update the ray tracing data structures
  gpuScene.tlas.desc = {};
  if ((dirtyFlags & (uint64_t)DirtyFlagBit::Geometry) != 0) {
    for (auto entity : node_mesh_view) {
      auto [transform, mesh_renderer] = node_mesh_view.get<Transform, MeshRenderer>(entity);
      mesh_renderer.blasInstance.resize(mesh_renderer.mesh->primitives.size());
      mesh_renderer.uvblasInstance.resize(mesh_renderer.mesh->primitives.size());
      size_t primitive_index = 0;
      // if there are custom primitive we do not use traingles
      if (mesh_renderer.mesh->custom_primitives.size() > 0) {
        for (auto& primitive : mesh_renderer.mesh->custom_primitives) {
          primitive.back_blas = std::move(primitive.prim_blas);
          primitive.blasDesc.allowCompaction = true;
          primitive.blasDesc.customGeometries.push_back(rhi::BLASCustomGeometry{
            rhi::AffineTransformMatrix{},
            std::vector<se::bounds3>{se::bounds3{primitive.min, primitive.max}},
            (uint32_t)rhi::BLASGeometryFlagBits::NO_DUPLICATE_ANY_HIT_INVOCATION
            | (uint32_t)rhi::BLASGeometryFlagBits::OPAQUE_GEOMETRY,
          });
          primitive.prim_blas = GFXContext::device->createBLAS(primitive.blasDesc);
          // update the instance of the mesh resource
          rhi::BLASInstance& instance = mesh_renderer.blasInstance[primitive_index++];
          instance.blas = primitive.prim_blas.get();
          instance.transform = transform.global;
          instance.instanceCustomIndex = primitive.primitive_type; // geometry_start
          instance.instanceShaderBindingTableRecordOffset = 0;
          gpuScene.tlas.desc.instances.push_back(instance);
        }
      }
      // otherwise we push the triangles to BLAS
      else {
        se::root::print::log("scene :: update gpu-scene :: start create triangle BLAS " + registry.get<NodeProperty>(entity).name);

        for (auto& primitive : mesh_renderer.mesh->primitives) {
          primitive.back_blas = std::move(primitive.prim_blas);
          primitive.blasDesc.allowCompaction = true;
          primitive.blasDesc.triangleGeometries.push_back(rhi::BLASTriangleGeometry{
            gpuScene.position_buffer->buffer.get(),
            gpuScene.index_buffer->buffer.get(),
            rhi::IndexFormat::UINT32_T,
            uint32_t(primitive.numVertex - 1),
            uint32_t(primitive.baseVertex + mesh_renderer.mesh->vertex_offset / (sizeof(float) * 3)),
            uint32_t(primitive.size / 3),
            uint32_t(primitive.offset * sizeof(uint32_t) + mesh_renderer.mesh->index_offset),
            rhi::AffineTransformMatrix{},
            (uint32_t)rhi::BLASGeometryFlagBits::NO_DUPLICATE_ANY_HIT_INVOCATION
            | (uint32_t)rhi::BLASGeometryFlagBits::OPAQUE_GEOMETRY,
            0 });
          primitive.prim_blas = GFXContext::device->createBLAS(primitive.blasDesc);
          // update the instance of the mesh resource
          rhi::BLASInstance& instance = mesh_renderer.blasInstance[primitive_index++];
          instance.blas = primitive.prim_blas.get();
          instance.transform = transform.global;
          instance.instanceCustomIndex = 0; // geometry_start
          instance.instanceShaderBindingTableRecordOffset = 0;
          gpuScene.tlas.desc.instances.push_back(instance);

          if (gpuSceneSetting.useTexcoordTLAS) {
            // create the blas for sampling from the texture coordinates.
            primitive.back_uv_blas = std::move(primitive.prim_uv_blas);
            primitive.uvblasDesc.allowCompaction = true;
            primitive.uvblasDesc.triangleGeometries.push_back(rhi::BLASTriangleGeometry{
              gpuScene.texcoord_buffer->buffer.get(),
              gpuScene.index_buffer->buffer.get(),
              rhi::IndexFormat::UINT32_T,
              uint32_t(primitive.numVertex - 1),
              uint32_t(primitive.baseVertex + mesh_renderer.mesh->vertex_offset / (sizeof(float) * 3)),
              uint32_t(primitive.size / 3),
              uint32_t(primitive.offset * sizeof(uint32_t) + mesh_renderer.mesh->index_offset),
              rhi::AffineTransformMatrix{},
              (uint32_t)rhi::BLASGeometryFlagBits::NO_DUPLICATE_ANY_HIT_INVOCATION,
              0, 2 * sizeof(float), 0, rhi::BLASTriangleGeometry::VertexFormat::RG32 });
            primitive.prim_uv_blas = GFXContext::device->createBLAS(primitive.uvblasDesc);
            // update the instance of the mesh resource
            rhi::BLASInstance& uvinstance = mesh_renderer.uvblasInstance[primitive_index - 1];
            uvinstance.blas = primitive.prim_uv_blas.get();
            uvinstance.transform = {};
            uvinstance.instanceCustomIndex = 0; // geometry_start
            uvinstance.instanceShaderBindingTableRecordOffset = 0;
            gpuScene.tlas.uvdesc.instances.push_back(uvinstance);
          }
        }
        gfx::GFXContext::getDevice()->waitIdle();
        se::root::print::log("scene :: update gpu-scene :: end create triangle BLAS ");
      }
    }

    gpuScene.tlas.prim = GFXContext::device->createTLAS(gpuScene.tlas.desc);
    gpuScene.tlas.uvprim = GFXContext::device->createTLAS(gpuScene.tlas.uvdesc);
  }

  se::root::print::log("scene :: update gpu-scene :: end create accelerate-structures ");
  // set the dirty flag to 0
  dirtyFlags = 0;
}

auto Scene::getGPUScene() noexcept -> GPUScene* { return &gpuScene; }

auto Scene::createTexcoord(TexcoordKind kind) noexcept -> void {
  if (kind == TexcoordKind::CopyCoord0) {
    std::vector<unsigned char>& host = gpuScene.vertex_buffer->getHost();
    std::span<float> span(reinterpret_cast<float*>(host.data()), host.size() / sizeof(float));
    std::vector<unsigned char>& texcoord = gpuScene.texcoord_buffer->getHost();
    texcoord.resize(span.size() / 8 * 2 * sizeof(float));
    std::span<float> target_span(reinterpret_cast<float*>(texcoord.data()), texcoord.size() / sizeof(float));
    for (size_t i = 0; i * 8 < span.size(); ++i) {
      target_span[i * 2 + 0] = span[i * 8 + 6];
      target_span[i * 2 + 1] = span[i * 8 + 7];
    }
 }
}

auto Scene::GPUScene::bindingResourcePosition() noexcept -> rhi::BindingResource {
  return rhi::BindingResource{ {position_buffer->buffer.get(), 0, position_buffer->buffer->size()} };
}

auto Scene::GPUScene::bindingResourceIndex() noexcept -> rhi::BindingResource {
  return rhi::BindingResource{ {index_buffer->buffer.get(), 0, index_buffer->buffer->size()} };
}

auto Scene::GPUScene::bindingResourceVertex() noexcept -> rhi::BindingResource {
  return rhi::BindingResource{ {vertex_buffer->buffer.get(), 0, vertex_buffer->buffer->size()} };
}

auto Scene::GPUScene::bindingResourceGeometry() noexcept -> rhi::BindingResource {
  return rhi::BindingResource{ {geometry_buffer->buffer.get(), 0, geometry_buffer->buffer->size()} };
}

auto Scene::GPUScene::bindingResourceCamera() noexcept -> rhi::BindingResource {
  return rhi::BindingResource{ {camera_buffer->buffer.get(), 0, camera_buffer->buffer->size()} };
}

auto Scene::GPUScene::bindingResourceMaterial() noexcept -> rhi::BindingResource {
  return rhi::BindingResource{ {material_buffer->buffer.get(), 0, material_buffer->buffer->size()} };
}

auto Scene::GPUScene::bindingResourceLight() noexcept -> rhi::BindingResource {
    return rhi::BindingResource{ {light_buffer->buffer.get(), 0, light_buffer->buffer->size()} };
}

auto Scene::GPUScene::bindingResourceMedium() noexcept -> rhi::BindingResource {
  return rhi::BindingResource{ {medium_buffer->buffer.get(), 0, medium_buffer->buffer->size()} };
}

auto Scene::GPUScene::bindingResourceGrids() noexcept -> rhi::BindingResource {
  return rhi::BindingResource{ {grid_storage_buffer->buffer.get(), 0, grid_storage_buffer->buffer->size()} };
}

auto Scene::GPUScene::bindingResourceLightBVH() noexcept -> rhi::BindingResource {
  return rhi::BindingResource{ {lbvh.light_bvh_buffer->buffer.get(), 0, lbvh.light_bvh_buffer->buffer->size()} };
}

auto Scene::GPUScene::bindingResourceLightTrail() noexcept -> rhi::BindingResource {
  return rhi::BindingResource{ {lbvh.light_trail_buffer->buffer.get(), 0, lbvh.light_trail_buffer->buffer->size()} };
}

auto Scene::GPUScene::bindingSceneDescriptor() noexcept -> rhi::BindingResource {
  return rhi::BindingResource{ {scene_desc_buffer->buffer.get(), 0, scene_desc_buffer->buffer->size()} };
}

auto Scene::GPUScene::bindingResourceTextures() noexcept -> rhi::BindingResource {
  return rhi::BindingResource{ texp.prim_t, texp.prim_s };
}

auto Scene::GPUScene::bindingResourceTLAS() noexcept -> rhi::BindingResource {
  return rhi::BindingResource{ tlas.prim.get() };
}

auto Scene::GPUScene::bindingResourceTLASPrev() noexcept -> rhi::BindingResource {
  return rhi::BindingResource{ tlas.back.get() };
}

auto Scene::GPUScene::bindingResourceUvTLAS() noexcept -> rhi::BindingResource {
  return rhi::BindingResource{ tlas.uvprim.get() };
}

auto Scene::GPUScene::getPositionBuffer() noexcept -> BufferHandle {
  return position_buffer;
}

auto Scene::GPUScene::getIndexBuffer() noexcept -> BufferHandle {
  return index_buffer;
}

auto Scene::GPUScene::try_fetch_texture_index(TextureHandle& handle) noexcept -> int {
  auto iter = texture_loc_index.find(handle.ruid);
  if (iter == texture_loc_index.end()) {
    int index = texture_loc_index.size();
    texture_loc_index[handle.ruid] = { index, handle };
    texp.prim_t.push_back(handle->getSRV(0, 1, 0, 1));
    SamplerHandle sampler = GFXContext::create_sampler_desc(rhi::SamplerDescriptor{});
    texp.prim_s.push_back(sampler.get());
    return index;
  }
  return iter->second.first;
}

auto Scene::GPUScene::try_fetch_material_index(MaterialHandle& handle) noexcept -> int {
  auto iter = material_loc_index.find(handle.ruid);
  if (iter == material_loc_index.end()) {
    int index = material_loc_index.size();
    material_loc_index[handle.ruid] = { index, handle };
    material_buffer->host.resize(sizeof(Material::MaterialPacket) * (index + 1));
    Material::MaterialPacket pack = handle->getDataPacket();
    if (handle->basecolorTex.get())
      pack.base_tex = try_fetch_texture_index(handle->basecolorTex);
    memcpy((float*)&(material_buffer->host[sizeof(Material::MaterialPacket) * index]), &pack, sizeof(pack));
    material_buffer->host_stamp++;
    return index;
  }
  return iter->second.first;
}

auto Scene::GPUScene::try_fetch_medium_index(MediumHandle& handle) noexcept -> int {
  auto iter = medium_loc_index.find(handle.ruid);
  if (iter == medium_loc_index.end()) {
    int index = medium_loc_index.size();
    medium_loc_index[handle.ruid] = { index, handle };

    // upload density grid
    if (handle->density.has_value()) {
      handle->packet.bound_min = handle->density->bounds.pMin;
      handle->packet.bound_max = handle->density->bounds.pMax;
      handle->packet.density_nxyz = ivec3{ handle->density->nx, handle->density->ny, handle->density->nz };
      int size = handle->density->values.size();
      int offset = grid_storage_buffer->host.size() / sizeof(float);
      offset = int((offset + 63) / 64) * 64;
      handle->packet.density_offset = offset;
      grid_storage_buffer->host.resize(sizeof(float) * (offset + size));
      memcpy(&(grid_storage_buffer->host[offset * sizeof(float)]),
        handle->density->values.data(), size * sizeof(float));
      grid_storage_buffer->host_stamp++;
    }

    // upload temperature grid
    if (handle->temperatureGrid.has_value()) {
      handle->packet.temperature_nxyz = ivec3{ handle->temperatureGrid->nx, handle->temperatureGrid->ny, handle->temperatureGrid->nz };
      int size = handle->temperatureGrid->values.size();
      int offset = grid_storage_buffer->host.size() / sizeof(float);
      offset = int((offset + 63) / 64) * 64;
      handle->packet.temperature_offset = offset;
      handle->packet.temperature_bound_min = handle->temperatureGrid->bounds.pMin;
      handle->packet.temperature_bound_max = handle->temperatureGrid->bounds.pMax;
      grid_storage_buffer->host.resize(sizeof(float) * (offset + size));
      memcpy(&(grid_storage_buffer->host[offset * sizeof(float)]),
        handle->temperatureGrid->values.data(), size * sizeof(float));
      grid_storage_buffer->host_stamp++;
    }

    // upload majorant grid
    if (handle->majorantGrid.has_value()) {
      handle->packet.majorant_nxyz = handle->majorantGrid->res;
      int size = handle->majorantGrid->voxels.size();
      int offset = grid_storage_buffer->host.size() / sizeof(float);
      offset = int((offset + 63) / 64) * 64;
      handle->packet.majorant_offset = offset;
      grid_storage_buffer->host.resize(sizeof(float) * (offset + size));
      memcpy(&(grid_storage_buffer->host[offset * sizeof(float)]),
        handle->majorantGrid->voxels.data(), size * sizeof(float));
      grid_storage_buffer->host_stamp++;
    }

    // copy the packet to GPU
    medium_buffer->host.resize(sizeof(Medium::MediumPacket) * (index + 1));
    Medium::MediumPacket pack = handle->packet;
    memcpy((float*)&(medium_buffer->host[sizeof(Medium::MediumPacket) * index]), &pack, sizeof(pack));
    medium_buffer->host_stamp++;


    return index;
  }
  return iter->second.first;
}

auto Scene::GPUScene::fetch_geometry_data(int geometryID) noexcept -> GeometryDrawData {
  std::span<GeometryDrawData> geometry_array = {
  (GeometryDrawData*)geometry_buffer->host.data(),
  geometry_buffer->host.size() / sizeof(GeometryDrawData),
  };
  return geometry_array[geometryID];
}

auto Scene::GPUScene::fetch_triangle_indices(
  GeometryDrawData const& geometry, int triangleID) noexcept -> uvec3 {
  std::span<uint32_t> index_array = {
    (uint32_t*)index_buffer->host.data(),
    index_buffer->host.size() / sizeof(uint32_t),
  };
  return uvec3 {
    index_array[geometry.indexOffset + triangleID * 3 + 0],
    index_array[geometry.indexOffset + triangleID * 3 + 1],
    index_array[geometry.indexOffset + triangleID * 3 + 2]
  };
}

auto Scene::GPUScene::fetch_vertex_position(int vertexID) noexcept -> vec3 {
  std::span<float> position_array = {
    (float*)position_buffer->host.data(),
    position_buffer->host.size() / sizeof(float),
  };
  return vec3 {
    position_array[vertexID * 3 + 0],
    position_array[vertexID * 3 + 1],
    position_array[vertexID * 3 + 2]
  };
}

auto Scene::GPUScene::fetch_vertex_normal(int vertexID) noexcept -> vec3 {
  std::span<float> vertex_array = {
  (float*)vertex_buffer->host.data(),
  vertex_buffer->host.size() / sizeof(float),
  };
  return vec3{
    vertex_array[vertexID * 8 + 0],
    vertex_array[vertexID * 8 + 1],
    vertex_array[vertexID * 8 + 2]
  };
}

auto loadObjMesh(std::string path, Scene& scene) noexcept -> MeshHandle {
      // load obj file
  tinyobj::ObjReaderConfig reader_config;
  reader_config.mtl_search_path =
    std::filesystem::path(path).parent_path().string();  // Path to material files
  tinyobj::ObjReader reader;
  if (!reader.ParseFromFile(path, reader_config)) {
    if (!reader.Error().empty()) {
      root::print::error("TinyObjReader: " + reader.Error());
    }
    return MeshHandle{};
  }
  if (!reader.Warning().empty()) {
    root::print::warning("TinyObjReader: " + reader.Warning());
  }
  auto& attrib = reader.GetAttrib();
  auto& shapes = reader.GetShapes();
  auto& materials = reader.GetMaterials();

  std::vector<float> vertexBufferV = {};
  std::vector<float> positionBufferV = {};
  std::vector<uint32_t> indexBufferWV = {};
  // create mesh resource
  MeshHandle mesh = GFXContext::load_mesh_empty();

  // check whether tangent is need in mesh attributes
  bool needTangent = false;
  for (auto const& entry : defaultMeshDataLayout.layout)
    if (entry.info == MeshDataLayout::VertexInfo::TANGENT) needTangent = true;

  // Loop over shapes
  uint64_t global_index_offset = 0;
  uint32_t submesh_vertex_offset = 0, submesh_index_offset = 0;
  for (size_t s = 0; s < shapes.size(); s++) {
    uint32_t vertex_offset = 0;
    std::unordered_map<uint64_t, uint32_t> uniqueVertices{};
    vec3 position_max = vec3(-1e9);
    vec3 position_min = vec3(1e9);
    // Loop over faces(polygon)
    size_t index_offset = 0;
    for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
      size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);
      // require tangent
      if (fv != 3) {
        root::print::error(
          "GFX :: SceneNodeLoader_obj :: non-triangle geometry not "
          "supported when required TANGENT attribute now.");
        return MeshHandle{};
      }
      vec3 tangent;
      vec3 bitangent;
      if (needTangent) {
        vec3 positions[3];
        vec3 normals[3];
        vec2 uvs[3];
        for (size_t v = 0; v < fv; v++) {
          // index finding
          tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
          positions[v] = {attrib.vertices[3 * size_t(idx.vertex_index) + 0],
                          attrib.vertices[3 * size_t(idx.vertex_index) + 1],
                          attrib.vertices[3 * size_t(idx.vertex_index) + 2]};
          if (attrib.normals.size() == 0) {
            normals[v] = {0, 0, 0};  
          } else {
            normals[v] = {attrib.normals[3 * size_t(idx.normal_index) + 0],
                          attrib.normals[3 * size_t(idx.normal_index) + 1],
                          attrib.normals[3 * size_t(idx.normal_index) + 2]};          
          }
          if (attrib.texcoords.size() == 0) {
            uvs[v] = {0, 0};
          } else {
            uvs[v] = {attrib.texcoords[2 * size_t(idx.texcoord_index) + 0],
                      -attrib.texcoords[2 * size_t(idx.texcoord_index) + 1]};

          }
        }
        vec3 edge1 = positions[1] - positions[0];
        vec3 edge2 = positions[2] - positions[0];
        vec2 deltaUV1 = uvs[1] - uvs[0];
        vec2 deltaUV2 = uvs[2] - uvs[0];

        float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);

        tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
        tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
        tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
        tangent = normalize(tangent);

        bitangent.x = f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x);
        bitangent.y = f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y);
        bitangent.z = f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z);
        bitangent = normalize(bitangent);
      }
      // Loop over vertices in the face.
      for (size_t v = 0; v < fv; v++) {
        // index finding
        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
        // atrributes filling
        std::vector<float> vertex = {};
        std::vector<float> position = {};
        for (auto const& entry : defaultMeshDataLayout.layout) {
          // vertex position
          if (entry.info == MeshDataLayout::VertexInfo::POSITION) {
            if (entry.format == rhi::VertexFormat::FLOAT32X3) {
              tinyobj::real_t vx =
                  attrib.vertices[3 * size_t(idx.vertex_index) + 0];
              tinyobj::real_t vy =
                  attrib.vertices[3 * size_t(idx.vertex_index) + 1];
              tinyobj::real_t vz =
                  attrib.vertices[3 * size_t(idx.vertex_index) + 2];
              vertex.push_back(vx);
              vertex.push_back(vy);
              vertex.push_back(vz);
              position_min = se::min(position_min, vec3{ vx,vy,vz });
              position_max = se::max(position_max, vec3{ vx,vy,vz });
              if (defaultMeshLoadConfig.usePositionBuffer) {
                position.push_back(vx);
                position.push_back(vy);
                position.push_back(vz);
              }
            } else {
                root::print::error(
                  "GFX :: SceneNodeLoader_obj :: unwanted vertex format for "
                  "POSITION attributes.");
              return MeshHandle{};
            }
          } else if (entry.info == MeshDataLayout::VertexInfo::NORMAL) {
            // Check if `normal_index` is zero or positive. negative = no
            // normal data
            if (idx.normal_index >= 0) {
              tinyobj::real_t nx =
                  attrib.normals[3 * size_t(idx.normal_index) + 0];
              tinyobj::real_t ny =
                  attrib.normals[3 * size_t(idx.normal_index) + 1];
              tinyobj::real_t nz =
                  attrib.normals[3 * size_t(idx.normal_index) + 2];
              vertex.push_back(nx);
              vertex.push_back(ny);
              vertex.push_back(nz);
            } else {
              vertex.push_back(0);
              vertex.push_back(0);
              vertex.push_back(0);
            }
          } else if (entry.info == MeshDataLayout::VertexInfo::UV) {
            if (idx.texcoord_index >= 0) {
              tinyobj::real_t tx =
                  attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
              tinyobj::real_t ty =
                  attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
              vertex.push_back(tx);
              vertex.push_back(1 - ty);
            } else {
              vertex.push_back(0);
              vertex.push_back(0);
            }
          } else if (entry.info == MeshDataLayout::VertexInfo::TANGENT) {
            if (isnan(tangent.x) || isnan(tangent.y) || isnan(tangent.z)) {
              vertex.push_back(0);
              vertex.push_back(0);
              vertex.push_back(0);
            }
            else {
              vertex.push_back(tangent.x);
              vertex.push_back(tangent.y);
              vertex.push_back(tangent.z);
            }
          } else if (entry.info == MeshDataLayout::VertexInfo::COLOR) {
            // Optional: vertex colors
            tinyobj::real_t red =
                attrib.colors[3 * size_t(idx.vertex_index) + 0];
            tinyobj::real_t green =
                attrib.colors[3 * size_t(idx.vertex_index) + 1];
            tinyobj::real_t blue =
                attrib.colors[3 * size_t(idx.vertex_index) + 2];
            vertex.push_back(red);
            vertex.push_back(green);
            vertex.push_back(blue);
          } else if (entry.info == MeshDataLayout::VertexInfo::CUSTOM) {
          }
        }

        if (defaultMeshLoadConfig.deduplication) {
          uint64_t hashed_vertex = hash_vertex(vertex);
          if (uniqueVertices.count(hashed_vertex) == 0) {
            uniqueVertices[hashed_vertex] =
                static_cast<uint32_t>(vertex_offset);
            vertexBufferV.insert(vertexBufferV.end(), vertex.begin() + 3,
                                 vertex.end());
            positionBufferV.insert(positionBufferV.end(), position.begin(),
                                   position.end());
            ++vertex_offset;
          }

          // index filling
          if (defaultMeshLoadConfig.layout.format == rhi::IndexFormat::UINT16_t)
            indexBufferWV.push_back(uniqueVertices[hashed_vertex]);
          else if (defaultMeshLoadConfig.layout.format == rhi::IndexFormat::UINT32_T)
            indexBufferWV.push_back(uniqueVertices[hashed_vertex]);
        } 
        else {
          vertexBufferV.insert(vertexBufferV.end(), vertex.begin() + 3,
                               vertex.end());
          positionBufferV.insert(positionBufferV.end(), position.begin(),
            position.end());
          // index filling
          if (defaultMeshLoadConfig.layout.format == rhi::IndexFormat::UINT16_t)
            indexBufferWV.push_back(vertex_offset);
          else if (defaultMeshLoadConfig.layout.format == rhi::IndexFormat::UINT32_T)
            indexBufferWV.push_back(vertex_offset);
          ++vertex_offset;
        }
      }
      index_offset += fv;
      // per-face material
      shapes[s].mesh.material_ids[f];
    }
    global_index_offset += index_offset;

    // load Material
    Mesh::MeshPrimitive sePrimitive; 
    sePrimitive.offset = submesh_index_offset;
    sePrimitive.size = index_offset;
    sePrimitive.baseVertex = submesh_vertex_offset;
    sePrimitive.numVertex = positionBufferV.size() / 3 - submesh_vertex_offset;
    sePrimitive.max = position_max;
    sePrimitive.min = position_min;
    mesh.get()->primitives.emplace_back(std::move(sePrimitive));
    // todo:: add material
    submesh_index_offset = global_index_offset;
    submesh_vertex_offset += positionBufferV.size() / 3 - submesh_vertex_offset;
  }
  { // register mesh
    Buffer* position_buffer = scene.gpuScene.position_buffer.get();
    size_t position_size = sizeof(float) * positionBufferV.size();
    size_t position_offset = position_buffer->host.size();
    position_buffer->host.resize(position_size + position_offset);
    memcpy(&position_buffer->host[position_offset], positionBufferV.data(), position_size);
    mesh.get()->vertex_offset = position_offset;

    Buffer* index_buffer = scene.gpuScene.index_buffer.get();
    size_t index_size = sizeof(uint32_t) * indexBufferWV.size();
    size_t index_offset = index_buffer->host.size();
    index_buffer->host.resize(index_size + index_offset);
    memcpy(&index_buffer->host[index_offset], indexBufferWV.data(), index_size);
    mesh.get()->index_offset = index_offset;

    Buffer* vertex_buffer = scene.gpuScene.vertex_buffer.get();
    size_t vertex_size = sizeof(float) * vertexBufferV.size();
    size_t vertex_offset = vertex_buffer->host.size();
    vertex_buffer->host.resize(vertex_size + vertex_offset);
    memcpy(&vertex_buffer->host[vertex_offset], vertexBufferV.data(), vertex_size);
    
    //if (true) {
    //  mesh.jointIndexBuffer_host = Core::Buffer(sizeof(uint64_t) * JointIndexBuffer.size());
    //  memcpy(mesh.jointIndexBuffer_host.data, JointIndexBuffer.data(), mesh.jointIndexBuffer_host.size);
    //  mesh.jointIndexBufferInfo.onHost = true;
    //  mesh.jointIndexBufferInfo.size = mesh.jointIndexBuffer_host.size;

    //  mesh.jointWeightBuffer_host = Core::Buffer(sizeof(float) * JointweightsBuffer.size());
    //  memcpy(mesh.jointWeightBuffer_host.data, JointweightsBuffer.data(), mesh.jointWeightBuffer_host.size);
    //  mesh.jointWeightBufferInfo.onHost = true;
    //  mesh.jointWeightBufferInfo.size = mesh.jointWeightBuffer_host.size;
    //}
  }
  return mesh;
}

struct glTFLoaderEnv {
  std::string directory;
  std::unordered_map<tinygltf::Texture const*, TextureHandle> textures;
  std::unordered_map<tinygltf::Material const*, MaterialHandle> materials;
  std::vector<std::vector<se::mat4>> skinning_matrices;
  std::vector<std::vector<int>> skinning_indices;
  std::vector<MediumHandle> mediums;
  //std::vector<std::vector<GFX::AnimationComponent::AnimationSampler>> anim_samplers;
  //std::vector<std::unordered_map<uint32_t, std::vector<GFX::AnimationComponent::AnimationChannel>>> anim_channels;
  std::unordered_map<int, Node> node2go;
};

auto loadGLTFMaterial(tinygltf::Material const* glmaterial, tinygltf::Model const* model,
    glTFLoaderEnv& env, gfx::Scene& gfxscene, MeshLoaderConfig meshConfig = {}) noexcept
    -> MaterialHandle {
  if (env.materials.find(glmaterial) != env.materials.end()) {
    return env.materials[glmaterial];
  }

  MaterialHandle mat = GFXContext::load_material_empty();
  std::string name = glmaterial->name;

  auto to_sampler = [&](int sampler_idx) {
    tinygltf::Sampler const& sampler = model->samplers[sampler_idx];
    rhi::SamplerDescriptor desc;
    // Min and Mipmap filter
    if (sampler.minFilter == TINYGLTF_TEXTURE_FILTER_NEAREST) {
      desc.minFilter = rhi::FilterMode::NEAREST;
    } else if (sampler.minFilter == TINYGLTF_TEXTURE_FILTER_LINEAR) {
      desc.minFilter = rhi::FilterMode::LINEAR;
    } else if (sampler.minFilter == TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST) {
      desc.minFilter = rhi::FilterMode::NEAREST;
      desc.mipmapFilter = rhi::MipmapFilterMode::NEAREST;
    } else if (sampler.minFilter == TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_NEAREST) {
      desc.minFilter = rhi::FilterMode::LINEAR;
      desc.mipmapFilter = rhi::MipmapFilterMode::NEAREST;
    } else if (sampler.minFilter == TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR) {
      desc.minFilter = rhi::FilterMode::NEAREST;
      desc.mipmapFilter = rhi::MipmapFilterMode::LINEAR;
    } else if (sampler.minFilter == TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR) {
      desc.minFilter = rhi::FilterMode::LINEAR;
      desc.mipmapFilter = rhi::MipmapFilterMode::LINEAR;
    }
    // Mag filter
    if (sampler.magFilter == TINYGLTF_TEXTURE_FILTER_NEAREST) {
      desc.magFilter = rhi::FilterMode::NEAREST;
    } else if (sampler.magFilter == TINYGLTF_TEXTURE_FILTER_LINEAR) {
      desc.magFilter = rhi::FilterMode::LINEAR;
    }
    // WarpS
    if (sampler.wrapS == TINYGLTF_TEXTURE_WRAP_REPEAT) {
      desc.addressModeU = rhi::AddressMode::REPEAT;
    } else if (sampler.wrapS == TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE) {
      desc.addressModeU = rhi::AddressMode::CLAMP_TO_EDGE;
    } else if (sampler.wrapS == TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT) {
      desc.addressModeU = rhi::AddressMode::MIRROR_REPEAT;
    }
    // WarpT
    if (sampler.wrapT == TINYGLTF_TEXTURE_WRAP_REPEAT) {
      desc.addressModeV = rhi::AddressMode::REPEAT;
    } else if (sampler.wrapT == TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE) {
      desc.addressModeV = rhi::AddressMode::CLAMP_TO_EDGE;
    } else if (sampler.wrapT == TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT) {
      desc.addressModeV = rhi::AddressMode::MIRROR_REPEAT;
    }
    return desc;
  };

  // load diffuse information
  { // load diffuse color
    if (glmaterial->pbrMetallicRoughness.baseColorFactor.size() > 0) {
      mat->baseOrDiffuseColor = se::vec3{
        (float)glmaterial->pbrMetallicRoughness.baseColorFactor[0],
        (float)glmaterial->pbrMetallicRoughness.baseColorFactor[1],
        (float)glmaterial->pbrMetallicRoughness.baseColorFactor[2],
      };
      mat->roughnessFactor = (float)glmaterial->pbrMetallicRoughness.roughnessFactor;
      mat->metallicFactor = (float)glmaterial->pbrMetallicRoughness.metallicFactor;
    }
    mat->emissiveColor = se::vec3{
        (float)glmaterial->emissiveFactor[0],
        (float)glmaterial->emissiveFactor[1],
        (float)glmaterial->emissiveFactor[2],
    };
  }
  //{ // load diffuse texture
  //  if (glmaterial->pbrMetallicRoughness.baseColorTexture.index != -1) {
  //    tinygltf::Texture const& texture = model->textures[glmaterial->pbrMetallicRoughness.baseColorTexture.index];
  //    Core::GUID texBasecolor = loadGLTFMaterialTextures(&texture, model, env, gfxscene, meshConfig);
  //    gfxmat.textures["base_color"] = GFX::Material::TextureEntry{texBasecolor, 0, to_sampler(texture.sampler)};
  //  }
  //}

  return mat;
}

static inline auto loadGLTFMesh(tinygltf::Mesh const& gltfmesh,
  Node& gfxNode, Scene& scene, int node_id, tinygltf::Model const* model,
  glTFLoaderEnv& env) noexcept -> MeshHandle {
#define INDEX_TYPE uint32_t
  // Load meshes into Runtime resource managers.
  rhi::Device* device = GFXContext::device;
  std::vector<INDEX_TYPE> indexBuffer_uint = {};
  std::vector<float> vertexBuffer = {};
  std::vector<float> PositionBuffer = {};
  std::vector<uint64_t> JointIndexBuffer = {};
  std::vector<float> JointweightsBuffer = {};
  // Create GFX mesh, and add it to resource manager
  size_t submesh_index_offset = 0;
  size_t submesh_vertex_offset = 0;
  MeshHandle mesh = GFXContext::load_mesh_empty();
  // For each primitive
  for (auto const& meshPrimitive : gltfmesh.primitives) {
    std::vector<INDEX_TYPE> indexArray_uint = {};
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
    switch (indicesAccessor.componentType) {
      case TINYGLTF_COMPONENT_TYPE_BYTE: {
        ArrayAdapter<char> originIndexArray(dataAddress, count, byteStride);
        for (size_t i = 0; i < count; ++i)
          indexArray_uint.emplace_back(INDEX_TYPE(originIndexArray[i]));
      } break;
      case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
        ArrayAdapter<unsigned char> originIndexArray(dataAddress, count,
                                                     byteStride);
        for (size_t i = 0; i < count; ++i)
          indexArray_uint.emplace_back(INDEX_TYPE(originIndexArray[i]));
      } break;
      case TINYGLTF_COMPONENT_TYPE_SHORT: {
        ArrayAdapter<short> originIndexArray(dataAddress, count, byteStride);
        for (size_t i = 0; i < count; ++i)
          indexArray_uint.emplace_back(INDEX_TYPE(originIndexArray[i]));
      } break;
      case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
        ArrayAdapter<unsigned short> originIndexArray(dataAddress, count,
                                                      byteStride);
        for (size_t i = 0; i < count; ++i)
          indexArray_uint.emplace_back(INDEX_TYPE(originIndexArray[i]));
      } break;
      case TINYGLTF_COMPONENT_TYPE_INT: {
        ArrayAdapter<int> originIndexArray(dataAddress, count, byteStride);
        for (size_t i = 0; i < count; ++i)
          indexArray_uint.emplace_back(INDEX_TYPE(originIndexArray[i]));
      } break;
      case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
        ArrayAdapter<unsigned int> originIndexArray(dataAddress, count,
                                                    byteStride);
        for (size_t i = 0; i < count; ++i)
          indexArray_uint.emplace_back(INDEX_TYPE(originIndexArray[i]));
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
                      root::print::error(
                          "GFX :: tinygltf :: unrecognized vector type for "
                          "UV");
                  }
                } break;
                default:
                  root::print::error("GFX :: tinygltf :: unreconized componant type for UV");
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
                        se::Vector4<uint16_t> j0= joints[i];
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
        root::print::error("GFX :: tinygltf :: primitive mode not implemented");
        break;
    }
    // Compute the tangent vector if no provided
    if (vertexBuffer_tangentOnly.size() == 0) {
      //for (size_t i = 0; i < indexArray_uint.size(); i += 3) {
      //  if (vertexBuffer_uvOnly.size() == 0) {
      //      // if has no uv
      //      for (int i = 0; i < 3; ++i) {
      //        vertexBuffer_tangentOnly.push_back(0);
      //        vertexBuffer_tangentOnly.push_back(0);
      //        vertexBuffer_tangentOnly.push_back(0);
      //      }
      //  } else {
      //      size_t i0 = i + 0; size_t i1 = i + 1; size_t i2 = i + 2;
      //      se::vec3 pos1 = {vertexBuffer_positionOnly[i0 * 3 + 0],
      //                         vertexBuffer_positionOnly[i0 * 3 + 1],
      //                         vertexBuffer_positionOnly[i0 * 3 + 2]};
      //      se::vec3 pos2 = {vertexBuffer_positionOnly[i1 * 3 + 0],
      //                         vertexBuffer_positionOnly[i1 * 3 + 1],
      //                         vertexBuffer_positionOnly[i1 * 3 + 2]};
      //      se::vec3 pos3 = {vertexBuffer_positionOnly[i2 * 3 + 0],
      //                         vertexBuffer_positionOnly[i2 * 3 + 1],
      //                         vertexBuffer_positionOnly[i2 * 3 + 2]};
      //      se::vec2 uv1  = {vertexBuffer_uvOnly[i0 * 2 + 0] - int(vertexBuffer_uvOnly[i0 * 2 + 0]),
      //                         vertexBuffer_uvOnly[i0 * 2 + 1] - int(vertexBuffer_uvOnly[i0 * 2 + 1]) };
      //      se::vec2 uv2  = {vertexBuffer_uvOnly[i1 * 2 + 0] - int(vertexBuffer_uvOnly[i1 * 2 + 0]),
      //                         vertexBuffer_uvOnly[i1 * 2 + 1] - int(vertexBuffer_uvOnly[i1 * 2 + 1]) };
      //      se::vec2 uv3  = {vertexBuffer_uvOnly[i2 * 2 + 0] - int(vertexBuffer_uvOnly[i2 * 2 + 0]),
      //                         vertexBuffer_uvOnly[i2 * 2 + 1] - int(vertexBuffer_uvOnly[i2 * 2 + 1]) };

      //      se::vec3 tangent;
      //      se::vec3 edge1 = pos2 - pos1;
      //      se::vec3 edge2 = pos3 - pos1;
      //      se::vec2 deltaUV1 = uv2 - uv1;
      //      se::vec2 deltaUV2 = uv3 - uv1;
      //      float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);
      //      tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
      //      tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
      //      tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
      //      tangent = se::normalize(tangent);
      //      for (int i = 0; i < 3; ++i) {
      //        vertexBuffer_tangentOnly.push_back(tangent.x);
      //        vertexBuffer_tangentOnly.push_back(tangent.y);
      //        vertexBuffer_tangentOnly.push_back(tangent.z);
      //      }
      //  }
      //}
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
            } else {
              vertexBuffer.push_back(vertexBuffer_normalOnly[i * 3 + 0]);
              vertexBuffer.push_back(vertexBuffer_normalOnly[i * 3 + 1]);
              vertexBuffer.push_back(vertexBuffer_normalOnly[i * 3 + 2]);            
            }
        } else if (entry.info == MeshDataLayout::VertexInfo::UV) {
            if (vertexBuffer_uvOnly.size() == 0) { // if uv is not provided
              vertexBuffer.push_back(0.f);
              vertexBuffer.push_back(0.f);
            } else {
              vertexBuffer.push_back(vertexBuffer_uvOnly[i * 2 + 0] - int(vertexBuffer_uvOnly[i * 2 + 0]));
              vertexBuffer.push_back(vertexBuffer_uvOnly[i * 2 + 1] - int(vertexBuffer_uvOnly[i * 2 + 1]));
            }
        } else if (entry.info == MeshDataLayout::VertexInfo::TANGENT) {
            vertexBuffer.push_back(0.f);
            vertexBuffer.push_back(0.f);
            vertexBuffer.push_back(0.f);
            //vertexBuffer.push_back(vertexBuffer_tangentOnly[i * 3 + 0]);
            //vertexBuffer.push_back(vertexBuffer_tangentOnly[i * 3 + 1]);
            //vertexBuffer.push_back(vertexBuffer_tangentOnly[i * 3 + 2]);
        } else if (entry.info == MeshDataLayout::VertexInfo::COLOR) {
            // Optional: vertex colors
            vertexBuffer.push_back(0);
            vertexBuffer.push_back(0);
            vertexBuffer.push_back(0);
        } else if (entry.info == MeshDataLayout::VertexInfo::CUSTOM) {}
      }
    }

    //for (size_t i = 0; i < indexArray_uint.size(); ++i) {
    //  for (auto const& entry : defaultMeshLoadConfig.layout.layout) {
    //    // vertex position
    //    if (entry.info == MeshDataLayout::VertexInfo::POSITION) {
    //        if (entry.format == rhi::VertexFormat::FLOAT32X3) {
    //          //vertexBuffer.push_back(vertexBuffer_positionOnly[i * 3 + 0]);
    //          //vertexBuffer.push_back(vertexBuffer_positionOnly[i * 3 + 1]);
    //          //vertexBuffer.push_back(vertexBuffer_positionOnly[i * 3 + 2]);
    //          if (defaultMeshLoadConfig.usePositionBuffer) {
    //            if (vertexBuffer_positionOnly.size() != 0) {
    //                PositionBuffer.push_back(vertexBuffer_positionOnly[i * 3 + 0]);
    //                PositionBuffer.push_back(vertexBuffer_positionOnly[i * 3 + 1]);
    //                PositionBuffer.push_back(vertexBuffer_positionOnly[i * 3 + 2]);
    //            } else {
    //            
    //            }
    //          }
    //        } else {
    //          root::print::error(
    //              "GFX :: SceneNodeLoader_assimp :: unwanted vertex format for "
    //              "POSITION attributes.");
    //          return {};
    //        }
    //    } 
    //    else if (entry.info == MeshDataLayout::VertexInfo::NORMAL) {
    //        if (vertexBuffer_normalOnly.size() == 0) { // if normal is not provided
    //          vertexBuffer.push_back(0.f);
    //          vertexBuffer.push_back(0.f);
    //          vertexBuffer.push_back(0.f);            
    //        } else {
    //          vertexBuffer.push_back(vertexBuffer_normalOnly[i * 3 + 0]);
    //          vertexBuffer.push_back(vertexBuffer_normalOnly[i * 3 + 1]);
    //          vertexBuffer.push_back(vertexBuffer_normalOnly[i * 3 + 2]);            
    //        }
    //    } else if (entry.info == MeshDataLayout::VertexInfo::UV) {
    //        if (vertexBuffer_uvOnly.size() == 0) { // if uv is not provided
    //          vertexBuffer.push_back(0.f);
    //          vertexBuffer.push_back(0.f);
    //        } else {
    //          vertexBuffer.push_back(vertexBuffer_uvOnly[i * 2 + 0] - int(vertexBuffer_uvOnly[i * 2 + 0]));
    //          vertexBuffer.push_back(vertexBuffer_uvOnly[i * 2 + 1] - int(vertexBuffer_uvOnly[i * 2 + 1]));
    //        }
    //    } else if (entry.info == MeshDataLayout::VertexInfo::TANGENT) {
    //        vertexBuffer.push_back(vertexBuffer_tangentOnly[i * 3 + 0]);
    //        vertexBuffer.push_back(vertexBuffer_tangentOnly[i * 3 + 1]);
    //        vertexBuffer.push_back(vertexBuffer_tangentOnly[i * 3 + 2]);
    //    } else if (entry.info == MeshDataLayout::VertexInfo::COLOR) {
    //        // Optional: vertex colors
    //        vertexBuffer.push_back(0);
    //        vertexBuffer.push_back(0);
    //        vertexBuffer.push_back(0);
    //    } else if (entry.info == MeshDataLayout::VertexInfo::CUSTOM) {
    //    }
    //  }

    //  indexBuffer_uint.push_back(i);
    //}
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
    tinygltf::Value primitive_extra = meshPrimitive.extras;
    if (primitive_extra.Has("exterior")) {
      int exterior_index = primitive_extra.Get("exterior").GetNumberAsInt();
      if (exterior_index >= 0) sePrimitive.exterior = env.mediums[exterior_index];
    }
    if (primitive_extra.Has("interior")) {
      int interior_index = primitive_extra.Get("interior").GetNumberAsInt();
      if (interior_index >= 0) sePrimitive.interior = env.mediums[interior_index];
    }

    mesh.get()->primitives.emplace_back(std::move(sePrimitive));
    // todo:: add material
    submesh_index_offset = indexBuffer_uint.size();
    submesh_vertex_offset = PositionBuffer.size() / 3;
  }
  // create mesh resource
  { // register mesh
    Buffer* position_buffer = scene.gpuScene.position_buffer.get();
    size_t position_size = sizeof(float) * PositionBuffer.size();
    size_t position_offset = position_buffer->host.size();
    position_buffer->host.resize(position_size + position_offset);
    memcpy(&position_buffer->host[position_offset], PositionBuffer.data(), position_size);
    mesh.get()->vertex_offset = position_offset;

    Buffer* index_buffer = scene.gpuScene.index_buffer.get();
    size_t index_size = sizeof(uint32_t) * indexBuffer_uint.size();
    size_t index_offset = index_buffer->host.size();
    index_buffer->host.resize(index_size + index_offset);
    memcpy(&index_buffer->host[index_offset], indexBuffer_uint.data(), index_size);
    mesh.get()->index_offset = index_offset;

    Buffer* vertex_buffer = scene.gpuScene.vertex_buffer.get();
    size_t vertex_size = sizeof(float) * vertexBuffer.size();
    size_t vertex_offset = vertex_buffer->host.size();
    vertex_buffer->host.resize(vertex_size + vertex_offset);
    memcpy(&vertex_buffer->host[vertex_offset], vertexBuffer.data(), vertex_size);
    
    //if (true) {
    //  mesh.jointIndexBuffer_host = Core::Buffer(sizeof(uint64_t) * JointIndexBuffer.size());
    //  memcpy(mesh.jointIndexBuffer_host.data, JointIndexBuffer.data(), mesh.jointIndexBuffer_host.size);
    //  mesh.jointIndexBufferInfo.onHost = true;
    //  mesh.jointIndexBufferInfo.size = mesh.jointIndexBuffer_host.size;

    //  mesh.jointWeightBuffer_host = Core::Buffer(sizeof(float) * JointweightsBuffer.size());
    //  memcpy(mesh.jointWeightBuffer_host.data, JointweightsBuffer.data(), mesh.jointWeightBuffer_host.size);
    //  mesh.jointWeightBufferInfo.onHost = true;
    //  mesh.jointWeightBufferInfo.size = mesh.jointWeightBuffer_host.size;
    //}
  }
  return mesh;
 }
 
struct xmlLoaderEnv {
  std::string directory;
  std::unordered_map<TPM_NAMESPACE::Object const*, TextureHandle> textures;
  std::unordered_map<TPM_NAMESPACE::Object const*, MaterialHandle> materials;
  std::unordered_map<TPM_NAMESPACE::Object const*, MediumHandle> mediums;
};

auto loadXMLTextures(TPM_NAMESPACE::Object const* node,
  xmlLoaderEnv* env) noexcept -> TextureHandle {
  if (env->textures.find(node) != env->textures.end()) {
    return env->textures[node];
  }
  if (node->type() != TPM_NAMESPACE::OT_TEXTURE) {
    se::root::print::error("GFX :: Mitsuba Loader :: Try load texture node not actually texture.");
  }
  std::string filename = node->property("filename").getString();
  std::string tex_path = env->directory + "/" + filename;
  TextureHandle texture = gfx::GFXContext::create_texture_file(tex_path);
  env->textures[node] = texture;
  return texture;
}

/// To support spectral data, we need to convert spectral measurements (how much energy at each wavelength) to
/// RGB. To do this, we first convert the spectral data to CIE XYZ, by
/// integrating over the XYZ response curve. Here we use an analytical response
/// curve proposed by Wyman et al.: https://jcgt.org/published/0002/02/01/
inline float xFit_1931(float wavelength) {
  float t1 = (wavelength - float(442.0)) * ((wavelength < float(442.0)) ? float(0.0624) : float(0.0374));
  float t2 = (wavelength - float(599.8)) * ((wavelength < float(599.8)) ? float(0.0264) : float(0.0323));
  float t3 = (wavelength - float(501.1)) * ((wavelength < float(501.1)) ? float(0.0490) : float(0.0382));
  return float(0.362) * exp(-float(0.5) * t1 * t1) +
    float(1.056) * exp(-float(0.5) * t2 * t2) -
    float(0.065) * exp(-float(0.5) * t3 * t3);
}
inline float yFit_1931(float wavelength) {
  float t1 = (wavelength - float(568.8)) * ((wavelength < float(568.8)) ? float(0.0213) : float(0.0247));
  float t2 = (wavelength - float(530.9)) * ((wavelength < float(530.9)) ? float(0.0613) : float(0.0322));
  return float(0.821) * exp(-float(0.5) * t1 * t1) +
    float(0.286) * exp(-float(0.5) * t2 * t2);
}
inline float zFit_1931(float wavelength) {
  float t1 = (wavelength - float(437.0)) * ((wavelength < float(437.0)) ? float(0.0845) : float(0.0278));
  float t2 = (wavelength - float(459.0)) * ((wavelength < float(459.0)) ? float(0.0385) : float(0.0725));
  return float(1.217) * exp(-float(0.5) * t1 * t1) +
    float(0.681) * exp(-float(0.5) * t2 * t2);
}
inline vec3 XYZintegral_coeff(float wavelength) {
  return Vector3{ xFit_1931(wavelength), yFit_1931(wavelength), zFit_1931(wavelength) };
}

inline vec3 integrate_XYZ(const std::vector<std::pair<float, float>>& data) {
  static const float CIE_Y_integral = 106.856895;
  static const float wavelength_beg = 400;
  static const float wavelength_end = 700;
  if (data.size() == 0) {
    return vec3{ 0, 0, 0 };
  }
  vec3 ret = vec3{ 0, 0, 0 };
  int data_pos = 0;
  // integrate from wavelength 400 nm to 700 nm, increment by 1nm at a time
  // linearly interpolate from the data
  for (float wavelength = wavelength_beg; wavelength <= wavelength_end; wavelength += float(1)) {
    // assume the spectrum data is sorted by wavelength
    // move data_pos such that wavelength is between two data or at one end
    while (data_pos < (int)data.size() - 1 &&
      !((data[data_pos].first <= wavelength &&
        data[data_pos + 1].first > wavelength) ||
        data[0].first > wavelength)) {
      data_pos += 1;
    }
    float measurement = 0;
    if (data_pos < (int)data.size() - 1 && data[0].first <= wavelength) {
      float curr_data = data[data_pos].second;
      float next_data = data[std::min(data_pos + 1, (int)data.size() - 1)].second;
      float curr_wave = data[data_pos].first;
      float next_wave = data[std::min(data_pos + 1, (int)data.size() - 1)].first;
      // linearly interpolate
      measurement = curr_data * (next_wave - wavelength) / (next_wave - curr_wave) +
        next_data * (wavelength - curr_wave) / (next_wave - curr_wave);
    }
    else {
      // assign the endpoint
      measurement = data[data_pos].second;
    }
    vec3 coeff = XYZintegral_coeff(wavelength);
    ret += coeff * measurement;
  }
  float wavelength_span = wavelength_end - wavelength_beg;
  ret *= (wavelength_span / (CIE_Y_integral * (wavelength_end - wavelength_beg)));
  return ret;
}

inline vec3 XYZ_to_RGB(const vec3& xyz) {
  return vec3{
      float(3.240479) * xyz[0]  - float(1.537150) * xyz[1] - float(0.498535) * xyz[2],
      float(-0.969256) * xyz[0] + float(1.875991) * xyz[1] + float(0.041556) * xyz[2],
      float(0.055648) * xyz[0]  - float(0.204043) * xyz[1] + float(1.057311) * xyz[2] };
}

vec3 spectrum_to_rgb(TPM_NAMESPACE::Spectrum spec) {
  std::vector<std::pair<float, float>> vec (spec.wavelengths().size());
  for (int i = 0; i < spec.wavelengths().size(); ++i) {
    vec[i] = { spec.wavelengths()[i], spec.weights()[i] };
  }
  vec3 xyz = integrate_XYZ(vec);
  return XYZ_to_RGB(xyz);
}

auto loadXMLMaterial(TPM_NAMESPACE::Object const* node,
    xmlLoaderEnv* env) noexcept -> gfx::MaterialHandle {
  if (env->materials.find(node) != env->materials.end()) {
    return env->materials[node];
  }
  if (node->type() != TPM_NAMESPACE::OT_BSDF) {
    se::root::print::error("gfx :: XML Loader :: Try load material node not actually bsdf.");
    return gfx::MaterialHandle{};
  }

  TPM_NAMESPACE::Object const* mat_node = nullptr;
  if (node->pluginType() == "dielectric") {
    mat_node = node;
  }
  else if(node->pluginType() == "roughdielectric") {
    mat_node = node;
  }
  else if(node->pluginType() == "thindielectric") {
    mat_node = node;
  }
  else if (node->pluginType() == "twosided") {
    if (node->anonymousChildren().size() == 0) {
      root::print::error("Mitsuba Loader :: Material loading exception.");
      return gfx::MaterialHandle{};
    }
    mat_node = node->anonymousChildren()[0].get();
  }
  else if (node->pluginType() == "mask") {
    if (node->anonymousChildren().size() == 0) {
      root::print::error("Mitsuba Loader :: Material loading exception.");
      return gfx::MaterialHandle{};
    }
    return loadXMLMaterial(node->anonymousChildren()[0].get(), env);
  }
  else if (node->pluginType() == "bumpmap") {
    if (node->anonymousChildren().size() == 0) {
      root::print::error("Mitsuba Loader :: Material loading exception.");
      return gfx::MaterialHandle{};
    }
    return loadXMLMaterial(node->anonymousChildren()[0].get(), env);
  }
  else {
    mat_node = node;
  }

  MaterialHandle mat = GFXContext::load_material_empty();
  std::string name = std::string(node->id());
  if (name == "") name = "unnamed material";
  mat->name = name;

  if (mat_node->pluginType() == "roughplastic") {
    mat->bxdf = 2;
    float eta = mat_node->property("int_ior").getNumber(1.5f);
    float alpha = mat_node->property("alpha").getNumber(1.f);
    mat->roughnessFactor = alpha;

    TPM_NAMESPACE::Color reflectance = mat_node->property("diffuse_reflectance").getColor({ 1.f, 1.f,1.f });
    mat->baseOrDiffuseColor = { reflectance.r, reflectance.g, reflectance.b };
    TPM_NAMESPACE::Color spec_reflectance = mat_node->property("specular_reflectance").getColor({ 1.f, 1.f,1.f });
    mat->floatvec_2 = { spec_reflectance.r, spec_reflectance.g, spec_reflectance.b, eta };
  } else if (mat_node->pluginType() == "diffuse") {
    mat->bxdf = 0;
    if (mat_node->property("reflectance").type() == TPM_NAMESPACE::PT_COLOR) {
      TPM_NAMESPACE::Color reflectance = mat_node->property("reflectance").getColor();
      mat->baseOrDiffuseColor = { reflectance.r, reflectance.g, reflectance.b };
    }
    else if (mat_node->property("reflectance").type() == TPM_NAMESPACE::PT_SPECTRUM) {
      TPM_NAMESPACE::Spectrum reflectance = mat_node->property("reflectance").getSpectrum();
      if (reflectance.isUniform()) {
        mat->baseOrDiffuseColor = { reflectance.uniformValue() };
      }
      else {
        mat->baseOrDiffuseColor = spectrum_to_rgb(reflectance);
      }
    }
  } else if (mat_node->pluginType() == "roughconductor") {
    mat->bxdf = 1;
    TPM_NAMESPACE::Color eta = mat_node->property("eta").getColor();
    TPM_NAMESPACE::Color k = mat_node->property("k").getColor();
    float alpha = mat_node->property("alpha").getNumber(1.f);
    mat->baseOrDiffuseColor = vec3{ k.r, k.g, k.b };
    mat->roughnessFactor = alpha;
    TPM_NAMESPACE::Color spec = mat_node->property("specular_reflectance").getColor();
    mat->floatvec_2 = vec4{ eta.r, eta.g, eta.b, spec.r };
  } else {
    mat->bxdf = 0;
  }

  for (auto const& child : mat_node->namedChildren()) {
    if (child.first == "diffuse_reflectance" ||
        child.first == "reflectance") {
      mat->basecolorTex = loadXMLTextures(child.second.get(), env);
    }
  }
  env->materials[node] = mat;
  return mat;
}

auto loadXMLMesh(TPM_NAMESPACE::Object const* node, xmlLoaderEnv* env,
  Node& gfxNode, Scene& scene) -> void {
  if (node->type() != TPM_NAMESPACE::OT_SHAPE) {
    root::print::error("gfx :: xml loader :: try to load shape not actually shape");
    return;
  }

  auto handle_material_medium = [&node, &env, &gfxNode, &scene](MeshRenderer& mesh_renderer) {
    std::optional<MaterialHandle> mat;
    if (node->anonymousChildren().size() > 0) {
      se::vec3 radiance = se::vec3(0);
      for (auto& subnode : node->anonymousChildren()) {
        if (subnode->type() == TPM_NAMESPACE::OT_BSDF) {
          // material
          TPM_NAMESPACE::Object* mat_node = subnode.get();
          mat = loadXMLMaterial(mat_node, env);
        }
        else if (subnode->type() == TPM_NAMESPACE::OT_EMITTER) {
          if (subnode->pluginType() == "area") {
            bool is_rgb;
            TPM_NAMESPACE::Color rgb = subnode->property("radiance").
              getColor(TPM_NAMESPACE::Color(0, 0, 0), &is_rgb);
            if (is_rgb) radiance = se::vec3(rgb.r, rgb.g, rgb.b);
            bool is_float;
            float intensity = subnode->property("radiance").
              getNumber(0.f, &is_float);
            if (is_float) radiance = se::vec3(intensity);
            bool is_spectrum;
            TPM_NAMESPACE::Spectrum spectrum = subnode->property("radiance").
              getSpectrum({}, &is_spectrum);
            if (is_spectrum && spectrum.isUniform())
              radiance = se::vec3(spectrum.uniformValue());
            else if (is_spectrum)
              radiance = spectrum_to_rgb(spectrum);
          }
        }
      }
      if (mat.has_value()) {
        if (radiance.r == 0
          && radiance.g == 0
          && radiance.b == 0) {
        }
        else {
          MaterialHandle mat_copy = GFXContext::load_material_empty();
          *(mat_copy.get()) = *(mat.value().get());
          mat = mat_copy;
          mat.value()->emissiveColor = radiance;
        }
        for (auto& primitive : mesh_renderer.mesh->primitives) {
          primitive.material = mat.value();
        }
        for (auto& primitive : mesh_renderer.mesh->custom_primitives) {
          primitive.material = mat.value();
        }
      }

      if (radiance.r > 0 || radiance.g > 0 || radiance.b > 0) {
        auto& light_component = scene.registry.emplace<Light>(gfxNode.entity);
        light_component.primitives.push_back(0);
        light_component.type = Light::LightType::MESH_PRIMITIVE;

        if (!mat.has_value()) {
          mat = GFXContext::load_material_empty();
          mat.value()->bxdf = 0;
          mat.value()->emissiveColor = radiance;
          for (auto& primitive : mesh_renderer.mesh->primitives) {
              primitive.material = mat.value();
            }
          for (auto& primitive : mesh_renderer.mesh->custom_primitives) {
            primitive.material = mat.value();
          }
        }
      }
    }

    for (auto& subnode : node->namedChildren()) {
      if (subnode.first == "exterior") {
        for (auto& primitive : mesh_renderer.mesh->primitives)
          primitive.exterior = env->mediums[subnode.second.get()];
        for (auto& primitive : mesh_renderer.mesh->custom_primitives)
          primitive.exterior = env->mediums[subnode.second.get()];
      }
      else if (subnode.first == "interior") {
        for (auto& primitive : mesh_renderer.mesh->primitives)
          primitive.interior = env->mediums[subnode.second.get()];
        for (auto& primitive : mesh_renderer.mesh->custom_primitives)
          primitive.interior = env->mediums[subnode.second.get()];
      }
    }
  };
  
  if (node->pluginType() == "obj") {
    std::string filename = node->property("filename").getString();
    std::string obj_path = env->directory + "/" + filename;
    MeshHandle mesh = loadObjMesh(obj_path, scene);
    
    scene.gpuScene.position_buffer->host_stamp++;
    scene.gpuScene.index_buffer->host_stamp++;
    scene.gpuScene.vertex_buffer->host_stamp++;
    auto& mesh_renderer = scene.registry.emplace<MeshRenderer>(gfxNode.entity);
    mesh_renderer.mesh = mesh;

    { // process the transform
      TPM_NAMESPACE::Transform transform =
        node->property("to_world").getTransform();
      se::mat4 mat = {
        transform.matrix[0],  transform.matrix[1],  transform.matrix[2],
        transform.matrix[3],  transform.matrix[4],  transform.matrix[5],
        transform.matrix[6],  transform.matrix[7],  transform.matrix[8],
        transform.matrix[9],  transform.matrix[10], transform.matrix[11],
        transform.matrix[12], transform.matrix[13], transform.matrix[14],
        transform.matrix[15] };
      //mat = se::transpose(mat);
      se::vec3 t, s; se::Quaternion quat;
      se::decompose(mat, &t, &quat, &s);

      auto& transformComponent = scene.registry.get<Transform>(gfxNode.entity);
      transformComponent.translation = t;
      transformComponent.scale = s;
      transformComponent.rotation = quat;
    }
    handle_material_medium(mesh_renderer);
  }
  else if (node->pluginType() == "cube") {
    std::string engine_path = RuntimeConfig::get()->string_property("engine_path");
    std::string obj_path = engine_path + "binary/resources/meshes/cube.obj";
    MeshHandle mesh = loadObjMesh(obj_path, scene);
    Mesh::CustomPrimitive cube_primitive;
    cube_primitive.primitive_type = 3;
    cube_primitive.min = -vec3(1.f);
    cube_primitive.max = vec3(1.f);
    mesh->custom_primitives.emplace_back(std::move(cube_primitive));

    scene.gpuScene.position_buffer->host_stamp++;
    scene.gpuScene.index_buffer->host_stamp++;
    scene.gpuScene.vertex_buffer->host_stamp++;
    auto& mesh_renderer = scene.registry.emplace<MeshRenderer>(gfxNode.entity);
    mesh_renderer.mesh = mesh;

    { // process the transform
      TPM_NAMESPACE::Transform transform =
        node->property("to_world").getTransform();
      se::mat4 mat = {
        transform.matrix[0],  transform.matrix[1],  transform.matrix[2],
        transform.matrix[3],  transform.matrix[4],  transform.matrix[5],
        transform.matrix[6],  transform.matrix[7],  transform.matrix[8],
        transform.matrix[9],  transform.matrix[10], transform.matrix[11],
        transform.matrix[12], transform.matrix[13], transform.matrix[14],
        transform.matrix[15] };
      //mat = se::transpose(mat);
      se::vec3 t, s; se::Quaternion quat;
      se::decompose(mat, &t, &quat, &s);

      auto& transformComponent = scene.registry.get<Transform>(gfxNode.entity);
      transformComponent.translation = t;
      transformComponent.scale = s;
      transformComponent.rotation = quat;
    }
    handle_material_medium(mesh_renderer);
  }
  else if (node->pluginType() == "rectangle") {
    std::string engine_path = RuntimeConfig::get()->string_property("engine_path");
    std::string obj_path = engine_path + "binary/resources/meshes/rectangle.obj";
    MeshHandle mesh = loadObjMesh(obj_path, scene);
    Mesh::CustomPrimitive rect_primitive;
    rect_primitive.primitive_type = 2;
    rect_primitive.min = vec3(-1.f, -1.f, 0.f);
    rect_primitive.max = vec3(+1.f, +1.f, 0.f);
    mesh->custom_primitives.emplace_back(std::move(rect_primitive));

    scene.gpuScene.position_buffer->host_stamp++;
    scene.gpuScene.index_buffer->host_stamp++;
    scene.gpuScene.vertex_buffer->host_stamp++;
    auto& mesh_renderer = scene.registry.emplace<MeshRenderer>(gfxNode.entity);
    mesh_renderer.mesh = mesh;

    { // process the transform
      TPM_NAMESPACE::Transform transform =
        node->property("to_world").getTransform();
      se::mat4 mat = {
        transform.matrix[0],  transform.matrix[1],  transform.matrix[2],
        transform.matrix[3],  transform.matrix[4],  transform.matrix[5],
        transform.matrix[6],  transform.matrix[7],  transform.matrix[8],
        transform.matrix[9],  transform.matrix[10], transform.matrix[11],
        transform.matrix[12], transform.matrix[13], transform.matrix[14],
        transform.matrix[15] };
      //mat = se::transpose(mat);
      se::vec3 t, s; se::Quaternion quat;
      se::decompose(mat, &t, &quat, &s);

      auto& transformComponent = scene.registry.get<Transform>(gfxNode.entity);
      transformComponent.translation = t;
      transformComponent.scale = s;
      transformComponent.rotation = quat;
    }
    handle_material_medium(mesh_renderer);
  }
  else if (node->pluginType() == "sphere") {
    std::string engine_path = RuntimeConfig::get()->string_property("engine_path");
    std::string obj_path = engine_path + "binary/resources/meshes/sphere.obj";
    MeshHandle mesh = loadObjMesh(obj_path, scene);
    Mesh::CustomPrimitive sphere_primitive;
    sphere_primitive.primitive_type = 1;
    sphere_primitive.min = -vec3(1.f);
    sphere_primitive.max = vec3(1.f);
    mesh->custom_primitives.emplace_back(std::move(sphere_primitive));

    scene.gpuScene.position_buffer->host_stamp++;
    scene.gpuScene.index_buffer->host_stamp++;
    scene.gpuScene.vertex_buffer->host_stamp++;
    auto& mesh_renderer = scene.registry.emplace<MeshRenderer>(gfxNode.entity);
    mesh_renderer.mesh = mesh;

    { // process the transform
      TPM_NAMESPACE::Transform transform =
          node->property("to_world").getTransform();
      float radius = node->property("radius").getNumber();
      TPM_NAMESPACE::Vector center = node->property("center").getVector();
      auto& transformComponent = scene.registry.get<Transform>(gfxNode.entity);
      transformComponent.translation = vec3(center.x, center.y, center.z);
      transformComponent.scale = vec3(radius);
      transformComponent.rotation = Quaternion();
    }

    handle_material_medium(mesh_renderer);
  }
  else {
    root::print::error("gfx :: xml loader :: Unkown mesh type:" + node->pluginType());
  }

}

auto nanovdb_float_grid_loader(nanovdb::GridHandle<nanovdb::HostBuffer>& grid) -> Medium::SampledGrid {
  nanovdb::NanoGrid<float>* floatGrid = grid.grid<float>();
  float minValue, maxValue;
  floatGrid->tree().extrema(minValue, maxValue);
  nanovdb::Vec3dBBox bbox = floatGrid->worldBBox();

  auto grid_bounds = floatGrid->indexBBox();
  int nx = floatGrid->indexBBox().dim()[0];
  int ny = floatGrid->indexBBox().dim()[1];
  int nz = floatGrid->indexBBox().dim()[2];

  std::vector<float> values;

  int z0 = grid_bounds.min()[2], z1 = grid_bounds.max()[2];
  int y0 = grid_bounds.min()[1], y1 = grid_bounds.max()[1];
  int x0 = grid_bounds.min()[0], x1 = grid_bounds.max()[0];

  bounds3 bounds = bounds3(vec3(bbox.min()[0], bbox.min()[1], bbox.min()[2]),
    vec3(bbox.max()[0], bbox.max()[1], bbox.max()[2]));

  int downsample = 0;
  // Fix the resolution to be a multiple of 2^downsample just to make
  // downsampling easy. Chop off one at a time from the bottom and top
  // of the range until we get there; the bounding box is updated as
  // well so that the remaining volume doesn't shift spatially.
  auto round = [=](int& low, int& high, float& c0, float& c1) {
    float delta = (c1 - c0) / (high - low);
    int mult = 1 << downsample; // want a multiple of this in resolution
    while ((high - low) % mult) {
      ++low;
      c0 += delta;
      if ((high - low) % mult) {
        --high;
        c1 -= delta;
      }
    }
    return high - low;
  };
  nz = round(z0, z1, bounds.pMin.z, bounds.pMax.z);
  ny = round(y0, y1, bounds.pMin.y, bounds.pMax.y);
  nx = round(x0, x1, bounds.pMin.x, bounds.pMax.x);

  int x_mid = (x0 + x1) / 2;
  int y_mid = (y0 + y1) / 2;
  int z_mid = (z0 + z1) / 2;

  for (int z = z0; z < z1; ++z)
    for (int y = y0; y < y1; ++y)
      for (int x = x0; x < x1; ++x) {
        values.push_back(floatGrid->getAccessor().getValue({ x, y, z }));
      }

  while (downsample > 0) {
    std::vector<float> v2;
    for (int z = 0; z < nz / 2; ++z)
      for (int y = 0; y < ny / 2; ++y)
        for (int x = 0; x < nx / 2; ++x) {
          auto v = [&](int dx, int dy, int dz) -> float {
            return values[(2 * x + dx) + nx * ((2 * y + dy) + ny * (2 * z + dz))];
          };
          v2.push_back((v(0, 0, 0) + v(1, 0, 0) + v(0, 1, 0) + v(1, 1, 0) +
            v(0, 0, 1) + v(1, 0, 1) + v(0, 1, 1) + v(1, 1, 1)) / 8);
        }

    values = std::move(v2);
    nx /= 2;
    ny /= 2;
    nz /= 2;
    --downsample;
  }

  Medium::SampledGrid sgrid;
  sgrid.nx = nx;
  sgrid.ny = ny;
  sgrid.nz = nz;
  sgrid.values = std::move(values);
  sgrid.bounds = bounds;
  return sgrid;
}

auto nanovdb_loader(std::string file_name, MediumHandle& medium) {
  auto list = nanovdb::io::readGridMetaData(file_name);
  bounds3 bound;
  for (auto& m : list) {
    std::string grid_name = m.gridName;
    if (grid_name == "density") {
      nanovdb::GridHandle<nanovdb::HostBuffer> handle = nanovdb::io::readGrid(file_name, m.gridName);
      medium->density = nanovdb_float_grid_loader(handle);
      bound = unionBounds(bound, medium->density.value().bounds);
    }
    if (grid_name == "temperature") {
      nanovdb::GridHandle<nanovdb::HostBuffer> handle = nanovdb::io::readGrid(file_name, m.gridName);
      medium->temperatureGrid = nanovdb_float_grid_loader(handle);
      bound = unionBounds(bound, medium->temperatureGrid.value().bounds);
    }
  }
  medium->packet.bound_min = bound.pMin;
  medium->packet.bound_max = bound.pMax;
}

auto mitsuba_volume_loader(std::string file_name) -> Medium::SampledGrid {
  // code from https://github.com/mitsuba-renderer/mitsuba/blob/master/src/volume/gridvolume.cpp#L217
  enum EVolumeType {
    EFloat32 = 1,
    EFloat16 = 2,
    EUInt8 = 3,
    EQuantizedDirections = 4
  };

  std::fstream fs(file_name.c_str(), std::fstream::in | std::fstream::binary);
  char header[3];
  fs.read(header, 3);
  if (header[0] != 'V' || header[1] != 'O' || header[2] != 'L') {
    root::print::error(std::string("Error loading volume from a file (incorrect header). Filename:") + file_name);
  }
  uint8_t version;
  fs.read((char*)&version, 1);
  if (version != 3) {
    root::print::error(std::string("Error loading volume from a file (incorrect header). Filename:") + file_name);
  }

  int type;
  fs.read((char*)&type, sizeof(int));
  if (type != EFloat32) {
    root::print::error(std::string("Unsupported volume format (only support Float32). Filename:") + file_name);
  }

  int xres, yres, zres;
  fs.read((char*)&xres, sizeof(int));
  fs.read((char*)&yres, sizeof(int));
  fs.read((char*)&zres, sizeof(int));

  int channels;
  fs.read((char*)&channels, sizeof(int));
  if (channels != 1 && channels != 3) {
    root::print::error(std::string("Unsupported volume format (wrong number of channels). Filename:") + file_name);
  }

  if (type != EFloat32) {
    root::print::error(std::string("Unsupported volume format (not Float32). Filename:") + file_name);
  }

  float xmin, ymin, zmin;
  float xmax, ymax, zmax;
  fs.read((char*)&xmin, sizeof(float));
  fs.read((char*)&ymin, sizeof(float));
  fs.read((char*)&zmin, sizeof(float));
  fs.read((char*)&xmax, sizeof(float));
  fs.read((char*)&ymax, sizeof(float));
  fs.read((char*)&zmax, sizeof(float));

  std::vector<float> raw_data(xres * yres * zres * channels, 0.f);
  fs.read((char*)raw_data.data(), sizeof(float) * xres * yres * zres * channels);

  if (channels == 1) {
    std::vector<float> data(xres * yres * zres);
    float max_data = 0;
    for (int i = 0; i < xres * yres * zres; i++) {
      data[i] = raw_data[channels * i];
      max_data = std::max(max_data, data[i]);
    }
    return Medium::SampledGrid{
      xres, yres, zres,
      std::move(data),
      bounds3{
        vec3{ xmin, ymin, zmin }, // pmin
        vec3{ xmax, ymax, zmax }, // pmax
      }, 1
    };
  }
  else {
    assert(channels == 3);
    std::vector<float> data(xres * yres * zres * 3);
    for (int i = 0; i < xres * yres * zres; i++) {
      if (channels == 1) {
        float v = raw_data[i];
        data[i * 3 + 0] = v;
        data[i * 3 + 1] = v;
        data[i * 3 + 2] = v;
      }
      else {
        data[i * 3 + 0] = raw_data[3 * i + 0];
        data[i * 3 + 1] = raw_data[3 * i + 1];
        data[i * 3 + 2] = raw_data[3 * i + 2];
      }
    }
    return Medium::SampledGrid{
      xres, yres, zres,
      std::move(data),
      bounds3{
        vec3{ xmin, ymin, zmin }, // pmin
        vec3{ xmax, ymax, zmax }, // pmax
      }, 3
    };
  }
}

void mitsuba_parse_volume_spectrum(TPM_NAMESPACE::Object* node, MediumHandle& medium) {
  float a = 1.f;
  std::string type = node->pluginType();
  if (type == "constvolume") {
    TPM_NAMESPACE::Color rgb = node->property("value").getColor();
    medium->packet.sigmaS = { rgb.r, rgb.g, rgb.b };
    medium->packet.sigmaA = { 1.f - rgb.r, 1.f - rgb.g, 1.f - rgb.b };
  }
  else if (type == "gridvolume") {
    std::string filename = node->property("filename").getString();
    auto a = mitsuba_volume_loader(filename);
  }
  else {
    root::print::error(std::string("Unknown volume type:") + type);
  }
}

SceneLoader::result_type SceneLoader::operator()(SceneLoader::from_gltf_tag, std::string const& path) {
  // load the gltf file
  tinygltf::TinyGLTF loader;
  tinygltf::Model model;
  std::string err;
  std::string warn;
  bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, path);
  if (!warn.empty()) {
    root::print::error("Scene::deserialize warn::" + warn); return nullptr;
  } if (!err.empty()) {
    root::print::error("Scene::deserialize error::" + err); return nullptr;
  } if (!ret) {
    root::print::error("Failed to parse glTF"); return nullptr;
  }
  // parse the gltf file to SE scene
  // -------------------------------------------------------------
  SceneLoader::result_type scene = std::make_shared<Scene>();
  glTFLoaderEnv env;

  // load medium if any
  if (model.extras.Has("medium")) {
    tinygltf::Value medium_extra = model.extras.Get("medium");
    int buffer_index = medium_extra.Get("buffer_id").GetNumberAsInt();
    tinygltf::Value medium_instances = medium_extra.Get("mediums");
    env.mediums.resize(medium_instances.ArrayLen());
    std::span<float> medium_buffer;
    if (buffer_index >= 0) {
      medium_buffer = std::span<float>{
        (float*)model.buffers[buffer_index].data.data(),
        model.buffers[buffer_index].data.size() / 4
      };
    }
    for (int medium_index = 0; medium_index < medium_instances.ArrayLen(); ++medium_index) {
      MediumHandle medium = GFXContext::load_medium_empty();
      tinygltf::Value instance = medium_instances.Get(medium_index);
      if (instance.Get("type").GetNumberAsInt() == 2) {
        medium->packet.type = Medium::MediumType::RGBGridMedium;
        medium->packet.aniso = {
          (float)instance.Get("aniso_x").GetNumberAsDouble(),
          (float)instance.Get("aniso_y").GetNumberAsDouble(),
          (float)instance.Get("aniso_z").GetNumberAsDouble(),
        };
        medium->packet.bound_min = {
          (float)instance.Get("bound_min_x").GetNumberAsDouble(),
          (float)instance.Get("bound_min_y").GetNumberAsDouble(),
          (float)instance.Get("bound_min_z").GetNumberAsDouble(),
        };
        medium->packet.bound_max = {
          (float)instance.Get("bound_max_x").GetNumberAsDouble(),
          (float)instance.Get("bound_max_y").GetNumberAsDouble(),
          (float)instance.Get("bound_max_z").GetNumberAsDouble(),
        };
        medium->packet.scale = (float)instance.Get("scale").GetNumberAsDouble();
        ivec3 const grid_nxyz = {
          (float)instance.Get("grid_nx").GetNumberAsInt(),
          (float)instance.Get("grid_ny").GetNumberAsInt(),
          (float)instance.Get("grid_nz").GetNumberAsInt(),
        };
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

        bounds3 bounds = { medium->packet.bound_min,medium->packet.bound_max };

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
        medium->majorantGrid->bounds = { medium->packet.bound_min, medium->packet.bound_max };
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

  // first, create the nodes
  std::vector<Node> nodes(model.nodes.size());
  for (size_t i = 0; i < model.nodes.size(); ++i) {
    nodes[i] = scene->createNode(model.nodes[i].name);
  }
  // add the hierarchy information
  for (size_t i = 0; i < model.nodes.size(); ++i) {
    auto& children = scene->registry.get<NodeProperty>(nodes[i].entity).children;
    for (auto& child_id : model.nodes[i].children) {
      children.push_back(nodes[child_id]);
    }
  }
  // set default scene
  int scene_idx = model.defaultScene;
  if (model.scenes.empty()) {
    root::print::error("SceneLoader::from_gltf_tag :: Scene is empty");
    return nullptr;
  } else if (model.defaultScene >= int(model.scenes.size())) {
    root::print::error("SceneLoader::tinygltf :: Invalid defualtScene value");
    return nullptr;
  } else if (model.defaultScene == -1) {
    // Use the first scene.
    scene_idx = 0;
  }
  // set all root nodes for the default scene
  scene->roots.clear();
  for (auto node : model.scenes[scene_idx].nodes) {
    scene->roots.push_back(nodes[node]);
  }
  std::filesystem::path filepath = path;
  std::string const directory = filepath.parent_path().string();
  env.directory = directory;

  // load tag, transform, mesh
  for (size_t i = 0; i < model.nodes.size(); ++i) {
    auto const& gltfNode = model.nodes[i];
    auto& seNode = nodes[i];
    // process the transform
    {
      auto& transform = scene->registry.get<Transform>(seNode.entity);
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
      }
    }
    // process the mesh
    if (gltfNode.mesh != -1) {
      tinygltf::Mesh& mesh_gltf = model.meshes[gltfNode.mesh];
      MeshHandle mesh = loadGLTFMesh(mesh_gltf, seNode, *scene.get(), i, &model, env);
      scene->gpuScene.position_buffer->host_stamp++;
      scene->gpuScene.index_buffer->host_stamp++;
      scene->gpuScene.vertex_buffer->host_stamp++;
      auto& mesh_renderer = scene->registry.emplace<MeshRenderer>(seNode.entity);
      mesh_renderer.mesh = mesh;

      std::vector<int> emissive_primitives;
      for (int i = 0; i < mesh->primitives.size(); ++i) {
        if (mesh->primitives[i].material.get()) {
          MaterialHandle& mat = mesh->primitives[i].material;
          if (mat->emissiveColor.r > 0 ||
              mat->emissiveColor.g > 0 ||
              mat->emissiveColor.b > 0) {
              emissive_primitives.push_back(i);
          }
        }
      }
      if (emissive_primitives.size() > 0) {
        auto& light = scene->registry.emplace<Light>(seNode.entity);
        light.primitives = emissive_primitives;
        light.type = Light::LightType::MESH_PRIMITIVE;
      }
    }
    // process the camera
    if (gltfNode.camera != -1) {
      auto& camera = scene->registry.emplace<Camera>(seNode.entity);
      auto& gltf_camera = model.cameras[gltfNode.camera];
      if (gltf_camera.type == "perspective") {
        camera.zfar = gltf_camera.perspective.zfar;
        camera.znear = gltf_camera.perspective.znear;
        camera.yfov = gltf_camera.perspective.yfov * 180 / se::float_Pi;
        camera.aspectRatio = gltf_camera.perspective.aspectRatio;
      }
    }
  }

  //processGLTFMesh(gfxNode, node_id, model, env, gfxscene, meshConfig);
  //processGLTFAnimation(gfxNode, node_id, model, env, gfxscene, meshConfig);

  //// process the meshes for all the following nodes
  //tinygltf::Node const& node = model->nodes[node_id];
  //for (uint32_t i : node.children) {
  //    GameObjectHandle subNode = gfxscene.createGameObject(gfxNode);
  //    processGLTFNode(subNode, i, model, env, gfxscene, meshConfig);
  //}

  //for (int i = 0; i < env.skinning_indices.size(); ++i) {
  //  auto& skin_indices = env.skinning_indices[i];
  //  auto& skin_matrices = env.skinning_matrices[i];
  //  for (int j = 0; j < skin_indices.size(); ++j) {
  //    int const index = skin_indices[j];
  //    GFX::GameObjectHandle go = env.node2go[index];
  //    TransformComponent* trans = gfxscene.getGameObject(go)
  //      ->getEntity().getComponent<TransformComponent>();
  //    trans->flag |= (uint32_t)TransformComponent::FlagBit::IS_SKELETON_JOINT;
  //    trans->inverseJointTransform = skin_matrices[j];
  //  }
  //}

  //// post-process all nodes
  //for (auto pair : env.node2go) {
  //  MeshReference* meshref = gfxscene.getGameObject(pair.second)
  //      ->getEntity().getComponent<MeshReference>();
  //  if (meshref) {
  //    // if has joint index buffer, assign the real handle
  //    if (meshref->mesh->jointIndexBufferInfo.size > 0) {
  //      uint64_t* indices = (uint64_t*)meshref->mesh->jointIndexBuffer_host.data;
  //      size_t size = meshref->mesh->jointIndexBuffer_host.size / sizeof(uint64_t);
  //      for (size_t i = 0; i < size; ++i) {
  //        indices[i] = env.node2go[indices[i]];
  //      }
  //    }
  //  }
  //}
  return scene;
}

SceneLoader::result_type SceneLoader::operator()(SceneLoader::from_xml_tag, std::string const& path) {
  SceneLoader::result_type scene = std::make_shared<Scene>();
  TPM_NAMESPACE::SceneLoader loader;
  try {
    auto scene_xml = loader.loadFromFile(path.c_str());
    xmlLoaderEnv env;
    env.directory = std::filesystem::path(path).parent_path().string();

    auto process_xml_node = [&](
      TPM_NAMESPACE::Object* obj
    ) {
      switch (obj->type()) {
      case TPM_NAMESPACE::OT_SCENE:
          break;
      case TPM_NAMESPACE::OT_BSDF:
          //loadMaterial(node, env);
          break;
      case TPM_NAMESPACE::OT_FILM:
          break;
      case TPM_NAMESPACE::OT_INTEGRATOR:
          break;
      case TPM_NAMESPACE::OT_MEDIUM: {
        MediumHandle medium = GFXContext::load_medium_empty();
        medium->packet.scale = obj->property("scale").getNumber(1.f);
        if (obj->pluginType() == "homogeneous") {
          medium->packet.bitfield = (uint32_t)Medium::MediumType::Homogeneous;
          TPM_NAMESPACE::Color sigma_a = obj->property("sigma_a").getColor();
          TPM_NAMESPACE::Color sigma_s = obj->property("sigma_s").getColor();
          medium->packet.sigmaA = { sigma_a.r, sigma_a.g, sigma_a.b };
          medium->packet.sigmaS = { sigma_s.r, sigma_s.g, sigma_s.b };
          medium->packet.scale = obj->property("scale").getNumber();
        }
        else if (obj->pluginType() == "heterogeneous") {
          bounds3 bound;
          for (auto child : obj->namedChildren()) {
            std::string name = child.first;
            if (name == "albedo") {
              mitsuba_parse_volume_spectrum(child.second.get(), medium);
            }
            else if (name == "density") {
              //mitsuba_parse_volume_spectrum(child.second.get(), medium);
              auto* node = child.second.get();
              std::string type = node->pluginType();
              if (type == "constvolume") {
                TPM_NAMESPACE::Color rgb = node->property("value").getColor();
                medium->packet.sigmaS = { rgb.r, rgb.g, rgb.b };
                medium->packet.sigmaA = { 1.f - rgb.r, 1.f - rgb.g, 1.f - rgb.b };
              }
              else if (type == "gridvolume") {
                std::string filename = node->property("filename").getString();
                medium->density = mitsuba_volume_loader(env.directory + "/" + filename);
                bound = unionBounds(bound, medium->density->bounds);
              }
              else {
                root::print::error(std::string("Unknown volume type:") + type);
              }
            }
          }

          medium->packet.type = Medium::MediumType::GridMedium;
          medium->packet.bound_min = bound.pMin;
          medium->packet.bound_max = bound.pMax;

          // create majorant grid
          medium->majorantGrid = Medium::MajorantGrid();
          medium->majorantGrid->res = ivec3(16, 16, 16);
          medium->majorantGrid->bounds = { medium->packet.bound_min, medium->packet.bound_max };
          medium->majorantGrid->voxels.resize(16 * 16 * 16);
          // Initialize _majorantGrid_ for _GridMedium_
          for (int z = 0; z < medium->majorantGrid->res.z; ++z)
            for (int y = 0; y < medium->majorantGrid->res.y; ++y)
              for (int x = 0; x < medium->majorantGrid->res.x; ++x) {
                bounds3 bounds = medium->majorantGrid->voxel_bounds(x, y, z);
                medium->majorantGrid->set(x, y, z, medium->density->max_value(bounds));
              }
        }
        env.mediums[obj] = medium;
        break;
      }
      case TPM_NAMESPACE::OT_PHASE:
          break;
      case TPM_NAMESPACE::OT_RFILTER:
          break;
      case TPM_NAMESPACE::OT_SAMPLER:
          break;
      case TPM_NAMESPACE::OT_SENSOR: {
        Node node = scene->createNode(obj->id());
        auto& camera = scene->registry.emplace<Camera>(node.entity);
        camera.zfar = 1000.f;
        camera.znear = 0.02f;
        camera.yfov = obj->property("fov").getNumber();
        camera.aspectRatio = 1.f;

        for (auto& child : obj->anonymousChildren()) {
          if (child->type() == TPM_NAMESPACE::OT_FILM) {
            int width = child->property("width").getInteger();
            int height = child->property("height").getInteger();
            scene->resolution = { width,height };
            camera.aspectRatio = width * 1. / height;
            if (obj->property("fov_axis").getString() == "x") {
              float tmp = width / std::tan(radians(camera.yfov) * 0.5);
              camera.yfov = 2 * degrees(std::atan(height * 1. / tmp));
              //camera.yfov
            }
          }
        }

        auto& transformComponent = scene->registry.get<Transform>(node.entity);
        auto transform = obj->property("to_world").getTransform();
        se::mat4 transform_mat = {
          transform.matrix[0], transform.matrix[1], transform.matrix[2], transform.matrix[3],
          transform.matrix[4], transform.matrix[5], transform.matrix[6], transform.matrix[7],
          transform.matrix[8], transform.matrix[9], transform.matrix[10], transform.matrix[11],
          transform.matrix[12], transform.matrix[13], transform.matrix[14], transform.matrix[15]
        };
        se::vec3 t, s; Quaternion q;
        se::Transform rotate = se::rotateY(180);
        se::decompose(transform_mat, &t, &q, &s);
        transformComponent.translation = t;
        transformComponent.scale = s;
        transformComponent.rotation = Quaternion(rotate.m) * q;

        for (auto& subnode : obj->namedChildren()) {
          if (subnode.second->type() == TPM_NAMESPACE::OT_MEDIUM)
            camera.medium = env.mediums[subnode.second.get()];
        }
        for (auto& subnode : obj->anonymousChildren()) {
          if (subnode->type() == TPM_NAMESPACE::OT_MEDIUM)
            camera.medium = env.mediums[subnode.get()];
        }
        break;
      }
      case TPM_NAMESPACE::OT_SHAPE: {
          Node node = scene->createNode(obj->id());
          loadXMLMesh(obj, &env, node, *(scene.get()));
          break;
      }
      case TPM_NAMESPACE::OT_SUBSURFACE:
          break;
      case TPM_NAMESPACE::OT_TEXTURE:
          break;
      case TPM_NAMESPACE::OT_VOLUME:
          break;
      case TPM_NAMESPACE::_OT_COUNT:
          break;
      default:
          break;
      }
    };

    //std::vector<Node> nodes(model.nodes.size());
    //for (size_t i = 0; i < model.nodes.size(); ++i) {
    //    nodes[i] = scene->createNode(model.nodes[i].name);
    //}

    for (auto& object : scene_xml.anonymousChildren()) {
      process_xml_node(object.get());
    }
    for (auto& object : scene_xml.namedChildren()) {
      process_xml_node(object.second.get());
    }

  } catch (...) {
    root::print::error("gfx :: load xml failed!");
  }
  return scene;
}

std::string loadFileAsString(const std::string& filePath) {
  std::ifstream file(filePath); // Open the file
  if (!file) {
    throw std::runtime_error("Unable to open file");
  }
  std::stringstream buffer;
  buffer << file.rdbuf(); // Read the file's content into the stringstream
  return buffer.str(); // Return the string content
}

auto loadPbrtDefineddMesh(std::vector<tiny_pbrt_loader::Point3f> p,
  std::vector<int> indices, Scene& scene) noexcept -> MeshHandle {
  // load obj file
  std::vector<float> vertexBufferV = {};
  std::vector<float> positionBufferV = {};
  std::vector<uint32_t> indexBufferWV = {};
  // create mesh resource
  MeshHandle mesh = GFXContext::load_mesh_empty();

  // check whether tangent is need in mesh attributes
  bool needTangent = false;
  for (auto const& entry : defaultMeshDataLayout.layout)
    if (entry.info == MeshDataLayout::VertexInfo::TANGENT) needTangent = true;

  // Loop over shapes
  uint64_t global_index_offset = 0;
  uint32_t submesh_vertex_offset = 0, submesh_index_offset = 0;
  for (size_t s = 0; s < 1; s++) {
    int index_offset = 0;
    uint32_t vertex_offset = 0;
    std::unordered_map<uint64_t, uint32_t> uniqueVertices{};
    vec3 position_max = vec3(-1e9);
    vec3 position_min = vec3(1e9);
    // Loop over faces(polygon)
    for (size_t f = 0; f < indices.size() / 3; f++) {
      vec3 normal;
      vec3 tangent;
      vec3 bitangent;
      if (needTangent) {
        vec3 positions[3];
        vec3 normals[3];
        vec2 uvs[3];
        for (size_t v = 0; v < 3; v++) {
          // index finding
          int idx = indices[f * 3 + v];
          positions[v] = { p[idx].v[0], p[idx].v[1], p[idx].v[2] };
          normals[v] = { 0, 0, 0 };
          uvs[v] = { 0, 0 };
        }
        vec3 edge1 = positions[1] - positions[0];
        vec3 edge2 = positions[2] - positions[0];
        vec2 deltaUV1 = uvs[1] - uvs[0];
        vec2 deltaUV2 = uvs[2] - uvs[0];

        normal = normalize(cross(edge1, edge2));

        float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);

        tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
        tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
        tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
        tangent = normalize(tangent);

        bitangent.x = f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x);
        bitangent.y = f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y);
        bitangent.z = f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z);
        bitangent = normalize(bitangent);
      }
      // Loop over vertices in the face.
      for (size_t v = 0; v < 3; v++) {
        // index finding
        int idx = indices[f * 3 + v];
        // atrributes filling
        std::vector<float> vertex = {};
        std::vector<float> position = {};
        for (auto const& entry : defaultMeshDataLayout.layout) {
          // vertex position
          if (entry.info == MeshDataLayout::VertexInfo::POSITION) {
            if (entry.format == rhi::VertexFormat::FLOAT32X3) {
              float vx = p[idx].v[0];
              float vy = p[idx].v[1];
              float vz = p[idx].v[2];
              vertex.push_back(vx);
              vertex.push_back(vy);
              vertex.push_back(vz);
              position_min = se::min(position_min, vec3{ vx,vy,vz });
              position_max = se::max(position_max, vec3{ vx,vy,vz });
              if (defaultMeshLoadConfig.usePositionBuffer) {
                position.push_back(vx);
                position.push_back(vy);
                position.push_back(vz);
              }
            }
            else {
              root::print::error(
                "GFX :: SceneNodeLoader_obj :: unwanted vertex format for "
                "POSITION attributes.");
              return MeshHandle{};
            }
          }
          else if (entry.info == MeshDataLayout::VertexInfo::NORMAL) {
            // Check if `normal_index` is zero or positive. negative = no
            // normal data
            vertex.push_back(normal.x);
            vertex.push_back(normal.y);
            vertex.push_back(normal.z);
          }
          else if (entry.info == MeshDataLayout::VertexInfo::UV) {
            vertex.push_back(0);
            vertex.push_back(0);
          }
          else if (entry.info == MeshDataLayout::VertexInfo::TANGENT) {
            if (isnan(tangent.x) || isnan(tangent.y) || isnan(tangent.z)) {
              vertex.push_back(0);
              vertex.push_back(0);
              vertex.push_back(0);
            }
            else {
              vertex.push_back(tangent.x);
              vertex.push_back(tangent.y);
              vertex.push_back(tangent.z);
            }
          }
          else if (entry.info == MeshDataLayout::VertexInfo::COLOR) {
          }
          else if (entry.info == MeshDataLayout::VertexInfo::CUSTOM) {
          }
        }

        if (defaultMeshLoadConfig.deduplication) {
          uint64_t hashed_vertex = hash_vertex(vertex);
          if (uniqueVertices.count(hashed_vertex) == 0) {
            uniqueVertices[hashed_vertex] =
              static_cast<uint32_t>(vertex_offset);
            vertexBufferV.insert(vertexBufferV.end(), vertex.begin() + 3,
              vertex.end());
            positionBufferV.insert(positionBufferV.end(), position.begin(),
              position.end());
            ++vertex_offset;
          }

          // index filling
          if (defaultMeshLoadConfig.layout.format == rhi::IndexFormat::UINT16_t)
            indexBufferWV.push_back(uniqueVertices[hashed_vertex]);
          else if (defaultMeshLoadConfig.layout.format == rhi::IndexFormat::UINT32_T)
            indexBufferWV.push_back(uniqueVertices[hashed_vertex]);
        }
        else {
          vertexBufferV.insert(vertexBufferV.end(), vertex.begin() + 3,
            vertex.end());
          positionBufferV.insert(positionBufferV.end(), position.begin(),
            position.end());
          // index filling
          if (defaultMeshLoadConfig.layout.format == rhi::IndexFormat::UINT16_t)
            indexBufferWV.push_back(vertex_offset);
          else if (defaultMeshLoadConfig.layout.format == rhi::IndexFormat::UINT32_T)
            indexBufferWV.push_back(vertex_offset);
          ++vertex_offset;
        }
      }
      index_offset += 3;
      //// per-face material
      //shapes[s].mesh.material_ids[f];
    }
    global_index_offset += index_offset;

    // load Material
    Mesh::MeshPrimitive sePrimitive;
    sePrimitive.offset = submesh_index_offset;
    sePrimitive.size = index_offset;
    sePrimitive.baseVertex = submesh_vertex_offset;
    sePrimitive.numVertex = positionBufferV.size() / 3 - submesh_vertex_offset;
    sePrimitive.max = position_max;
    sePrimitive.min = position_min;
    mesh.get()->primitives.emplace_back(std::move(sePrimitive));
    // todo:: add material
    submesh_index_offset = global_index_offset;
    submesh_vertex_offset += positionBufferV.size() / 3 - submesh_vertex_offset;
  }
  { // register mesh
    Buffer* position_buffer = scene.gpuScene.position_buffer.get();
    size_t position_size = sizeof(float) * positionBufferV.size();
    size_t position_offset = position_buffer->host.size();
    position_buffer->host.resize(position_size + position_offset);
    memcpy(&position_buffer->host[position_offset], positionBufferV.data(), position_size);
    mesh.get()->vertex_offset = position_offset;

    Buffer* index_buffer = scene.gpuScene.index_buffer.get();
    size_t index_size = sizeof(uint32_t) * indexBufferWV.size();
    size_t index_offset = index_buffer->host.size();
    index_buffer->host.resize(index_size + index_offset);
    memcpy(&index_buffer->host[index_offset], indexBufferWV.data(), index_size);
    mesh.get()->index_offset = index_offset;

    Buffer* vertex_buffer = scene.gpuScene.vertex_buffer.get();
    size_t vertex_size = sizeof(float) * vertexBufferV.size();
    size_t vertex_offset = vertex_buffer->host.size();
    vertex_buffer->host.resize(vertex_size + vertex_offset);
    memcpy(&vertex_buffer->host[vertex_offset], vertexBufferV.data(), vertex_size);

  }
  return mesh;
}

se::mat4 pbrt_mat_to_semat4x4(tiny_pbrt_loader::TransformData const& pbrt_trans) {
  return se::mat4{
  (float)pbrt_trans.m[0][0],  (float)pbrt_trans.m[0][1], (float)pbrt_trans.m[0][2], (float)pbrt_trans.m[0][3],
  (float)pbrt_trans.m[1][0],  (float)pbrt_trans.m[1][1], (float)pbrt_trans.m[1][2], (float)pbrt_trans.m[1][3],
  (float)pbrt_trans.m[2][0],  (float)pbrt_trans.m[2][1], (float)pbrt_trans.m[2][2], (float)pbrt_trans.m[2][3],
  (float)pbrt_trans.m[3][0],  (float)pbrt_trans.m[3][1], (float)pbrt_trans.m[3][2], (float)pbrt_trans.m[3][3], };
}

void FillTransfromFromPBRT(tiny_pbrt_loader::TransformData const& pbrt_trans, Transform& transformComponent) {
  se::mat4 mat = {
  (float)pbrt_trans.m[0][0],  (float)pbrt_trans.m[0][1], (float)pbrt_trans.m[0][2], (float)pbrt_trans.m[0][3],
  (float)pbrt_trans.m[1][0],  (float)pbrt_trans.m[1][1], (float)pbrt_trans.m[1][2], (float)pbrt_trans.m[1][3],
  (float)pbrt_trans.m[2][0],  (float)pbrt_trans.m[2][1], (float)pbrt_trans.m[2][2], (float)pbrt_trans.m[2][3],
  (float)pbrt_trans.m[3][0],  (float)pbrt_trans.m[3][1], (float)pbrt_trans.m[3][2], (float)pbrt_trans.m[3][3], };
  //mat = se::transpose(mat);
  se::vec3 t, s; se::Quaternion quat;
  se::decompose(mat, &t, &quat, &s);

  transformComponent.translation = t;
  transformComponent.scale = s;
  transformComponent.rotation = quat;
}

bounds3 Medium::MajorantGrid::voxel_bounds(int x, int y, int z) const {
  vec3 p0(float(x) / res.x, float(y) / res.y, float(z) / res.z);
  vec3 p1(float(x + 1) / res.x, float(y + 1) / res.y, float(z + 1) / res.z);
  return bounds3(p0, p1);
}

void Medium::MajorantGrid::set(int x, int y, int z, float v) {
  voxels[x + res.x * (y + res.y * z)] = v;
}

float Medium::SampledGrid::max_value(const bounds3& bounds) const {
  vec3 ps[2] = { vec3(bounds.pMin.x * nx - .5f, bounds.pMin.y * ny - .5f,
                         bounds.pMin.z * nz - .5f),
                 vec3(bounds.pMax.x * nx - .5f, bounds.pMax.y * ny - .5f,
                         bounds.pMax.z * nz - .5f) };
  ivec3 pi[2] = { max(ivec3(floor(ps[0])), ivec3(0, 0, 0)),
                   min(ivec3(floor(ps[1])) + ivec3(1, 1, 1),
                       ivec3(nx - 1, ny - 1, nz - 1)) };

  float maxValue;
  if (grid_channel == 1) {
    maxValue = lookup(ivec3(pi[0]));
    for (int z = pi[0].z; z <= pi[1].z; ++z)
      for (int y = pi[0].y; y <= pi[1].y; ++y)
        for (int x = pi[0].x; x <= pi[1].x; ++x)
          maxValue = std::max(maxValue, lookup(ivec3(x, y, z)));
  }
  else if (grid_channel == 3) {
    maxValue = maxComponent(lookup3(ivec3(pi[0])));
    for (int z = pi[0].z; z <= pi[1].z; ++z)
      for (int y = pi[0].y; y <= pi[1].y; ++y)
        for (int x = pi[0].x; x <= pi[1].x; ++x)
          maxValue = std::max(maxValue, maxComponent(lookup3(ivec3(x, y, z))));

  }
  return maxValue;
}

float Medium::SampledGrid::lookup(const ivec3& p) const {
  ibounds3 sampleBounds(ivec3(0, 0, 0), ivec3(nx, ny, nz));
  //if (!InsideExclusive(p, sampleBounds))
  //  return convert(T{});
  return values[(p.z * ny + p.y) * nx + p.x];
}

vec3 Medium::SampledGrid::lookup3(const ivec3& p) const {
  ibounds3 sampleBounds(ivec3(0, 0, 0), ivec3(nx, ny, nz));
  return vec3{
  values[((p.z * ny + p.y) * nx + p.x) * 3 + 0],
  values[((p.z * ny + p.y) * nx + p.x) * 3 + 1],
  values[((p.z * ny + p.y) * nx + p.x) * 3 + 2] };
}

SceneLoader::result_type SceneLoader::operator()(SceneLoader::from_pbrt_tag, std::string const& path) {
  SceneLoader::result_type scene = std::make_shared<Scene>();
  std::string fileContent = loadFileAsString(path);
  std::string dir_path = path.substr(0, path.find_last_of("/") + 1);
  std::unique_ptr<tiny_pbrt_loader::BasicScene> scene_pbrt = tiny_pbrt_loader::load_scene_from_string(fileContent, dir_path);
  std::string prefix = path.substr(0, path.find_last_of("/") + 1);
  // camera
  {
    auto camera_node = scene->createNode("camera");
    Transform& transformComponent = scene->registry.get<Transform>(camera_node.entity);
    Camera& cameraComponent = scene->registry.emplace<Camera>(camera_node.entity);
    //cameraComponent.yfov = scene_pbrt->camera.cameraFromWorld;
    FillTransfromFromPBRT(scene_pbrt->camera.cameraFromWorld, transformComponent);
    se::Transform rotate = se::rotateX(180);
    transformComponent.rotation = Quaternion(rotate.m) * transformComponent.rotation;
    transformComponent.translation.y *= -1;
    cameraComponent.yfov = scene_pbrt->camera.dict.GetOneFloat("fov", 0.f);
    float b = 1.f;
  }

  std::vector<std::optional<MaterialHandle>> material_map(scene_pbrt->materials.size());

  for (size_t i = 0; i < scene_pbrt->materials.size(); ++i) {
    auto& material = scene_pbrt->materials[i];
    std::optional<MaterialHandle> handle = std::nullopt;
    if (material.name == "interface") {
    }
    else {
      float a = 1.f;
    }
    material_map[i] = handle;
  }

  // shapes  // first, create the nodes
  //std::vector<Node> nodes(model.nodes.size());
  //for (size_t i = 0; i < model.nodes.size(); ++i) {
  //  nodes[i] = scene->createNode(model.nodes[i].name);
  //}
  //// add the hierarchy information
  //for (size_t i = 0; i < model.nodes.size(); ++i) {
  //  auto& children = scene->registry.get<NodeProperty>(nodes[i].entity).children;
  //  for (auto& child_id : model.nodes[i].children) {
  //    children.push_back(nodes[child_id]);
  //  }
  //}

  std::unordered_map<std::string, MediumHandle> medium_map;

  for (auto& medium : scene_pbrt->mediums) {
    MediumHandle medium_handle = GFXContext::load_medium_empty();
    medium_handle->packet.scale = medium.dict.GetOneFloat("scale", 1.f);
    medium_handle->packet.temperatureScale = medium.dict.GetOneFloat("temperaturescale", 1.f);
    medium_handle->packet.LeScale = medium.dict.GetOneFloat("Lescale", 1.f);
    tiny_pbrt_loader::Vector3f sigma_a = medium.dict.GetOneRGB3f("sigma_a", { 0.f ,0.f ,0.f });
    tiny_pbrt_loader::Vector3f sigma_s = medium.dict.GetOneRGB3f("sigma_s", { 0.f ,0.f ,0.f });
    medium_handle->packet.sigmaA = { sigma_a.v[0], sigma_a.v[1] ,sigma_a.v[2] };
    medium_handle->packet.sigmaS = { sigma_s.v[0], sigma_s.v[1] ,sigma_s.v[2] };
    //medium_handle->packet.geometryTransform = medium.dict[]
    std::string type = medium.dict.GetOneString("type", "");
    if (type == "nanovdb") {
      std::string filename = prefix + medium.dict.GetOneString("filename", "");
      nanovdb_loader(filename, medium_handle);
      medium_handle->packet.type = Medium::MediumType::GridMedium;

      // create majorant grid
      medium_handle->majorantGrid = Medium::MajorantGrid();
      medium_handle->majorantGrid->res = ivec3(16, 16, 16);
      medium_handle->majorantGrid->bounds = { medium_handle->packet.bound_min, medium_handle->packet.bound_max };
      medium_handle->majorantGrid->voxels.resize(16 * 16 * 16);
      // Initialize _majorantGrid_ for _GridMedium_
      for (int z = 0; z < medium_handle->majorantGrid->res.z; ++z)
        for (int y = 0; y < medium_handle->majorantGrid->res.y; ++y)
          for (int x = 0; x < medium_handle->majorantGrid->res.x; ++x) {
            bounds3 bounds = medium_handle->majorantGrid->voxel_bounds(x, y, z);
            medium_handle->majorantGrid->set(x, y, z, medium_handle->density->max_value(bounds));
          }
    }
    else if (type == "rgbgrid") {
      medium_handle->packet.type = Medium::MediumType::RGBGridMedium;
      int nx = medium.dict.GetOneInt("nx", 1);
      int ny = medium.dict.GetOneInt("ny", 1);
      int nz = medium.dict.GetOneInt("nz", 1);
      float g = medium.dict.GetOneFloat("g", 0.);
      float scale = medium.dict.GetOneFloat("scale", 1.f);
      medium_handle->packet.sigmaA = { 1,1,1 };
      medium_handle->packet.sigmaS = { 1,1,1 };
      medium_handle->packet.scale = scale;
      medium_handle->packet.aniso = { g };
      
      std::vector<tiny_pbrt_loader::Float> const& p0 = medium.dict.GetAllFloats("p0");
      std::vector<tiny_pbrt_loader::Float> const& p1 = medium.dict.GetAllFloats("p1");

      bounds3 bound;
      bound.pMin = { p0[0], p0[1], p0[2] };
      bound.pMax = { p1[0], p1[1], p1[2] };
      medium_handle->packet.bound_min = bound.pMin;
      medium_handle->packet.bound_max = bound.pMax;
      medium_handle->packet.geometryTransform = pbrt_mat_to_semat4x4(medium.objectFromRender);
      medium_handle->packet.geometryTransformInverse = pbrt_mat_to_semat4x4(medium.renderFromObject);

      // load sigma_a grid
      {
        Medium::SampledGrid sigma_a;
        sigma_a.nx = nx; sigma_a.ny = ny; sigma_a.nz = nz;
        sigma_a.bounds = bound; sigma_a.grid_channel = 3;
        auto const& sigma_a_double_array = medium.dict.GetAllFloats("sigma_a");
        sigma_a.values.resize(sigma_a_double_array.size());
        for (size_t i = 0; i < sigma_a_double_array.size(); ++i) {
          sigma_a.values[i] = sigma_a_double_array[i];
        }
        medium_handle->density = std::move(sigma_a);

        /*
        int downsample = 8;
        while (downsample > 0) {
          std::vector<float> v2;
          for (int z = 0; z < nz / 2; ++z)
            for (int y = 0; y < ny / 2; ++y)
              for (int x = 0; x < nx / 2; ++x) {
                auto v = [&](int dx, int dy, int dz) -> float {
                  return values[(2 * x + dx) + nx * ((2 * y + dy) + ny * (2 * z + dz))];
                };
                v2.push_back((v(0, 0, 0) + v(1, 0, 0) + v(0, 1, 0) + v(1, 1, 0) +
                  v(0, 0, 1) + v(1, 0, 1) + v(0, 1, 1) + v(1, 1, 1)) / 8);
              }

          values = std::move(v2);
          nx /= 2;
          ny /= 2;
          nz /= 2;
          --downsample;
        }*/
      }
      
      // load sigma_s grid
      {
        Medium::SampledGrid sigma_s;
        sigma_s.nx = nx; sigma_s.ny = ny; sigma_s.nz = nz;
        sigma_s.bounds = bound; sigma_s.grid_channel = 3;
        auto const& sigma_s_double_array = medium.dict.GetAllFloats("sigma_s");
        sigma_s.values.resize(sigma_s_double_array.size());
        for (size_t i = 0; i < sigma_s_double_array.size(); ++i) {
          sigma_s.values[i] = sigma_s_double_array[i];
        }
        medium_handle->temperatureGrid = std::move(sigma_s);
      }

      // load sigma_s grid
      
      // create majorant grid
      medium_handle->majorantGrid = Medium::MajorantGrid();
      medium_handle->majorantGrid->res = ivec3(16, 16, 16);
      medium_handle->majorantGrid->bounds = { medium_handle->packet.bound_min, medium_handle->packet.bound_max };
      medium_handle->majorantGrid->voxels.resize(16 * 16 * 16);
      // Initialize _majorantGrid_ for _RGBGridMediumm_
      for (int z = 0; z < medium_handle->majorantGrid->res.z; ++z)
        for (int y = 0; y < medium_handle->majorantGrid->res.y; ++y)
          for (int x = 0; x < medium_handle->majorantGrid->res.x; ++x) {
            bounds3 bounds = medium_handle->majorantGrid->voxel_bounds(x, y, z);
            medium_handle->majorantGrid->set(x, y, z, 
              (medium_handle->density->max_value(bounds) + medium_handle->temperatureGrid->max_value(bounds)) * scale
            );
          }

    }

    medium_map[medium.name] = medium_handle;
  }

  for (auto& shape : scene_pbrt->shapes) {
    std::vector<tiny_pbrt_loader::Point3f> p = shape.dict.GetPoint3fArray("P");
    std::vector<int> idx = shape.dict.GetIntArray("indices");
    auto node = scene->createNode(shape.name);
    NodeProperty& prop = scene->registry.get<NodeProperty>(node.entity);
    Transform& transformComponent = scene->registry.get<Transform>(node.entity);
    auto& pbrt_trans = shape.renderFromObject;
    FillTransfromFromPBRT(pbrt_trans, transformComponent);

    // if the mesh is defined by points and indices
    if (idx.size() > 0) {
      MeshRenderer& mesh_renderer = scene->registry.emplace<MeshRenderer>(node.entity);
      mesh_renderer.mesh = loadPbrtDefineddMesh(p, idx, *(scene.get()));

      if (!material_map[shape.materialIndex].has_value()) {
        
      }
      else {

      }

      if (shape.insideMedium != "") {
        auto& handle = medium_map[shape.insideMedium];
        for (auto& primitive : mesh_renderer.mesh->primitives)
          primitive.interior = handle;
        for (auto& primitive : mesh_renderer.mesh->custom_primitives)
          primitive.interior = handle;
      }

      if (shape.outsideMedium != "") {
        auto& handle = medium_map[shape.outsideMedium];
        for (auto& primitive : mesh_renderer.mesh->primitives)
          primitive.exterior = handle;
        for (auto& primitive : mesh_renderer.mesh->custom_primitives)
          primitive.exterior = handle;
      }
    }
  }

  return scene;
}

SceneLoader::result_type SceneLoader::operator()(SceneLoader::from_scratch_tag) {
  SceneLoader::result_type scene = std::make_shared<Scene>();
  // create node
  //Node cameraNode = scene->createNode("Main Camera");
  //auto& camera = scene->registry.emplace<Camera>(cameraNode.entity);
  //camera.zfar = 1000.f;
  //camera.znear = 0.02f;
  //camera.yfov = 45.f;
  //camera.aspectRatio = 1.f;
  // create default light
  Node envmapNode = scene->createNode("Environment");
  auto& envLight = scene->registry.emplace<Light>(envmapNode.entity);
  envLight.type = Light::LightType::ENVIRONMENT;
  return scene;
}

ex::resource_cache<Scene, SceneLoader> GFXContext::scenes;

auto GFXContext::load_scene_gltf(std::string const& path) noexcept -> SceneHandle {
  RUID const ruid = root::resource::queryRUID();
  auto ret = scenes.load(ruid, SceneLoader::from_gltf_tag{}, path);
  ret.first->second->dirtyFlags = (uint64_t)se::gfx::Scene::DirtyFlagBit::ALL;
  return SceneHandle{ ret.first->second };
}

auto GFXContext::load_scene_xml(std::string const& path) noexcept -> SceneHandle {
  RUID const ruid = root::resource::queryRUID();
  auto ret = scenes.load(ruid, SceneLoader::from_xml_tag{}, path);
  ret.first->second->dirtyFlags = (uint64_t)se::gfx::Scene::DirtyFlagBit::ALL;
  return SceneHandle{ ret.first->second };
}

auto GFXContext::load_scene_pbrt(std::string const& path) noexcept -> SceneHandle {
  RUID const ruid = root::resource::queryRUID();
  auto ret = scenes.load(ruid, SceneLoader::from_pbrt_tag{}, path);
  ret.first->second->dirtyFlags = (uint64_t)se::gfx::Scene::DirtyFlagBit::ALL;
  return SceneHandle{ ret.first->second };
}

auto GFXContext::create_scene(std::string const& name) noexcept -> SceneHandle {
  RUID const ruid = root::resource::queryRUID();
  auto ret = scenes.load(ruid, SceneLoader::from_scratch_tag{});
  ret.first->second->dirtyFlags = (uint64_t)se::gfx::Scene::DirtyFlagBit::ALL;
  return SceneHandle{ ret.first->second };
}
}