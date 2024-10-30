#pragma once
#include <map>
#include <span>
#include <vector>
#include <string>
#include <memory>
#include <string_view>

namespace tiny_pbrt_loader {
  // Hack the pbrt code features
  using Float = double;

  struct RGB {

  };

  struct Vector2f {
    float v[2];
    Vector2f() { v[0] = 0; v[1] = 0; }
    Vector2f(float x, float y) { v[0] = x; v[1] = y; }
  };

  struct Point2f {
    float v[2];
    Point2f() { v[0] = 0; v[1] = 0; }
    Point2f(float x, float y) { v[0] = x; v[1] = y; }
  };

  struct Vector3f {
    float v[3];
    Vector3f() { v[0] = 0; v[1] = 0; v[2] = 0; }
    Vector3f(float x, float y, float z) { v[0] = x; v[1] = y; v[2] = z; }
  };

  struct Point3f {
    float v[3];
    Point3f() { v[0] = 0; v[1] = 0; v[2] = 0; }
    Point3f(float x, float y, float z) { v[0] = x; v[1] = y; v[2] = z; }
  };

  struct Normal3f {
    float v[3];
    Normal3f() { v[0] = 0; v[1] = 0; v[2] = 0; }
    Normal3f(float x, float y, float z) { v[0] = x; v[1] = y; v[2] = z; }
  };

  struct Spectrum {

  };

  enum struct SpectrumType {

  };

  struct NamedTextures {

  };

  struct Camera {

  };

  struct Sampler {

  };

  struct Material {

  };

  struct Light {

  };

  struct Medium {

  };

  struct Primitive {

  };

  struct Integrator {

  };

  struct TransformData {
    Float m[4][4];
  };

  // RGBColorSpace Definition
  class RGBColorSpace {

  };

  // records the location of the corresponding statement in a file
  struct FileLoc {
    FileLoc() = default;
    FileLoc(std::string_view filename) : filename(filename) {}
    std::string ToString() const;

    std::string_view filename;
    int line = 1, column = 0;
  };

  // ParsedParameter provides the parameter type and name as strings 
  // as well as the location of the parameter in the scene description file.
  struct ParsedParameter {
    ParsedParameter(FileLoc loc) : loc(loc) {}

    void AddFloat(Float v);
    void AddInt(int i);
    void AddString(std::string_view str);
    void AddBool(bool v);

    std::string ToString() const;
    std::string type, name;
    FileLoc loc;

    std::vector<Float> floats;
    std::vector<int> ints;
    std::vector<std::string> strings;
    std::vector<uint8_t> bools;
    mutable bool lookedUp = false;
  };

  // ParameterType Definition
  enum class ParameterType {
    Boolean,
    Float,
    Integer,
    Point2f,
    Vector2f,
    Point3f,
    Vector3f,
    Normal3f,
    Spectrum,
    String,
    Texture,
    RGB,
  };

  template <ParameterType PT>
  struct ParameterTypeTraits {};

  template <>
  struct ParameterTypeTraits<ParameterType::Boolean> {
    static constexpr char typeName[] = "bool";
    static constexpr int nPerItem = 1;
    using ReturnType = uint8_t;
    static bool Convert(const uint8_t* v, const FileLoc* loc) { return *v; }
    static const auto& GetValues(const ParsedParameter& param) { return param.bools; }
  };

  constexpr char ParameterTypeTraits<ParameterType::Boolean>::typeName[];

  template <>
  struct ParameterTypeTraits<ParameterType::Float> {
    static constexpr char typeName[] = "float";
    static constexpr int nPerItem = 1;
    using ReturnType = Float;
    static Float Convert(const Float* v, const FileLoc* loc) { return *v; }
    static const auto& GetValues(const ParsedParameter& param) { return param.floats; }
  };

  constexpr char ParameterTypeTraits<ParameterType::Float>::typeName[];

  template <>
  struct ParameterTypeTraits<ParameterType::Integer> {
    static constexpr char typeName[] = "integer";
    static constexpr int nPerItem = 1;
    using ReturnType = int;
    static int Convert(const int* i, const FileLoc* loc) { return *i; }
    static const auto& GetValues(const ParsedParameter& param) { return param.ints; }
  };

  constexpr char ParameterTypeTraits<ParameterType::Integer>::typeName[];

  template <>
  struct ParameterTypeTraits<ParameterType::Point2f> {
    static constexpr char typeName[] = "point2";
    static constexpr int nPerItem = 2;
    using ReturnType = Point2f;
    static Point2f Convert(const Float* v, const FileLoc* loc) {
      return Point2f(v[0], v[1]);
    }
    static const auto& GetValues(const ParsedParameter& param) { return param.floats; }
  };

  constexpr char ParameterTypeTraits<ParameterType::Point2f>::typeName[];

  template <>
  struct ParameterTypeTraits<ParameterType::Vector2f> {
    static constexpr char typeName[] = "vector2";
    static constexpr int nPerItem = 2;
    using ReturnType = Vector2f;
    static Vector2f Convert(const Float* v, const FileLoc* loc) {
      return Vector2f(v[0], v[1]);
    }
    static const auto& GetValues(const ParsedParameter& param) { return param.floats; }
  };

  constexpr char ParameterTypeTraits<ParameterType::Vector2f>::typeName[];

  // Point3f ParameterTypeTraits Definition
  template <>
  struct ParameterTypeTraits<ParameterType::Point3f> {
    // ParameterType::Point3f Type Traits
    using ReturnType = Point3f;

    static constexpr char typeName[] = "point3";

    static const auto& GetValues(const ParsedParameter& param) { return param.floats; }

    static constexpr int nPerItem = 3;

    static Point3f Convert(const Float* f, const FileLoc* loc) {
      return Point3f(f[0], f[1], f[2]);
    }
  };

  constexpr char ParameterTypeTraits<ParameterType::Point3f>::typeName[];

  template <>
  struct ParameterTypeTraits<ParameterType::RGB> {
    static constexpr char typeName[] = "rgb";
    static constexpr int nPerItem = 3;
    using ReturnType = Vector3f;
    static Vector3f Convert(const Float* v, const FileLoc* loc) {
      return Vector3f(v[0], v[1], v[2]);
    }
    static const auto& GetValues(const ParsedParameter& param) { return param.floats; }
  };

  constexpr char ParameterTypeTraits<ParameterType::RGB>::typeName[];

  template <>
  struct ParameterTypeTraits<ParameterType::Vector3f> {
    static constexpr char typeName[] = "vector3";
    static constexpr int nPerItem = 3;
    using ReturnType = Vector3f;
    static Vector3f Convert(const Float* v, const FileLoc* loc) {
      return Vector3f(v[0], v[1], v[2]);
    }
    static const auto& GetValues(const ParsedParameter& param) { return param.floats; }
  };

  constexpr char ParameterTypeTraits<ParameterType::Vector3f>::typeName[];

  template <>
  struct ParameterTypeTraits<ParameterType::Normal3f> {
    static constexpr char typeName[] = "normal";
    static constexpr int nPerItem = 3;
    using ReturnType = Normal3f;
    static Normal3f Convert(const Float* v, const FileLoc* loc) {
      return Normal3f(v[0], v[1], v[2]);
    }
    static const auto& GetValues(const ParsedParameter& param) { return param.floats; }
  };

  constexpr char ParameterTypeTraits<ParameterType::Normal3f>::typeName[];

  template <>
  struct ParameterTypeTraits<ParameterType::String> {
    static constexpr char typeName[] = "string";
    static constexpr int nPerItem = 1;
    using ReturnType = std::string;
    static std::string Convert(const std::string* s, const FileLoc* loc) { return *s; }
    static const auto& GetValues(const ParsedParameter& param) { return param.strings; }
  };

  constexpr char ParameterTypeTraits<ParameterType::String>::typeName[];

  // adds both semantics and convenience to vectors of ParsedParameters
  struct ParameterDictionary {
    std::vector<ParsedParameter*> params;
    const RGBColorSpace* colorSpace = nullptr;
    int nOwnedParams;

    ParameterDictionary() = default;
    ParameterDictionary(std::vector<ParsedParameter*>&& i) : params(i) {}

    const FileLoc* loc(const std::string&) const;

    void FreeParameters();

    Float GetOneFloat(const std::string& name, Float def) const;
    int GetOneInt(const std::string& name, int def) const;
    bool GetOneBool(const std::string& name, bool def) const;
    std::string GetOneString(const std::string& name, const std::string& def) const;

    Point2f GetOnePoint2f(const std::string& name, Point2f def) const;
    Vector2f GetOneVector2f(const std::string& name, Vector2f def) const;
    //Point3f GetOnePoint3f(const std::string& name, Point3f def) const;
    Vector3f GetOneVector3f(const std::string& name, Vector3f def) const;
    Normal3f GetOneNormal3f(const std::string& name, Normal3f def) const;
    Vector3f GetOneRGB3f(const std::string& name, Vector3f def) const;

    Spectrum GetOneSpectrum(const std::string& name, Spectrum def,
      SpectrumType spectrumType) const;

    std::vector<Float> GetFloatArray(const std::string& name) const;
    std::vector<int> GetIntArray(const std::string& name) const;
    std::vector<uint8_t> GetBoolArray(const std::string& name) const;

    std::vector<Point2f> GetPoint2fArray(const std::string& name) const;
    std::vector<Vector2f> GetVector2fArray(const std::string& name) const;
    std::vector<Point3f> GetPoint3fArray(const std::string& name) const;
    std::vector<Vector3f> GetVector3fArray(const std::string& name) const;
    std::vector<Normal3f> GetNormal3fArray(const std::string& name) const;
    std::vector<Spectrum> GetSpectrumArray(const std::string& name,
      SpectrumType spectrumType) const;
    std::vector<std::string> GetStringArray(const std::string& name) const;

    void ReportUnused() const;

    std::vector<Float> const& GetAllFloats(const std::string& name) const;

    // ParameterDictionary Private Methods
    template <ParameterType PT>
    typename ParameterTypeTraits<PT>::ReturnType lookupSingle(
      const std::string& name,
      typename ParameterTypeTraits<PT>::ReturnType defaultValue) const;

    template <ParameterType PT>
    std::vector<typename ParameterTypeTraits<PT>::ReturnType> lookupArray(
      const std::string& name) const;

    template <typename ReturnType, typename G, typename C>
    std::vector<ReturnType> lookupArray(const std::string& name, ParameterType type,
      const char* typeName, int nPerItem, G getValues,
      C convert) const;
  };

  struct SceneEntity {
    std::string name;
    FileLoc loc;
    ParameterDictionary dict;
  };

  struct AnimatedShapeSceneEntity : public SceneEntity {

  };

  struct InstanceSceneEntity : public SceneEntity {

  };

  struct CameraSceneEntity : public SceneEntity {
    std::string outsideMedium;
    TransformData cameraFromWorld;
  };

  struct MediumSceneEntity : public SceneEntity {
    TransformData renderFromObject;
    TransformData objectFromRender;
  };

  struct TextureSceneEntity : public SceneEntity {

  };

  struct LightSceneEntity : public SceneEntity {

  };

  struct ShapeSceneEntity : public SceneEntity {
    TransformData renderFromObject;
    TransformData objectFromRender;
    std::string insideMedium;
    std::string outsideMedium;
    int materialIndex;
  };

  struct BasicScene {
    //void SetOptions(SceneEntity filter, SceneEntity film, CameraSceneEntity camera,
    //  SceneEntity sampler, SceneEntity integrator, SceneEntity accelerator);
    void AddNamedMaterial(std::string name, SceneEntity material);
    int AddMaterial(SceneEntity material);
    void AddMedium(MediumSceneEntity medium);
    //void AddFloatTexture(std::string name, TextureSceneEntity texture);
    //void AddSpectrumTexture(std::string name, TextureSceneEntity texture);
    //void AddLight(LightSceneEntity light);
    int AddAreaLight(SceneEntity light);
    void AddShapes(std::span<ShapeSceneEntity> shape);
    //void AddAnimatedShape(AnimatedShapeSceneEntity shape);
    //void AddInstanceDefinition(InstanceDefinitionSceneEntity instance);
    //void AddInstanceUses(std::span<InstanceSceneEntity> in);

    CameraSceneEntity camera;
    std::vector<SceneEntity> materials;
    std::vector<SceneEntity> areaLights;
    std::vector<MediumSceneEntity> mediums;
    std::vector<ShapeSceneEntity> shapes;
    std::vector<std::pair<std::string, SceneEntity>> namedMaterials;
  };

  std::unique_ptr<BasicScene> load_scene_from_string(std::string str, std::string dir_path = "");
}