# import se modules
from se.common import *
from se.editor import *
import se.pycore as se
import se.pyeditor as sed
import matplotlib.pyplot as plt

class LightingCommon:
    def __init__(self):
        self.inspector_mode = se.Int32(0)
        self.wireframe_thickness = se.Float32(0.01)
        self.wireframe_color = se.vec3(0, 0, 0)
        self.wireframe_smoothness = se.Float32(0.5)

class ForwardPass(core.rdg.ComputePass):
    def __init__(self, common:LightingCommon):
        core.rdg.ComputePass.__init__(self)
        [self.comp] = core.gfx.Context.load_shader_slang(
            "inspector/_shaders/render.slang",
            [("ComputeMain", core.rhi.EnumShaderStage.COMPUTE)], [], False)
        self.init(self.comp.get())

        self.common = common
        self.envmap = se.lights.EnvmapLight(
            "examples/lighting/_data/skylight-colorfull.exr",
            se.lights.EnvmapLight.ImportanceType.Luminance)
        # self.brdf = se.bxdfs.EPFLBrdf("C:/Users/suika/Downloads/cc_ibiza_sunset_rgb.bsdf")
        # se.bxdfs.EPFLBrdf.updateGPUResource()
    
    def __del__(self):
        del self.envmap
        # del self.brdf

    def reflect(self) -> core.rdg.PassReflection:
        reflector = core.rdg.PassReflection()
        reflector.addOutput("Color")\
            .isTexture().withSize(se.vec3(1, 1, 1))\
            .withFormat(se.rhi.TextureFormat.RGBA32_FLOAT)\
            .withUsages(se.rhi.TextureUsageBit.STORAGE_BINDING)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.StorageBinding)\
                .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        return reflector
    
    def execute(self, rdrCtx:core.rdg.RenderContext, rdrDat:core.rdg.RenderData):
        color = rdrDat.getTexture("Color")
        scene = rdrDat.getScene()

        self.updateBindings(rdrCtx, [
            ["output", core.rhi.BindingResource(color.get().getUAV(0,0,1))],
            # ["u_envmap", core.rhi.BindingResource(self.envmap.get_texture().get().getUAV(0,0,1))],
            # ["se_cdf_buffer", se.gfx.PMFConstructor.binding_resource_buffer()],
            # ["epfl_data_tensors", se.bxdfs.EPFLBrdf.bindingResourceBuffer()],
            # ["epfl_materials", se.bxdfs.EPFLBrdf.bindingResourceBRDFs()],
            ["GPUScene_camera", scene.get().getGPUScene().bindingResourceCamera()],
            ["GPUScene_index", scene.get().getGPUScene().bindingResourceIndex()],
            ["GPUScene_vertex", scene.get().getGPUScene().bindingResourceVertex()],
            ["GPUScene_geometry", scene.get().getGPUScene().bindingResourceGeometry()],
            ["GPUScene_material", scene.get().getGPUScene().bindingResourceMaterial()],
            ["GPUScene_position", scene.get().getGPUScene().bindingResourcePosition()],
            ["GPUScene_tlas", scene.get().getGPUScene().bindingResourceTLAS()],
            ["GPUScene_light", scene.get().getGPUScene().bindingResourceLight()],
            ["GPUScene_tlas", scene.get().getGPUScene().bindingResourceTLAS()],
            ["GPUScene_textures", scene.get().getGPUScene().bindingResourceTextures()],
            ["GPUScene_light_bvh", scene.get().getGPUScene().bindingResourceLightBVH()],
            ["GPUScene_light_trail", scene.get().getGPUScene().bindingResourceLightTrail()],
            ["GPUScene_param_packets", scene.get().getGPUScene().bindingResourceParamPacket()],
            ["GPUScene_param_primal", scene.get().getGPUScene().bindingResourceParamPrimal()],
            ["GPUScene_param_gradient", scene.get().getGPUScene().bindingResourceParamGradient()],
            ["GPUScene_description", scene.get().getGPUScene().bindingSceneDescriptor()],
        ])
        
        class PushConstant(ctypes.Structure):
          _fields_ = [("random_seed", ctypes.c_int),
                      ("inspector_mode", ctypes.c_int),
                      ("wireframe_thickness", ctypes.c_float),
                      ("wireframe_smoothness", ctypes.c_float),
                      ("wireframe_color", ctypes.c_float * 3)]

        pConst = PushConstant(random_seed=np.random.randint(0, 1000000),
                            inspector_mode=self.common.inspector_mode.get(),
                            wireframe_thickness=self.common.wireframe_thickness.get(),
                            wireframe_smoothness=self.common.wireframe_smoothness.get(),
                            wireframe_color=(self.common.wireframe_color.x,
                                             self.common.wireframe_color.y,
                                             self.common.wireframe_color.z))
        
        encoder = self.beginPass(rdrCtx)
        encoder.pushConstants(get_ptr(pConst), int(core.rhi.EnumShaderStage.COMPUTE), 0, ctypes.sizeof(pConst))
        encoder.dispatchWorkgroups(int(1024 / 32), int(1024 / 4), 1)
        encoder.end()

    def renderUI(self):
        sed.ImGui.Combo("Inspector Mode", 
            self.common.inspector_mode, [
            "Albedo",
            "Emission",
            "Face Forward",
            "Texcoord",
            "Normal",
            "Position",
            "Additional Texture"
        ])
        sed.ImGui.DragFloat("Wireframe Smoothness", self.common.wireframe_smoothness, 0.01, 0.0, 1.0)
        sed.ImGui.DragFloat("Wireframe Thickness", self.common.wireframe_thickness, 0.01, 0.0, 1.0)
        sed.ImGui.ColorEditVec3("Wireframe Color", self.common.wireframe_color)

class RenderGraph(core.rdg.Graph):
    def __init__(self, common:LightingCommon):
        core.rdg.Graph.__init__(self)
        self.fwd_pass = ForwardPass(common)
        self.addPass(self.fwd_pass, "Render Pass")
        
        self.accum_pass = se.passes.AccumulatePass(se.ivec3(1024, 1024, 1))
        self.addPass(self.accum_pass, "Accum Pass")
        self.addEdge("Render Pass", "Color", "Accum Pass", "Input")
        
        self.markOutput("Accum Pass", "Output")

class RenderPipeline(core.rdg.SingleGraphPipeline):
    def __init__(self):
        core.rdg.SingleGraphPipeline.__init__(self)
        self.common = LightingCommon()
        self.graph = RenderGraph(self.common)
        self.setGraph(self.graph)