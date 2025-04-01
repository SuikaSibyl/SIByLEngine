# import se modules
import torch
from sibylengine.common import *
from sibylengine.editor import *
import sibylengine.pycore as se
import sibylengine.pyeditor as sed

class WoSCommon:
    def __init__(self):
        self.estimator = se.Int32(2)

class WoSPass(core.rdg.ComputePass):
    def __init__(self, common:WoSCommon):
        core.rdg.ComputePass.__init__(self)
        [self.comp] = core.gfx.Context.load_shader_slang(
            "wos/_shaders/wos.slang",
            [("ComputeMain", core.rhi.EnumShaderStage.COMPUTE)], [], False)
        self.init(self.comp.get())
        
        self.common = common
        self.estimator = se.Int32(0)
        self.spp = se.Int32(1)
        self.max_steps = se.Int32(0)
        self.preview_enum = se.Int32(0)

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

        self.updateBindings(rdrCtx, [
            ["output", core.rhi.BindingResource(color.get().getUAV(0,0,1))],
        ])
        
        class PushConstant(ctypes.Structure):
          _fields_ = [("random_seed", ctypes.c_int),
                      ("render_mode", ctypes.c_int),
                      ("spp", ctypes.c_int),
                      ("max_steps", ctypes.c_int),
                      ("preview_enum", ctypes.c_int)]
        
        pConst = PushConstant(random_seed=np.random.randint(0, 1000000),
                            render_mode=self.common.estimator.get(),
                            spp=self.spp.get(),
                            max_steps=self.max_steps.get(),
                            preview_enum=self.preview_enum.get())
        
        encoder = self.beginPass(rdrCtx)
        encoder.pushConstants(get_ptr(pConst), int(core.rhi.EnumShaderStage.COMPUTE), 0, ctypes.sizeof(pConst))
        encoder.dispatchWorkgroups(int(1024 / 32), int(1024 / 4), 1)
        encoder.end()
        
    def renderUI(self):
        sed.ImGui.Combo("Estimator Mode", 
            self.common.estimator, [
            "SDF Preview",
            "Coeff Preview",
            "WoS Solution",
            "WoS w/RCV Solution",
        ])
        sed.ImGui.DragInt("SPP", self.spp, 1, 1, 100)
        sed.ImGui.DragInt("Max Steps", self.max_steps, 1, 1, 100)
        sed.ImGui.Combo("Preview Enum", self.preview_enum, [
            "DIRICHLET",
            "SOURCE",
        ])

class WoSGraph(core.rdg.Graph):
    def __init__(self, common:WoSCommon):
        core.rdg.Graph.__init__(self)
        self.fwd_pass = WoSPass(common)
        self.addPass(self.fwd_pass, "Render Pass")
        
        self.accum_pass = se.passes.AccumulatePass(se.ivec3(1024, 1024, 1))
        self.addPass(self.accum_pass, "Accum Pass")
        self.addEdge("Render Pass", "Color", "Accum Pass", "Input")
        
        self.markOutput("Accum Pass", "Output")

class WoSPipeline(core.rdg.SingleGraphPipeline):
    def __init__(self):
        core.rdg.SingleGraphPipeline.__init__(self)
        self.common = WoSCommon()
        self.graph = WoSGraph(self.common)
        self.setGraph(self.graph)