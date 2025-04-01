# import se modules
from sibylengine.common import *
from sibylengine.editor import *
import sibylengine.pycore as se
import sibylengine.pyeditor as sed

class WoSReverseCommon:
    def __init__(self):
        self.spp = se.Int32(1)
        self.max_steps = se.Int32(15)
        self.maximum_threads = 512 * 128
        self.maximum_spp = 1
        self.maximum_steps = 100
        self.total_spp = 0
        
    def num_greens(self):
        return self.maximum_threads * self.maximum_spp * self.max_steps.get()

class ReverseWoSPass(core.rdg.ComputePass):
    def __init__(self, common:WoSReverseCommon):
        core.rdg.ComputePass.__init__(self)
        self.common = common
        [self.comp] = core.gfx.Context.load_shader_slang(
            "wos/_shaders/reverse-wos.slang",
            [("ComputeMain", core.rhi.EnumShaderStage.COMPUTE)], [], False)
        self.init(self.comp.get())
        
    def reflect(self) -> core.rdg.PassReflection:
        reflector = core.rdg.PassReflection()
        reflector.addOutput("Greens")\
            .isBuffer().withSize(self.common.maximum_spp 
                * self.common.maximum_threads * 8
                * self.common.maximum_steps * 4).withUsages(
                int(se.rhi.EnumBufferUsage.STORAGE))\
            .consume(se.rdg.BufferInfo.ConsumeEntry()\
                .setAccess(int(se.rhi.AccessFlagBits.SHADER_WRITE_BIT))
                .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        return reflector

    def execute(self, rdrCtx:core.rdg.RenderContext, rdrDat:core.rdg.RenderData):
        greens = rdrDat.getBuffer("Greens")
        self.updateBindings(rdrCtx, [
            ["rw_spheres", greens.get().getBindingResource()],
        ])
        
        class PushConstant(ctypes.Structure):
          _fields_ = [("random_seed", ctypes.c_int),
                      ("spp", ctypes.c_int),
                      ("threads", ctypes.c_int),
                      ("max_steps", ctypes.c_int),
                    ]
        
        pConst = PushConstant(
            random_seed=np.random.randint(0, 1000000),
            spp=self.common.spp.get(),
            threads=self.common.maximum_threads,
            max_steps=self.common.max_steps.get())
        
        encoder = self.beginPass(rdrCtx)
        encoder.pushConstants(get_ptr(pConst), 
            int(core.rhi.EnumShaderStage.COMPUTE), 0, ctypes.sizeof(pConst))
        encoder.dispatchWorkgroups(int(self.common.maximum_threads / 128), 1, 1)
        encoder.end()
        
    def renderUI(self):
        pass


class GreenSplatPass(core.rdg.RenderPass):
    def __init__(self, common:WoSReverseCommon):
        core.rdg.RenderPass.__init__(self)
        self.common = common
        [self.vert, self.frag] = core.gfx.Context.load_shader_slang(
            "wos/_shaders/green-splat.slang",
            [("vertexMain", core.rhi.EnumShaderStage.VERTEX),
             ("fragmentMain", core.rhi.EnumShaderStage.FRAGMENT)], [], False)
        self.init(self.vert.get(), self.frag.get())

    def reflect(self) -> core.rdg.PassReflection:
        reflector = core.rdg.PassReflection()
        reflector.addInput("Greens")\
            .isBuffer().withUsages(
                int(se.rhi.EnumBufferUsage.STORAGE))\
            .consume(se.rdg.BufferInfo.ConsumeEntry()\
                .setAccess(int(se.rhi.AccessFlagBits.SHADER_READ_BIT))
                .addStage(int(se.rhi.PipelineStageBit.VERTEX_SHADER_BIT)
                        | int(se.rhi.PipelineStageBit.FRAGMENT_SHADER_BIT)))
        reflector.addOutput("Color")\
            .isTexture().withSize(se.vec3(1, 1, 1))\
            .withFormat(se.rhi.TextureFormat.RGBA32_FLOAT)\
            .withUsages(se.rhi.TextureUsageBit.COLOR_ATTACHMENT)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(
                se.rdg.TextureInfo.ConsumeType.ColorAttachment)\
                .setAttachmentLoc(0).setTargetBlenderFactor(se.rhi.BlendFactor.ONE))
        return reflector

    def execute(self, rdrCtx:core.rdg.RenderContext, rdrDat:core.rdg.RenderData):
        color:se.gfx.TextureHandle = rdrDat.getTexture("Color")
        greens = rdrDat.getBuffer("Greens")

        self.setRenderPassDescriptor(se.rhi.RenderPassDescriptor([
            se.rhi.RenderPassColorAttachment(color.get().getRTV(0,0,1), None, 
                se.vec4(0,0,0,1), se.rhi.LoadOp.CLEAR, se.rhi.StoreOp.STORE),
        ], se.rhi.RenderPassDepthStencilAttachment()))
        
        self.updateBindings(rdrCtx, [
            ["sb_spheres", greens.get().getBindingResource()],
        ])
        
        class PushConstant(ctypes.Structure):
          _fields_ = [("resolution_x", ctypes.c_int),
                      ("resolution_y", ctypes.c_int)]
        
        pConst = PushConstant(resolution_x=1024,
                              resolution_y=1024)
        
        encoder = self.beginPass(rdrCtx, color.get())
        # encoder.pushConstants(get_ptr(pConst), int(core.rhi.EnumShaderStage.VERTEX), 0, ctypes.sizeof(pConst))
        encoder.draw(30, self.common.num_greens(), 0, 0)
        encoder.end()
        
    def renderUI(self):
        pass        
        
        
class GreenAccumPass(core.rdg.ComputePass):
    def __init__(self, common:WoSReverseCommon):
        core.rdg.ComputePass.__init__(self)
        self.common = common
        [self.comp] = core.gfx.Context.load_shader_slang(
            "wos/_shaders/green-accum.slang",
            [("ComputeMain", core.rhi.EnumShaderStage.COMPUTE)], [], False)
        self.init(self.comp.get())
        
    def reflect(self) -> core.rdg.PassReflection:
        reflector = core.rdg.PassReflection()
        reflector.addInputOutput("Color")\
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
        
        encoder = self.beginPass(rdrCtx)
        encoder.dispatchWorkgroups(int(1024 / 32), int(1024 / 4), 1)
        encoder.end()
        
    def renderUI(self):
        pass


class WoSGraph(core.rdg.Graph):
    def __init__(self, common:WoSReverseCommon):
        core.rdg.Graph.__init__(self)
        self.rvs_pass = ReverseWoSPass(common)
        self.addPass(self.rvs_pass, "Walk Pass")
        
        self.splat_pass = GreenSplatPass(common)
        self.addPass(self.splat_pass, "Splat Pass")
        self.addEdge("Walk Pass", "Greens", "Splat Pass", "Greens")
        
        self.pp_pass = GreenAccumPass(common)
        self.addPass(self.pp_pass, "Post Pass")
        self.addEdge("Splat Pass", "Color", "Post Pass", "Color")
        
        self.accum_pass = se.passes.AccumulatePass(se.ivec3(1024, 1024, 1))
        self.addPass(self.accum_pass, "Accum Pass")
        self.addEdge("Post Pass", "Color", "Accum Pass", "Input")
        
        self.markOutput("Accum Pass", "Output")

class ReverseWoSPipeline(core.rdg.SingleGraphPipeline):
    def __init__(self):
        core.rdg.SingleGraphPipeline.__init__(self)
        self.common = WoSReverseCommon()
        self.graph = WoSGraph(self.common)
        self.setGraph(self.graph)
        
    def renderUI(self):
        sed.ImGui.DragInt("SPP", self.common.spp, 1, 1, 1)
        sed.ImGui.DragInt("Max Steps", self.common.max_steps, 1, 1, self.common.maximum_steps)