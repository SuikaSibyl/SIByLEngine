from se.common import *
import se.pycore as se
import se.pyeditor as sed

class EditorApplication(SEApplication):
    def __init__(self):
        super().__init__()
        sed.ImGuiContext.initialize(self.ctx.device)
        sed.EditorContext.initialize()
        sed.ImGui.SetCurrentContext(sed.ImGuiContext.getRawCtx())
        self.flights = se.gfx.Context.getFlights()
        self.timer = se.timer()
        sed.EditorBase.bindInput(self.ctx.window.getInput())
        sed.EditorBase.bindTimer(self.timer)
        self.onInit()
    
    def onUpdateInternal(self):
        # finalize the override function
        sed.ImGuiContext.startNewFrame()
        self.flights.frameStart()
        # update anything here
        self.onFrameBegin()
        self.onUpdate()
        # create a command encoder
        cmdEncoder = self.ctx.device.createCommandEncoder(
            se.rhi.CommandEncoderDescriptor(self.flights.getCommandBuffer()))
        # record the command
        self.onCommandRecord(cmdEncoder)
        # submit the command
        self.ctx.device.getGraphicsQueue().submit(
            [cmdEncoder.finish()],
            self.flights.getImageAvailableSeamaphore(),
            self.flights.getRenderFinishedSeamaphore(),
            self.flights.getFence())
        
        # start record the gui
        sed.ImGuiContext.startGuiRecording()
        sed.EditorBase.onImGuiDraw()
        sed.EditorBase.onUpdate()
        self.timer.update()
        self.onDrawGui()
        sed.ImGuiContext.render(self.flights.getRenderFinishedSeamaphore())
        
        self.onFrameEnd()
        self.flights.frameEnd()

    def onInit(self):
        pass
        
    def onUpdate(self):
        pass
        
    def onDrawGui(self):
        pass
        
    def onFrameBegin(self):
        pass

    def onFrameEnd(self):
        pass
    
    def onCommandRecord(self, cmdEncoder: se.rhi.CommandEncoder):
        pass
    
    def onClose(self):
        sed.EditorBase.finalize()
        sed.ImGuiContext.finalize()
        self.flights = None