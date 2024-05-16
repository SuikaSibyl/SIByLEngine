import torch
import numpy as np
import se.pycore as core
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.optim as optim
import torch.nn as nn
from functools import reduce
import operator
import ctypes

def get_ptr(obj:ctypes.Structure):
  return core.ptr_from_address(ctypes.addressof(obj))

class SEContext:
  def __init__(self, with_window = False, with_editor = False):
    self.window:core.window = None
    self.swapchain:core.rhi.SwapChain = None
    self.multiFrameFlights:core.rhi.MultiFrameFlights = None
    if with_window:
      self.window = core.window.create(
        core.WindowOption(core.window.EnumVendor.GLFW, 
        "SE24", 800, 600, core.window.EnumProperty.VulkanContext))
    context_extensions = \
      int(core.rhi.EnumContextExtension.RAY_TRACING) \
      | int(core.rhi.EnumContextExtension.BINDLESS_INDEXING) \
      | int(core.rhi.EnumContextExtension.FRAGMENT_BARYCENTRIC) \
      | int(core.rhi.EnumContextExtension.CONSERVATIVE_RASTERIZATION) \
      | int(core.rhi.EnumContextExtension.COOPERATIVE_MATRIX) \
      | int(core.rhi.EnumContextExtension.CUDA_INTEROPERABILITY) \
      | int(core.rhi.EnumContextExtension.ATOMIC_FLOAT)
    self.context = core.rhi.Context.create(core.rhi.Context.EnumBackend.Vulkan)
    self.context.init(self.window, context_extensions)
    self.adapater = self.context.requestAdapter(core.rhi.RequestAdapterOptions())
    self.device = self.adapater.requestDevice()
    core.rhi.CUDAContext.initialize(self.device)
    core.gfx.Context.initialize(self.device)
    if with_window and not with_editor:
      self.swapchain = self.device.createSwapChain(core.rhi.SwapChainDescriptor())
      self.multiFrameFlights = self.device.createMultiFrameFlights(core.rhi.MultiFrameFlightsDescriptor(2, self.swapchain))

  def __del__(self):
    core.gfx.Context.finalize()
    self.multiFrameFlights = None
    self.swapchain = None
    self.device = None
    self.adapater = None
    self.context = None
    if self.window is not None:
      self.window.destroy()
      self.window = None

  def getDevice(self):
    return self.device
  
  def executeOnePass(self, se_pass):
    with torch.no_grad():
      rdrCtx = core.rdg.RenderContext()
      rdrCtx.flightIdx = 0
      rdrDat = core.rdg.RenderData()
      cmdEncDesc = core.rhi.CommandEncoderDescriptor()
      queue = self.device.getGraphicsQueue()
      commandEncoder = self.device.createCommandEncoder(cmdEncDesc)
      rdrCtx.cmdEncoder = commandEncoder
      se_pass.execute(rdrCtx, rdrDat)
      cmdBuffer = commandEncoder.finish()
      queue.submit([cmdBuffer])
      self.device.waitIdle()


class SETensor:
  def __init__(self, device = None, shape = [16], requires_grad=False, dtype:core.rhi.EnumDataType=core.rhi.EnumDataType.Float32):
    # multiply all the dimensions of the shape
    self.size = reduce(operator.mul, shape, 1)
    if device is None:
      device = core.gfx.Context.getDevice()
    # create the primal buffer
    self.prim_se = device.createBuffer(core.rhi.BufferDescriptor(self.size * 4, 
      int(core.rhi.EnumBufferUsage.STORAGE),
      core.rhi.EnumBufferShareMode.EXCLUSIVE,
      int(core.rhi.EnumMemoryProperty.DEVICE_LOCAL_BIT)))
    self.prim_cu = core.rhi.CUDAContext.toCUDABuffer(self.prim_se)
    self.prim_torch = core.rhi.toTensor(self.prim_cu, shape, dtype)
    # create the gradient buffer
    if requires_grad:
      self.grad_se = device.createBuffer(core.rhi.BufferDescriptor(self.size * 4, 
        int(core.rhi.EnumBufferUsage.STORAGE),
        core.rhi.EnumBufferShareMode.EXCLUSIVE,
        int(core.rhi.EnumMemoryProperty.DEVICE_LOCAL_BIT)))
      self.grad_cu = core.rhi.CUDAContext.toCUDABuffer(self.grad_se)
      self.grad_torch = core.rhi.toTensor(self.grad_cu, shape, dtype)
      self.prim_torch.requires_grad = True
      self.prim_torch.grad = self.grad_torch
    else:
      self.grad_se = None
      self.grad_cu = None
      self.grad_torch = None
  
  def __del__(self):
    self.prim_torch = None
    self.grad_torch = None
    self.prim_cu = None
    self.grad_cu = None
    self.prim_se = None
    self.grad_se = None

  def as_torch(self) -> torch.Tensor:
    return self.prim_torch
  def prim(self) -> core.rhi.Buffer:
    return self.prim_se
  def grad(self) -> core.rhi.Buffer:
    return self.grad_se


class SEApplication:
  def __init__(self):
    self.ctx = SEContext(with_window=True, with_editor=True)
    core.gfx.Context.createFlights(2, None)
    self.shoul_exit = False

  def onUpdateInternal(self):
    pass # virtual function
  
  def endUpdate(self):
    pass # virtual function

  def onClose(self):
    pass # virtual function
  
  def run(self):
    while not self.shoul_exit:
      # poll window events
      self.ctx.window.fetchEvents()
      self.ctx.window.endFrame()
      self.onUpdateInternal()
      self.shoul_exit = not self.ctx.window.isRunning()
    # end the application
    self.end()


  def end(self):
    self.ctx.device.waitIdle()
    self.onClose()
    del self.ctx
    self.ctx = None    

#   def addPass(self, se_pass):
#     self.passes.append(se_pass)
  
#   def run(self):
#     frameIDX = 0
#     while True:
#       self.ctx.multiFrameFlights.frameStart()
#       cmdEncoder = self.ctx.device.createCommandEncoder(self.ctx.multiFrameFlights.getCommandBuffer())
#       for se_pass in self.passes:
#         self.ctx.executeOnePass(se_pass)
#       self.ctx.device.present(self.ctx.swapchain, frameIDX)
#       frameIDX = (frameIDX + 1) % 2

class SEActiveImage:
  def __init__(self, ax, W:int, H:int, name, vmin:float, vmax:float, cmap:str) -> None:
    self.ax = ax
    self.rnd_data = np.random.rand(W,H) # random image data
    self.img_data = self.ax.imshow(self.rnd_data, vmin=vmin, vmax=vmax, cmap=cmap)
    self.ax.set_title(name)
    self.ax.axis('off')
  
  def update(self, image):
    self.img_data.set_data(image)


class SECurvePlot:
  def __init__(self, ax, name) -> None:
    self.ax = ax
    self.first_draw = True
    self.x_data, self.y_data = [], []
    # Initial plot
    self.line, = self.ax.plot(self.x_data, self.y_data)
    self.ax.set_xlim(0, 10)  # Initial x-axis limits
    self.ax.set_ylim(0, 1)  # Assuming y-data ranges within -1 to 1
    self.ax.set_title(name)
    self.max_y = 1
  
  def push(self, x, y) -> None:
    self.x_data.append(x)
    self.y_data.append(y)
    self.max_y = max(self.max_y, y)
  
  def invalidate(self) -> None:
    # Update the data of the plot
    self.line.set_data(self.x_data, self.y_data)
    # Adjust x-axis limits dynamically
    if self.x_data[-1] >= self.ax.get_xlim()[1]:
      self.ax.set_xlim(self.ax.get_xlim()[0], self.x_data[-1] + 10)
    # Adjust y-axis limits dynamically
    if self.max_y >= self.ax.get_ylim()[1]:
      self.ax.set_ylim(self.ax.get_ylim()[0], self.max_y * 1.05)


class SEPltCanvas:
  def __init__(self, nrows, ncols, figsize):
    # Enable interactive mode
    plt.ion()
    # Create a figure and axis
    self.fig, self.axs = plt.subplots(nrows, ncols, figsize=figsize)

  def flush(self):
    self.fig.canvas.flush_events()

  def has_been_closed(self):
    fig = self.fig.canvas.manager
    active_fig_managers = plt._pylab_helpers.Gcf.figs.values()
    return fig not in active_fig_managers