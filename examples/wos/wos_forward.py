import torch
from sibylengine.common import *
from sibylengine.editor import *
import sibylengine.pycore as se  
import sibylengine.pyeditor as sed
from pipeline_forward import *

class EditorApp(EditorApplication):
    def __init__(self):
        super().__init__()
        
    def onInit(self):
        # create and bind the pipeline
        self.pipeline = WoSPipeline()
        self.pipeline.setStandardSize(se.ivec3(1024, 1024, 1))
        self.pipeline.build()
        sed.EditorBase.bindPipeline(self.pipeline.pipeline())
        
    def onCommandRecord(self, cmdEncoder: se.rhi.CommandEncoder):
        self.pipeline.execute(cmdEncoder)
        
    def onClose(self):
        del self.pipeline
        super().onClose()
        
        
editor = EditorApp()
try: editor.run()
except (RuntimeError, TypeError, NameError):
    print("Oops!  That was no valid number.  Try again...")
print("Script End Successfully!")