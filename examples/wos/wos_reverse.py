# Append the path of the parent directory to the system path
# to import the modules from the parent directory
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(ROOT_DIR)
# import se modules
from sibylengine.common import *
from sibylengine.editor import *
import sibylengine.pycore as se  
import sibylengine.pyeditor as sed
from examples.wos.pipeline_reverse import *

class EditorApp(EditorApplication):
    def __init__(self):
        super().__init__()
        
    def onInit(self):
        # create and bind the pipeline
        self.pipeline = ReverseWoSPipeline()
        self.pipeline.setStandardSize(se.ivec3(1024, 1024, 1))
        self.pipeline.build()
        sed.EditorBase.bindPipeline(self.pipeline.pipeline())
        
    def onUpdate(self):
        pass
        
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