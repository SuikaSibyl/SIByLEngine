# Append the path of the parent directory to the system path
# to import the modules from the parent directory
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(ROOT_DIR)
# import se modules
from se.common import *
from se.editor import *
import se.pycore as se  
import se.pyeditor as sed
from examples.inspector.pipeline import *

class EditorApp(EditorApplication):
    def __init__(self):
        super().__init__()

    def onInit(self):
        # self.scene = se.gfx.Context.load_scene_pbrt("D:/Art/Scenes/pbrt-v4-volumes/scenes/ground_explosion/ground_explosion.pbrt")
        # self.scene = se.gfx.Context.load_scene_xml("P:/GitProjects/lajolla_public/scenes/volpath_test/vol_cbox.xml")
        # self.scene = se.gfx.Context.load_scene_xml("D:/Art/Scenes/living-room-3-mitsuba/living-room-3/scene_v3.xml")
        # self.scene = se.gfx.Context.load_scene_xml("D:/Art/Scenes/dragon/scene_v3.xml")
        # self.scene = se.gfx.Context.load_scene_xml("D:/Art/Scenes/living-room-mitsuba/scene_v3.xml")
        # self.scene = se.gfx.Context.load_scene_xml("D:/Art/Scenes/living-room-2/scene_v3.xml")
        # self.scene = se.gfx.Context.load_scene_xml("D:/Art/Scenes/kitchen-mitsuba/scene_v3.xml")
        # self.scene = se.gfx.Context.load_scene_xml("D:/Art/Scenes/living-room-2/scene_v3_colorful.xml")
        # self.scene = se.gfx.Context.load_scene_xml("D:/Art/Scenes/classroom/scene_v3.xml")
        # self.scene = se.gfx.Context.load_scene_xml("P:/GitProjects/lajolla_public/scenes/volpath_test/hetvol.xml")
        # self.scene = se.gfx.Context.load_scene_gltf("D:/Art/Scenes/veach-mis-mitsuba/scene_v3.gltf")
        # self.scene = se.gfx.Context.load_scene_gltf("examples/glt/_data/box.gltf")
        self.scene = se.gfx.Context.load_scene_xml("D:/Art/Scenes/living-room-3-mitsuba/living-room-3/scene_v3_simple.xml")

        # self.scene = se.gfx.Context.load_scene_xml("D:/Art/Scenes/house/scene_v3.xml")
        print("Scene Loaded Successfully!")
        
        sed.EditorBase.bindScene(self.scene)
        self.scene.get().updateTransform()
        self.scene.get().updateGPUScene()
        
        # create and bind the pipeline
        self.pipeline = RenderPipeline()
        self.pipeline.setStandardSize(se.ivec3(1024, 1024, 1))
        self.pipeline.build()
        sed.EditorBase.bindPipeline(self.pipeline.pipeline())
        # create and bind the scene
        # self.scene = se.gfx.Context.load_scene_gltf("examples/lighting/_data/manylight.gltf")
        # self.scene = se.gfx.Context.load_scene_gltf("examples/glt/_data/test2.gltf")
        # self.scene = se.gfx.Context.load_scene_gltf("examples/lighting/_data/onelight.gltf")
        # self.scene = se.gfx.Context.load_scene_gltf("examples/lighting/_data/manylight-2.gltf")
        # self.scene = se.gfx.Context.load_scene_xml("D:/Art/Scenes/veach-mis-mitsuba/scene_v3.xml")
        # self.scene = se.gfx.Context.load_scene_gltf("D:/Art/Scenes/veach-mis-mitsuba/scene_v3.gltf")
        # self.scene = se.gfx.Context.load_scene_xml("P:/GitProjects/lajolla_public/scenes/volpath_test/volpath_test3.xml")
        # self.scene = se.gfx.Context.load_scene_xml("D:/Art/Scenes/cornell-box-mitsuba/scene.xml")
        # self.scene = se.gfx.Context.load_scene_xml("D:/Art/Scenes/living-room-3-mitsuba/living-room-3/scene_v3.xml")
        # self.scene = se.gfx.Context.load_scene_xml("D:/Art/Scenes/living-room-3-mitsuba/living-room-3/scene_v3.xml")

        se.gfx.PMFConstructor.upload_datapack()
    
    def onUpdate(self):
        # update the scene
        self.ctx.device.waitIdle()
        self.scene.get().updateGPUScene()
    
    def onCommandRecord(self, cmdEncoder: se.rhi.CommandEncoder):
        self.pipeline.bindScene(self.scene)
        self.pipeline.execute(cmdEncoder)
    
    def onClose(self):
        se.gfx.PMFConstructor.clear_datapack()
        del self.pipeline
        del self.scene
        super().onClose()


editor = EditorApp()
editor.run()
print("Script End Successfully!")