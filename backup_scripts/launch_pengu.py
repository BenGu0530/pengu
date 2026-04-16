import mujoco
import mujoco.viewer
import time
# Load model
model = mujoco.MjModel.from_xml_path("penguV2/scene.xml")
data = mujoco.MjData(model)

# Launch viewer passive 
with mujoco.viewer.launch(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)