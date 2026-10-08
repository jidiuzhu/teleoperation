import os
import mujoco
import mujoco.viewer as viewer

ASSET_DIR = "/home/user/WorkSpace/factory_dexmanip/assets/RM75/urdf/RM75-6F/urdf"
XML_NAME = "RM75-6F_mujoco.xml"
XML_PATH = os.path.join(ASSET_DIR, XML_NAME)

os.chdir(ASSET_DIR)

print("Model：", XML_PATH)

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

print("nq =", model.nq, "nv =", model.nv, "nbody =", model.nbody)

data.qpos[:] = 0
data.qvel[:] = 0
mujoco.mj_forward(model, data)

with viewer.launch_passive(model, data) as v:
    while v.is_running():
        # mujoco.mj_step(model, data)
        v.sync()

