import os
os.environ['MUJOCO_GL'] = 'glfw'  # 强制使用 GLFW 后端

import mujoco
print("MuJoCo GL Backend:", mujoco.mujoco.glfw_get_backend())  # 输出当前后端

import numpy as np

print(mujoco.__version__)  # 检查是否有此属性
print(mujoco.mj_versionString())  # 输出 MuJoCo 版本

model = mujoco.MjModel.from_xml_string("""
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <geom name="floor" type="plane" size="1 1 0.1" rgba=".9 .9 .9 1"/>
    <body name="box" pos="0 0 0.3">
      <joint name="free" type="free"/>
      <geom name="box" type="box" size=".1 .1 .1" rgba=".9 .1 .1 1"/>
    </body>
  </worldbody>
</mujoco>
""")
data = mujoco.MjData(model)
# viewer = mujoco.viewer.launch_passive(model, data)
try:
    viewer = mujoco.viewer.launch_passive(model, data)
except Exception as e:
    print("Viewer 初始化失败，详细错误:")
    import traceback
    traceback.print_exc()
    
    # 检查 MuJoCo 内部日志
    if hasattr(mujoco, 'mj_getLastError'):
        print("MuJoCo 错误:", mujoco.mj_getLastError())

for _ in range(100):
    mujoco.mj_step(model, data)
    viewer.sync()  # 更新查看器

viewer.close()