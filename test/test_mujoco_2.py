import mujoco
import numpy as np

from OpenGL import GL
print(GL.glGetString(GL.GL_VERSION))  # 输出当前 OpenGL 版本

# 检查版本
print("MuJoCo Version:", mujoco.mj_versionString())

# 创建一个简单模型
model = mujoco.MjModel.from_xml_string("""
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="box" pos="0 0 0.3">
      <joint type="free"/>
      <geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
""")
data = mujoco.MjData(model)

# 基础仿真测试
for _ in range(50):
    mujoco.mj_step(model, data)
    print("Box position:", data.qpos[0:3])

# 尝试渲染（如果前几步成功但这里失败，说明渲染依赖有问题）
try:
    renderer = mujoco.Renderer(model)
    renderer.update_scene(data)
    img = renderer.render()
    print("Render successful! Image shape:", img.shape)
except Exception as e:
    print("Render failed:", e)