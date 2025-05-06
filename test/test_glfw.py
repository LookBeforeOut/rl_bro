import glfw

from OpenGL import GL
print(GL.glGetString(GL.GL_VERSION))  # 输出当前 OpenGL 版本

try:
    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    window = glfw.create_window(640, 480, "GLFW Test", None, None)
    if not window:
        raise RuntimeError("Window creation failed")
    print("GLFW works! Close the window to continue...")
    while not glfw.window_should_close(window):
        glfw.poll_events()
finally:
    glfw.terminate()