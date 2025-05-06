# import OpenGL.GL as gl
# import glfw

# def main():
#     if not glfw.init():
#         raise RuntimeError("GLFW init failed")
    
#     glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
#     glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
#     window = glfw.create_window(640, 480, "OpenGL Test", None, None)
#     if not window:
#         glfw.terminate()
#         raise RuntimeError("OpenGL 3.3 not supported!")
    
#     glfw.make_context_current(window)
#     print("OpenGL Version:", gl.glGetString(gl.GL_VERSION))
#     print("Renderer:", gl.glGetString(gl.GL_RENDERER))
    
#     glfw.terminate()

# if __name__ == "__main__":
#     main()




import glfw
from OpenGL import GL

def main():
    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    
    # 显式请求 OpenGL 3.3 上下文
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    
    window = glfw.create_window(640, 480, "OpenGL Context Test", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Window creation failed")
    
    glfw.make_context_current(window)  # 激活上下文
    print("OpenGL Version:", GL.glGetString(GL.GL_VERSION))
    print("Renderer:", GL.glGetString(GL.GL_RENDERER))
    
    glfw.terminate()

if __name__ == "__main__":
    main()