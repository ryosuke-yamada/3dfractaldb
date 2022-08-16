# depends: pegl, glm, PyOpenGL, plyfile, OpenEXR, PIL, matplotlib

import argparse

import pegl
from OpenGL.GL import *
from OpenGL.GLU import *
import glm
import ctypes
from plyfile import PlyData, PlyElement
import numpy as np
import OpenEXR, array, Imath
from PIL import Image
import math
import time
from matplotlib import pyplot as plt
import os
import glob 
import random
from PIL import Image

def myGLDebugCallback(source, mtype, id, severity, length, message, userParam):
    print("[GLDBG]")
    if mtype == GL_DEBUG_TYPE_ERROR:
        raise SystemError("[GLDBG]")

def createShader(vertFile, fragFile):
    with open(vertFile,'r',encoding='utf8') as fp:
        vertShader_code = fp.read()
    vertShader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertShader, [vertShader_code])
    glCompileShader(vertShader)
    compiledStatus = glGetShaderiv(vertShader, GL_COMPILE_STATUS)
    infoLog = glGetShaderInfoLog(vertShader)
    if infoLog != '':
        print(infoLog.decode('ascii'))
    if compiledStatus == GL_FALSE:
        raise Exception("Compile error in vertex shader.")

    with open(fragFile,'r',encoding='utf8') as fp:
        fragShader_code = fp.read()
    fragShader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragShader, [fragShader_code])
    glCompileShader(fragShader)
    compiledStatus = glGetShaderiv(fragShader, GL_COMPILE_STATUS)
    infoLog = glGetShaderInfoLog(fragShader)
    if infoLog != '':
        print(infoLog.decode('ascii'))
    if compiledStatus == GL_FALSE:
        raise Exception("Compile error in fragment shader.")

    shader_prog = glCreateProgram()
    glAttachShader(shader_prog, vertShader)
    glAttachShader(shader_prog, fragShader)
    glDeleteShader(vertShader)
    glDeleteShader(fragShader)

    glLinkProgram(shader_prog)
    shader_linked = ctypes.c_uint(0)
    glGetProgramiv(shader_prog, GL_LINK_STATUS, ctypes.pointer(shader_linked))
    infoLog = glGetProgramInfoLog(shader_prog)
    if infoLog != '':
        print(infoLog.decode('ascii'))
    if shader_linked == GL_FALSE:
        raise Exception("Link error.")

    return shader_prog

parser = argparse.ArgumentParser()
parser.add_argument("--load_root", default='./3dfractal_render/ifs_weight/weights_ins145.csv', type = str, help="load PLY root")
parser.add_argument("--save_root", default="./dataset/EXFractalDB", type = str, help="save .png root")
args = parser.parse_args()

Width = 512
Height = 512
zNear=0.1
zFar=100.0

#print(pegl.egl_version)

dpy = pegl.Display()

egl_version_info = 'EGL version: ' + dpy.version_string
egl_vendor_info = 'Vendor: ' + dpy.vendor
egl_all_configs = dpy.get_configs()

conf = dpy.choose_config({
    pegl.ConfigAttrib.SURFACE_TYPE: pegl.SurfaceTypeFlag.PBUFFER_BIT,
    pegl.ConfigAttrib.BLUE_SIZE: 8,
    pegl.ConfigAttrib.GREEN_SIZE: 8,
    pegl.ConfigAttrib.RED_SIZE: 8,
    pegl.ConfigAttrib.DEPTH_SIZE: 8,
    pegl.ConfigAttrib.RENDERABLE_TYPE: pegl.ClientAPIFlag.OPENGL_ES,
    pegl.ConfigAttrib.CONTEXT_OPENGL_DEBUG: pegl.EGL_TRUE,
    })[0]
pegl.bind_api(pegl.ClientAPI.OPENGL_API)
ctx = conf.create_context()
surf = conf.create_pbuffer_surface({pegl.SurfaceAttrib.WIDTH: 640,
                                    pegl.SurfaceAttrib.HEIGHT: 480})
ctx.make_current(draw=surf)


print("GL VERSION: " + glGetString(GL_VERSION).decode('utf8'))
print('glDebugMessageCallback Available: %s' % bool(glDebugMessageCallback))
gl_major_version = glGetInteger(GL_MAJOR_VERSION)
gl_minor_version = glGetInteger(GL_MAJOR_VERSION)
gl_version= gl_major_version+gl_minor_version/10

unlitShader_prog = createShader("./image_render/unlit_shader.vert","./image_render/unlit_shader.frag")

fbo = glGenFramebuffers(1)
rbos = glGenRenderbuffers(2)
glBindFramebuffer(GL_FRAMEBUFFER, fbo)
glBindRenderbuffer(GL_RENDERBUFFER, rbos[0])
glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB, Width, Height)
glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbos[0])
glBindRenderbuffer(GL_RENDERBUFFER, rbos[1])
glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, Width, Height)
glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbos[1])
glBindFramebuffer(GL_FRAMEBUFFER, 0)

pbo_color, pbo_depth = glGenBuffers(2)
glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_color)
glBufferData(GL_PIXEL_PACK_BUFFER, Width*Height*3, None, GL_STREAM_COPY)
glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_depth)
glBufferData(GL_PIXEL_PACK_BUFFER, 4*Width*Height, None, GL_STREAM_COPY)
glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)

glEnable(GL_DEPTH_TEST)
axes_data = [
    0.0,0.0,0.0, 1.0,0.0,0.0,
    1.0,0.0,0.0, 1.0,0.0,0.0,
    0.0,0.0,0.0, 0.0,1.0,0.0,
    0.0,1.0,0.0, 0.0,1.0,0.0,
    0.0,0.0,0.0, 0.0,0.0,1.0,
    0.0,0.0,1.0, 0.0,0.0,1.0
]
axes_vao = glGenVertexArrays(1)
glBindVertexArray(axes_vao)
axes_vbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, axes_vbo)
glBufferData(GL_ARRAY_BUFFER, 4 * len(axes_data), (ctypes.c_float * len(axes_data))(*axes_data), GL_STATIC_DRAW)
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4*6, ctypes.c_void_p(0))
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 4*6, ctypes.c_void_p(4*3))
glBindVertexArray(0)

if not os.path.exists(args.save_root):
    os.makedirs(args.save_root)

cat_list = sorted(os.listdir(args.load_root))
for cat_ in cat_list:
    cat_path = os.path.join(args.load_root, cat_)
    ply_list = sorted(glob.glob(cat_path+"/*.ply"))

    if not os.path.exists(args.save_root + "/" + cat_):
        os.makedirs(args.save_root + "/" + cat_)
    # if not os.path.exists(save_root + "/depth/" + cat_):
    #     os.makedirs(save_root + "/depth/" + cat_)

    for i, ply in enumerate(ply_list):
        with open(ply, 'rb') as f:
            plydata = PlyData.read(f)
            pts=np.array([
            np.asarray(plydata.elements[0].data['x']),
            np.asarray(plydata.elements[0].data['y']),
            np.asarray(plydata.elements[0].data['z'])
            ], dtype=np.float32).T

            pts_colors = np.ones((pts.shape[0], 3), dtype=np.float32)
            #
            pts_vao = glGenVertexArrays(1)
            pts_vbo = glGenBuffers(2)
            glBindVertexArray(pts_vao)
            glBindBuffer(GL_ARRAY_BUFFER, pts_vbo[0])
            glBufferData(GL_ARRAY_BUFFER, 4 * pts.size, (ctypes.c_float * pts.size)(*pts.reshape(pts.size)), GL_STATIC_DRAW)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4 * 3, ctypes.c_void_p(0))
            glBindBuffer(GL_ARRAY_BUFFER, pts_vbo[1])
            glBufferData(GL_ARRAY_BUFFER, 4 * pts_colors.size, (ctypes.c_float * pts_colors.size)(*pts_colors.reshape(pts_colors.size)), GL_STATIC_DRAW)
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 4 * 3, ctypes.c_void_p(0))
            glBindVertexArray(0)

            board_data = [
                0,-1, 0, 0,0,0,
                0, 1, 0, 0,0,0,
                1, 1, 0, 1,0,0,
                1,-1, 0, 1,0,0,
            ]
            board_vao = glGenVertexArrays(1)
            board_vbo = glGenBuffers(1)
            glBindVertexArray(board_vao)
            glBindBuffer(GL_ARRAY_BUFFER, board_vbo)
            glBufferData(GL_ARRAY_BUFFER, 4 * len(board_data), (ctypes.c_float * len(board_data))(*board_data), GL_STATIC_DRAW)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4*6, ctypes.c_void_p(0))
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 4*6, ctypes.c_void_p(4*3))
            glBindVertexArray(0)

            mat_identity = glm.mat4(1)
            unlitShader_mat_proj_ul = glGetUniformLocation(unlitShader_prog, "mat_proj")
            unlitShader_mat_view_ul = glGetUniformLocation(unlitShader_prog, "mat_view")
            unlitShader_mat_model_ul = glGetUniformLocation(unlitShader_prog, "mat_model")
            glUseProgram(unlitShader_prog)
            glUniformMatrix4fv(unlitShader_mat_proj_ul, 1, GL_FALSE, glm.value_ptr(mat_identity))
            glUniformMatrix4fv(unlitShader_mat_view_ul, 1, GL_FALSE, glm.value_ptr(mat_identity))
            glUniformMatrix4fv(unlitShader_mat_model_ul, 1, GL_FALSE, glm.value_ptr(mat_identity))
            glUseProgram(0)

            mat_proj = glm.perspective(glm.radians(45.0), Width / Height, zNear, zFar)
            mat_proj_np = np.asarray(mat_proj, dtype=np.float32).reshape((4,4)).T
            mat_proj_np_inv = np.linalg.inv(mat_proj_np)

            gazePos = glm.vec3((0,0,0))
            camDist = 1
            camDir = glm.vec3((0,0,0))
            camPos  =gazePos - camDir * camDist
            upDir = glm.vec3((0, 1, 0))
            mat_view = glm.lookAt(camPos, gazePos, upDir)

            glUseProgram(unlitShader_prog)
            glUniformMatrix4fv(unlitShader_mat_proj_ul, 1, GL_FALSE, glm.value_ptr(mat_proj))
            glUniformMatrix4fv(unlitShader_mat_view_ul, 1, GL_FALSE, glm.value_ptr(mat_view))
            glUseProgram(0)


            def backprojf(z):
                c=mat_proj_np_inv[2,2]
                d=mat_proj_np_inv[2,3]
                e=mat_proj_np_inv[3,2]
                f=mat_proj_np_inv[3,3]
                return - (c*z+d) / (e*z+f)
            backproj = np.frompyfunc(backprojf,1,1)

            fCount = 0
            bfCount = 0
            loopFlg = True
            fps_text="fps: "
            bc = time.time()
            while fCount <= 11:
                print(fCount)
                tbc = time.time()
                gazePos = glm.vec3([0, 0, 0])
                camDist = 4
                theta= fCount/12*math.pi*2.0
                mat_rot = glm.mat4(1)
                mat_rot = glm.rotate(mat_rot, theta, glm.vec3(0,1,0))
                mat_rot = glm.rotate(mat_rot, glm.radians(30), glm.vec3(0,1,0))
                mat_rot = glm.rotate(mat_rot, glm.radians(random.uniform(-math.pi, math.pi)), glm.vec3(0,1,0))
                camDir = (mat_rot * glm.vec4(1,0,0,0)).xyz
                camPos = gazePos - camDir * camDist
                upDir = (mat_rot * glm.vec4(0,1,0,0)).xyz

                mat_view = glm.lookAt(camPos, gazePos, upDir)
                glUseProgram(unlitShader_prog)
                glUniformMatrix4fv(unlitShader_mat_view_ul, 1, GL_FALSE, glm.value_ptr(mat_view))
                glUseProgram(0)


                glBindFramebuffer(GL_FRAMEBUFFER, fbo)

                glClearColor(0.0, 0.0, 0.0, 0.0)
                glViewport(0, 0, Width, Height)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                glUseProgram(unlitShader_prog)
                glPointSize(3) 
                glBindVertexArray(pts_vao)
                glDrawArrays(GL_POINTS, 0, pts.shape[0])
                glBindVertexArray(0)
                glUseProgram(0)

                glFlush()

                # glBindFramebuffer(GL_FRAMEBUFFER, 0)
                # glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo)
                # glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
                # #glReadBuffer(GL_FRONT)
                # glBlitFramebuffer(
                #     0,0,Width,Height,
                #     0,0,Width,Height,
                #     GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT,
                #     GL_NEAREST
                # )
                # glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)

                glFinish()

                glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo)
                glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_color)
                glReadPixels(0, 0, Width, Height, GL_RGB, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
                glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_depth)
                glReadPixels(0, 0, Width, Height, GL_DEPTH_COMPONENT, GL_FLOAT, ctypes.c_void_p(0))
                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
                glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)

                #color
                glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_color)
                ret_color_ptr = ctypes.cast(glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY), ctypes.POINTER(ctypes.c_ubyte))
                ret_color = np.ctypeslib.as_array(ret_color_ptr, shape=(Height,Width,3))
                print(ret_color.shape)
                print(ret_color.dtype)
                glUnmapBuffer(GL_PIXEL_PACK_BUFFER)
                img_color = Image.fromarray(ret_color)
                img_color.save(args.save_root + "/" + cat_ +"/"+ cat_ +"_{:05d}_{:03d}.png".format(i, fCount))

                tcc = time.time()
                print("duration: %.1f"%(tcc-tbc))

                fCount=fCount+1