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
from typing import List

# def myGLDebugCallback(source, mtype, id, severity, length, message, userParam):
#     print("[GLDBG]")
#     if mtype == GL_DEBUG_TYPE_ERROR:
#         raise SystemError("[GLDBG]")

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

import torch, torchvision

class DynamicRenderer(torch.utils.data.Dataset):
    def __init__(self, use_gpus : List[int], data_dir : str, base_seed: int, transform=None):
        self.use_gpus = use_gpus
        self.worker_id = -1
        self.base_seed = base_seed
        self.transform = transform

        self.cat_list = sorted(os.listdir(data_dir))
        self.ply_lists = []
        self.ptss = []
        for cat_id, cat_ in enumerate(self.cat_list):
            cat_path = os.path.join(data_dir, cat_)
            ply_list = sorted(glob.glob(cat_path+"/*.ply"))
            self.ply_lists += [[cat_id, ply] for ply in ply_list]
        for cat_id, ply in self.ply_lists: 
            with open(ply, 'rb') as f:
                plydata = PlyData.read(f)
                pts = np.array([
                    np.asarray(plydata.elements[0].data['x']),
                    np.asarray(plydata.elements[0].data['y']),
                    np.asarray(plydata.elements[0].data['z'])
                ], dtype=np.float32).T
            self.ptss.append([cat_id, pts])
        

    def open(self, gpu_id):
        if gpu_id<0 or gpu_id>=len(self.use_gpus):
            raise ValueError("Invalid gpu id : %d" % gpu_id)

        self.dpy = pegl.Display(self.use_gpus[gpu_id])
        conf = self.dpy.choose_config({
            pegl.ConfigAttrib.SURFACE_TYPE: pegl.SurfaceTypeFlag.PBUFFER_BIT,
            pegl.ConfigAttrib.BLUE_SIZE: 8,
            pegl.ConfigAttrib.GREEN_SIZE: 8,
            pegl.ConfigAttrib.RED_SIZE: 8,
            pegl.ConfigAttrib.DEPTH_SIZE: 8,
            pegl.ConfigAttrib.RENDERABLE_TYPE: pegl.ClientAPIFlag.OPENGL,
            #pegl.ConfigAttrib.CONTEXT_OPENGL_DEBUG: pegl.EGL_TRUE,
            })[0]
        pegl.bind_api(pegl.ClientAPI.OPENGL_API)
        self.ctx = conf.create_context()
        self.surf = conf.create_pbuffer_surface({pegl.SurfaceAttrib.WIDTH: 640,
                                            pegl.SurfaceAttrib.HEIGHT: 480})
        self.ctx.make_current(draw=self.surf)

        # generate gpu memories
        self.pts_vao
        self.pts_vbo
        # self.pts_fbo
        # self.pts_rbo_color
        # self.pts_rbo_depth
        # self.pts_pbo_color
        # self.pts_pbo_depth

        # prepare shaders
        self.shader

        return
    
    def __len__(self):
        return len(self.ply_lists)

    def __getitem__(self, id):
        cat_id, pts = self.ply_lists[id]
        # send pts to gpu
        # glBindBuffer(GL_ARRAY_BUFFER, vbo)
        # glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*3*npts, pts)
        # glBindBuffer(GL_ARRAY_BUFFER, 0)

        # render
        # glDrawArrays()
        # glFlush()
        # glFinish()
        # glReadPixels()

        img = None

        if self.transform is not None:
            img=self.transform(img)

        return cat_id, img



def worker_init_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    _seed=info.dataset.base_seed+worker_id
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)
    info.dataset.open(worker_id)
    return

if __name__=="__main__":
    use_gpus=[0,1,2,3]
    dataset = DynamicRenderer(use_gpus=use_gpus, data_dir="./", base_seed = 7743, transform = None)
    dataloader = torch.utils.data.DataLoader(dataset, num_worker=len(use_gpus), batch_size=256, shuffle=True,
                                             drop_last=True, worker_init_fn=worker_init_fn, persistent_workers=True)


    # test first loading
    it=iter(dataloader)
    labels, imgs = next(it)

    pimg=Image.fromarray(np.array(torchvision.utils.make_grid(imgs,nrow=16).permute(1,2,0)))
    pimg.resize((1024,1024))
    pimg.save("test.png", Image.BILINEAR)
