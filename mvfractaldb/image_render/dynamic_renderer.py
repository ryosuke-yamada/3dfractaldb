# depends: pegl, PyGLM (not glm), PyOpenGL, plyfile, OpenEXR, PIL, matplotlib

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
from typing import List,Tuple
from tqdm import tqdm
from joblib import Parallel, delayed

# def myGLDebugCallback(source, mtype, id, severity, length, message, userParam):
#     print("[GLDBG]")
#     if mtype == GL_DEBUG_TYPE_ERROR:
#         raise SystemError("[GLDBG]")

def createShader(vertFile:str, fragFile:str) -> GLuint:
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
        self.width=480
        self.height=480
        self.npts=4000

        print("Constructing ply file list from %s..."%data_dir)
        self.cat_list = sorted(os.listdir(data_dir))
        self.ply_lists = []
        self.ptss = []
        for cat_id, cat_ in tqdm(enumerate(self.cat_list)):
            cat_path = os.path.join(data_dir, cat_)
            ply_list = sorted(glob.glob(cat_path+"/*.ply"))
            self.ply_lists += [(cat_id, ply) for ply in ply_list]
        print("Done.")
        self.ply_lists=self.ply_lists[0:512]
        print("Loading ply files...")
        def load_ply(cat_ply) -> Tuple[int,np.ndarray]:
            cat_id, ply = cat_ply
            with open(ply, 'rb') as f:
                plydata = PlyData.read(f)
                pts = np.array([
                    np.asarray(plydata.elements[0].data['x']),
                    np.asarray(plydata.elements[0].data['y']),
                    np.asarray(plydata.elements[0].data['z'])
                ], dtype=np.float32).T
                return (cat_id, pts)
        self.ptss=Parallel(n_jobs=-1,verbose=10)(delayed(load_ply)(cat_ply) for cat_ply in self.ply_lists)
        # for cat_id, ply in self.ply_lists: 
        #     pts=load_ply(ply)
        #     self.ptss.append((cat_id, pts))
        print("Done.")
        
    def open(self, gpu_id:int):
        '''open
        Initialize OpenGL.
        An OpenGL context requires a process. So, it shoud be executed on worker_init_fn.
        '''
        if gpu_id<0 or gpu_id>=len(self.use_gpus):
            raise ValueError("Invalid gpu id : %d" % gpu_id)

        self.dpy = pegl.Display(self.use_gpus[gpu_id])
        print("Device %d: "%gpu_id,self.dpy.vendor,self.dpy.version_string)
        pegl.bind_api(pegl.ClientAPI.OPENGL_API)
        conf:pegl.Config = self.dpy.choose_config({
            pegl.ConfigAttrib.SURFACE_TYPE: pegl.SurfaceTypeFlag.PBUFFER_BIT,
            pegl.ConfigAttrib.BLUE_SIZE: 8,
            pegl.ConfigAttrib.GREEN_SIZE: 8,
            pegl.ConfigAttrib.RED_SIZE: 8,
            pegl.ConfigAttrib.ALPHA_SIZE: 8,
            pegl.ConfigAttrib.DEPTH_SIZE: 24,
            pegl.ConfigAttrib.RENDERABLE_TYPE: pegl.ClientAPIFlag.OPENGL,
            #pegl.ConfigAttrib.CONTEXT_OPENGL_DEBUG: pegl.EGL_TRUE,
            })[0]
        self.ctx = conf.create_context()
        self.surf = conf.create_pbuffer_surface({pegl.SurfaceAttrib.WIDTH: self.width,
                                            pegl.SurfaceAttrib.HEIGHT: self.height})
        self.ctx.make_current(draw=self.surf)
        print("config_id: ",self.ctx.config_id, self.ctx.config)

        # print(glGetInteger(GL_VERSION))
        print("GL version", glGetInteger(GL_MAJOR_VERSION), glGetInteger(GL_MINOR_VERSION))

        # generate gpu memories
        #self.npts = 10000
        self.pts_vao = glGenVertexArrays(1)
        self.pts_vbo = glGenBuffers(1)
        glBindVertexArray(self.pts_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.pts_vbo)
        glBufferData(GL_ARRAY_BUFFER, sizeof(ctypes.c_float)*3*self.npts, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(sizeof(ctypes.c_float)*3))
        glBindVertexArray(0)

        self.pts_fbo = glGenFramebuffers(1)
        self.pts_rbo_color = glGenRenderbuffers(1)
        self.pts_rbo_depth = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.pts_rbo_color)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, self.width, self.height)
        glBindRenderbuffer(GL_RENDERBUFFER, self.pts_rbo_depth)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, self.width, self.height)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, self.pts_fbo)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, self.pts_rbo_color)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.pts_rbo_depth)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        self.pts_pbo_color = glGenBuffers(1)
        self.pts_pbo_depth = glGenBuffers(1)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, self.pts_pbo_color)
        glBufferData(GL_PIXEL_PACK_BUFFER, sizeof(ctypes.c_uint8)*4*self.width*self.height, None, GL_STREAM_READ)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, self.pts_pbo_depth)
        glBufferData(GL_PIXEL_PACK_BUFFER, 24//8*self.width*self.height, None, GL_STREAM_READ)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)

        # prepare shaders
        self.shader = createShader(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),"unlit_shader.vert"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)),"unlit_shader.frag"))

        mat_proj_ul = glGetUniformLocation(self.shader,"mat_proj")
        self.mat_view_ul = glGetUniformLocation(self.shader,"mat_view")
        self.mat_model_ul = glGetUniformLocation(self.shader,"mat_model")
        fovy=math.pi/4
        aspect=self.width/self.height
        zNear=0.1
        zFar=10.0
        mat_proj=glm.perspective(fovy,aspect,zNear,zFar)
        camPos=glm.vec3(1,1,1)
        gazePos=glm.vec3(0,0,0)
        upDir=glm.vec3(0,1,0)
        mat_view = glm.lookAt(camPos, gazePos, upDir)
        mat_model = glm.mat4(1.0)
        glUseProgram(self.shader)
        glUniformMatrix4fv(mat_proj_ul, 1, GL_FALSE, glm.value_ptr(mat_proj))
        glUniformMatrix4fv(self.mat_view_ul, 1, GL_FALSE, glm.value_ptr(mat_view))
        glUniformMatrix4fv(self.mat_model_ul, 1, GL_FALSE, glm.value_ptr(mat_model))
        glUseProgram(0)

        return
    
    def __len__(self):
        return len(self.ply_lists)

    def __getitem__(self, id):
        cat_id, pts = self.ptss[id]
        pts = pts.astype(np.float32)
        # send pts to gpu
        glBindBuffer(GL_ARRAY_BUFFER, self.pts_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(ctypes.c_float)*3*self.npts, pts)
        glBindBuffer(GL_ARRAY_BUFFER, 0)


        # render
        glBindFramebuffer(GL_FRAMEBUFFER, self.pts_fbo)
        glClearColor(0,0,0.5,1)
        glViewport(0,0,self.width,self.height)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.shader)
        glBindBuffer(GL_ARRAY_BUFFER, self.pts_vao)
        glDrawArrays(GL_POINTS, 0, self.npts)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glUseProgram(0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        try:
            glBindFramebuffer(GL_READ_FRAMEBUFFER, self.pts_fbo)
            glBindBuffer(GL_PIXEL_PACK_BUFFER, self.pts_pbo_color)
            glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
            # glBindBuffer(GL_PIXEL_PACK_BUFFER, self.pts_pbo_depth)
            # glReadPixels(0, 0, self.width, self.height, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT)
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
            glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)
        except OpenGL.error.GLError as e:
            print(e.err, e.baseOperation,e.description)
            print(e)
            raise e


        glFlush()
        glFinish()

        glBindBuffer(GL_PIXEL_PACK_BUFFER, self.pts_pbo_color)
        p = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY)
        color_img = np.ctypeslib.as_array(ctypes.cast(p, ctypes.POINTER(ctypes.c_uint8)), shape=(self.height,self.width,4))
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER)
        # glBindBuffer(GL_PIXEL_PACK_BUFFER, self.pts_pbo_depth)
        # p = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY)
        # zbuf_img = np.ctypeslib.as_array(ctypes.cast(p, ctypes.POINTER(ctypes.c_uint16)), shape=(self.height,self.width)).astype(np.float)/65535.0
        # glUnmapBuffer(GL_PIXEL_PACK_BUFFER)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)

        color_img = torch.Tensor(color_img[:, :, 0:3]).permute(2,0,1)
        if self.transform is not None:
            color_img = self.transform(color_img)

        return cat_id, color_img



def worker_init_fn(worker_id:int) -> None:
    info = torch.utils.data.get_worker_info()
    _seed = info.dataset.base_seed+worker_id
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)
    info.dataset.open(worker_id)
    return

if __name__=="__main__":
    base_seed = 7743

    use_gpus=[0,1,2,3]
    dataset = DynamicRenderer(use_gpus=use_gpus, data_dir=data_dir, base_seed = base_seed, transform = None)
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=len(use_gpus), batch_size=256, shuffle=True,
                                             drop_last=True, worker_init_fn=worker_init_fn, persistent_workers=True)


    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    # test loading
    NUM_STEPS = 100
    LOG_STEP=1
    bt = time.time()
    _imgs = None
    for i, (labels, imgs) in enumerate(dataloader):
        _imgs=imgs.clone()
        if i%LOG_STEP==0:
            print("Iter %d done."%i)
        if i >= NUM_STEPS-1:
            break
    ct = time.time()
    print("elapsed time: %.3f s (avg. %.3f s)"%(ct-bt,(ct-bt)/NUM_STEPS))

    pimg=Image.fromarray(np.array(torchvision.utils.make_grid(_imgs,nrow=16).permute(1,2,0)))
    pimg.resize((1024,1024))
    pimg.save("test.png", Image.BILINEAR)
