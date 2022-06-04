# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fractaldb_path', default='./PC-FractalDB/3Dfractalmodel', help='load PLY path')
parser.add_argument('--save_dir', default='./PC-FractalDB/3Dfractalscene', help='save JSON path')
parser.add_argument('--numof_scene', type=int, default=10000, help='the number of 3D fractal scene')
parser.add_argument('--numof_classes', type=int, default=1000, help='the number of fractal category')
parser.add_argument('--numof_instance', type=int, default=1000, help='the number of intra-category')
parser.add_argument('--numof_object', type=int, default=15, help='the average object per 3D fractal scene')
parser.add_argument('--scene_size', type=float, default=15.0, help='the size of 3D fractal scene')
FLAGS = parser.parse_args()

src_dir = FLAGS.fractaldb_path
NUM_CLASS = FLAGS.numof_classes
NUM_INS = FLAGS.numof_instance
NUM_OBJ = FLAGS.numof_object
ROOM_SIZE = FLAGS.scene_size
dump_dir = FLAGS.save_dir
num_scenes = FLAGS.numof_scene

def setObjNum(NUM_OBJ):
    return NUM_OBJ + int(np.floor(np.random.poisson(lam=5)))

def setObjSize():
    base_size = 0.75 + np.random.random()*0.5   # uniform random in [1.00, 5.00]
    aspects = np.ones(3, dtype=np.float32)
    aspects[1:3] = 0.9 + np.random.random(2) * 0.2  # uniform random in [0.9, 1.1]
    return aspects * base_size

def _checkInterpositionBox(box1, box2):
    def checkInterpos(p1, p2):
        return p1[0] > p2[1] and p2[0] > p1[1]

    def getEdge(box, axis):
        c = box["c_pos"][axis]
        w = box["size"][axis] * 0.5
        return [ c+w, c-w ]

    for axis in [0,1]:
        if not checkInterpos(getEdge(box1, axis), getEdge(box2, axis)):
            return False
    return True

def _checkInterpositionBox2(box1, box2):
    c1 = np.array(box1["c_pos"][:2])
    c2 = np.array(box2["c_pos"][:2])
    s1 = np.array(box1["size"][:2])
    s2 = np.array(box2["size"][:2])
    return np.linalg.norm(c1-c2) < (np.linalg.norm(s1) + np.linalg.norm(s2)) * 0.5 + 0.25


def setOneObject(objects):
    s_box = setObjSize()
    c_z = s_box[2] * 0.5

    ITERATION_MAX = 100
    for itr in range(ITERATION_MAX):
        c_pos = np.zeros(3, dtype=np.float32)
        c_pos[:2] = (1.0 - np.random.random(2)*2.0)  * ROOM_SIZE*0.5     # uniform random in [-ROOM_SIZE/2, ROOM_SIZE/2]
        c_pos[2] = c_z
        box = { "size":s_box, "c_pos":c_pos }

        accept_flg = True
        for obj in objects:
            if _checkInterpositionBox2(box, obj):
                accept_flg = False
                break
        if accept_flg:
            return box

    # return None

def createOneScene(NUM_OBJ):
    num_obj = setObjNum(NUM_OBJ)
    objects = []
    for obj_id in range(num_obj):
        box = setOneObject(objects)
        if box is None:
            break
        
        obj = {}
        obj["size"] = box["size"].tolist()
        obj["c_pos"] = box["c_pos"].tolist()
        obj["theta"] = (1.0 - np.random.random()*2.0) * np.pi
        file_path = src_files[np.random.randint(0, len(src_files))]
        obj["class_name"] = os.path.basename(os.path.dirname(file_path))
        obj["file_name"] = os.path.basename(file_path)
        objects.append(obj)
    return objects

if __name__=='__main__':
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    src_files = []
    for class_idx in range(NUM_CLASS):
        c_files = glob.glob(os.path.join(src_dir, "%06d"%class_idx, "*.ply"))
        c_files.sort()
        src_files.extend(c_files[0:NUM_INS])

    print(len(src_files))

    if not os.path.exists(dump_dir):
        os.mkdir(dump_dir)
        print ("create", dump_dir)


    for idx in range (num_scenes):
        print ("\r%d/%d"%(idx+1, num_scenes), end='')
        file_name = os.path.join(dump_dir, "scene_%05d.json"%idx)
        if os.path.exists(file_name):
            continue

        objects = createOneScene(NUM_OBJ)
        with open(file_name, 'w') as f:
            json.dump(objects, f)
    print('')
