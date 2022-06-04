# -*- coding: utf-8 -*-
import os
import time
import argparse
import numpy as np
from IteratedFunctionSystem import ifs_function
import open3d
import random

def conf():
	parser = argparse.ArgumentParser()
	parser.add_argument("--load_root", default="./PC-FractalDB/3DIFS_param", type = str, help="load csv root")
	parser.add_argument("--save_root", default="./PC-FractalDB/3Dfractalmodel", type = str, help="save PLY root")
	parser.add_argument("--iteration", default=10000, type = int)
	parser.add_argument("--numof_classes", default=1000, type = int)
	parser.add_argument("--start_class", default=0, type = int)
	parser.add_argument("--numof_instance", default=1000, type = int)
	parser.add_argument("--normalize", default=1.0, type=float)
	parser.add_argument("--ratio", default=0.8, type=float)
	parser.add_argument('--visualize', action='store_true')
	args = parser.parse_args()
	return args

def min_max(args, x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = ((x-min)/(max-min)) * (args.normalize - (-args.normalize)) - args.normalize
    return result

def centoroid(point):
	new_centor = []
	sum_x = (sum(point[0]) / args.iteration)
	sum_y = (sum(point[1]) / args.iteration)
	sum_z = (sum(point[2]) / args.iteration)
	centor_of_gravity = [sum_x, sum_y, sum_z]
	fractal_point_x = (point[0] - centor_of_gravity[0]).tolist()
	fractal_point_y = (point[1] - centor_of_gravity[1]).tolist()
	fractal_point_z = (point[2] - centor_of_gravity[2]).tolist()
	new_centor.append(fractal_point_x)
	new_centor.append(fractal_point_y)
	new_centor.append(fractal_point_z)
	new = np.array(new_centor)
	return new

if __name__ == "__main__":
	starttime = time.time()
	args = conf()

	# object_list = ['object', 'main', 'mix']
	# for i in object_list:
	os.makedirs(args.save_root, exist_ok=True)

	csv_names = os.listdir(args.load_root)
	csv_names.sort()
	mix_csv_names = os.listdir(args.load_root)
	mix_csv_names = random.sample(mix_csv_names, k=args.numof_instance)

	for i, csv_name in enumerate(csv_names):
		name, ext = os.path.splitext(str(csv_name))
		name = args.start_class + int(name)
		name = '%06d' % name

		if i > args.numof_classes:
			break

		if ext != ".csv":
			continue

		os.makedirs(os.path.join(args.save_root, name), exist_ok=True)

		params = np.genfromtxt(args.load_root + "/" + csv_name, dtype=np.str, delimiter=",")
		main_generators = ifs_function()
		main_obj_num = args.iteration * args.ratio
			
		for param in params:
			main_generators.set_param(float(param[0]), float(param[1]),float(param[2]), float(param[3]),
				float(param[4]), float(param[5]),float(param[6]), float(param[7]),
				float(param[8]), float(param[9]),float(param[10]), float(param[11]), float(param[12]))

		main_fractal_point = main_generators.calculate(int(main_obj_num))
		main_fractal_point = min_max(args, main_fractal_point, axis=None)
		main_fractal_point = centoroid(main_fractal_point)
		main_point_data = main_fractal_point.transpose()
		main_pointcloud = open3d.geometry.PointCloud()
		main_pointcloud.points = open3d.utility.Vector3dVector(main_point_data)

		fractal_weight = 0
		for j, mix_csv in enumerate(mix_csv_names):
			padded_fractal_weight= '%04d' % fractal_weight
			mix_generators = ifs_function()
			if j == args.numof_instance:
				break
			mix_params = np.genfromtxt(args.load_root + "/" + mix_csv, dtype=np.str, delimiter=",")
			for mix_param in mix_params:
					mix_generators.set_param(float(mix_param[0]), float(mix_param[1]),float(mix_param[2]), float(mix_param[3]),
					float(mix_param[4]), float(mix_param[5]),float(mix_param[6]), float(mix_param[7]),
					float(mix_param[8]), float(mix_param[9]),float(mix_param[10]), float(mix_param[11]), float(mix_param[12]))
			mix_obj_num = args.iteration * (1 - args.ratio)
			mix_fractal_point = mix_generators.calculate(int(mix_obj_num) + 1)
			mix_fractal_point = min_max(args, mix_fractal_point, axis=None)
			mix_fractal_point = centoroid(mix_fractal_point)
			mix_point_data = mix_fractal_point.transpose()
			mix_pointcloud = open3d.geometry.PointCloud()
			mix_pointcloud.points = open3d.utility.Vector3dVector(mix_point_data)
			fractal_point = np.concatenate((main_fractal_point, mix_fractal_point), axis = 1) 

			# min-max normalize
			# fractal_point = min_max(args, fractal_point, axis=None)
			# move to center point 
			fractal_point = centoroid(fractal_point)
			# search N/A value
			arr = np.isnan(fractal_point).any(axis=1)

			if arr[1] == False:
				point_data = fractal_point.transpose()
				pointcloud = open3d.geometry.PointCloud()
				pointcloud.points = open3d.utility.Vector3dVector(point_data)
				if args.visualize == True:
					fractal_point_pcd = open3d.visualization.draw_geometries([pointcloud])
				open3d.io.write_point_cloud((args.save_root + "/" + name + "/" + name + "_" + padded_fractal_weight + ".ply"), pointcloud)
			fractal_weight += 1
	
	endtime = time.time()
	interval = endtime - starttime
	print("passed time = %dh %dm %ds" % (int(interval/3600),int((interval%3600)/60),int((interval%3600)%60)))
