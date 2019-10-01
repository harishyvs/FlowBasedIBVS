#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import demo_runner_me_harit_request_2 as dr
#import demo_runner_me_other as dr2
import numpy as np
import os
import subprocess

def shell_source(script):
    """Sometime you want to emulate the action of "source" in bash,
    settings some environment variables. Here is a way to do it."""
    import subprocess, os
    pipe = subprocess.Popen(". %s; env" % script, stdout=subprocess.PIPE, shell=True)
    output = pipe.communicate()[0]
    env = dict((line.split("=", 1) for line in output.splitlines()))
    os.environ.update(env)

parser = argparse.ArgumentParser()
parser.add_argument("--width", type=int, default=620)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--scene", type=str, default="rtown.glb")
parser.add_argument("--max_frames", type=int, default=400)
parser.add_argument("--total_frames",type=int,default=0)
parser.add_argument("--save_png", action="store_true")
parser.add_argument("--sensor_height", type=float, default=1.5)
parser.add_argument("--disable_color_sensor", action="store_true")
parser.add_argument("--semantic_sensor", action="store_true")
parser.add_argument("--depth_sensor", action="store_true")
parser.add_argument("--print_semantic_scene", action="store_true")
parser.add_argument("--print_semantic_mask_stats", action="store_true")
parser.add_argument("--compute_shortest_path", action="store_true")
parser.add_argument("--compute_action_shortest_path", action="store_true")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--silent", action="store_true")
# parser.add_argument("--goal_position", type=list,default=[-9.423, 0.07, -0.373])
# parser.add_argument("--goal_headings", type=float,default=[[0.0, -0.47, 0.0, 0.88], [0.0, 1.0, 0.0, 0.0]])

args = parser.parse_args()


def make_settings():
    settings = dr.default_sim_settings.copy()
    settings["max_frames"] = args.max_frames
    settings["total_frames"]=args.total_frames
    settings["width"] = args.width
    settings["height"] = args.height
    settings["scene"] = args.scene
    settings["save_png"] = args.save_png
    settings["sensor_height"] = args.sensor_height
    settings["color_sensor"] = not args.disable_color_sensor
    settings["semantic_sensor"] = args.semantic_sensor
    settings["depth_sensor"] = args.depth_sensor
    settings["print_semantic_scene"] = args.print_semantic_scene
    settings["print_semantic_mask_stats"] = args.print_semantic_mask_stats
    settings["compute_shortest_path"] = args.compute_shortest_path
    settings["compute_action_shortest_path"] = args.compute_action_shortest_path
    settings["seed"] = args.seed
    settings["silent"] = args.silent
    # settings["goal_position"] = args.goal_position
    # settings["goal_headings"] = args.goal_headings

    return settings


settings = make_settings()

demo_runner = dr.DemoRunner(settings, dr.DemoRunnerType.EXAMPLE)
# demo_runner_2 = dr2.DemoRunner(settings, dr2.DemoRunnerType.EXAMPLE)

# with open("/home/harish/RRC/ICRA_2019/visual_servo/src/data.txt") as f:
# print("harish")
# os.chdir("/home/harish/RRC/ICRA_2019/Flow_Net/flownet2/")
# os.system("source set-env.sh")
# print("harish")
# os.chdir("/home/harish/RRC/ICRA_2019/Flow_Net/flownet2/scripts/")
# os.system("python2 run-flownet.py /home/harish/RRC/ICRA_2019/Flow_Net/flownet2/models/FlowNet2/FlowNet2_weights.caffemodel.h5  /home/harish/RRC/ICRA_2019/Flow_Net/flownet2/models/FlowNet2/FlowNet2_deploy.prototxt.template /home/harish/RRC/ICRA_2019/habitat-sim/image_baseline_2/test.rgba.00000.png /home/harish/RRC/ICRA_2019/habitat-sim/image_baseline_2/test.rgba.00019.png out_single_1.flo")
#
# os.chdir("/home/harish/RRC/ICRA_2019/baseline_2/")
# #cd /home/harish/RRC/ICRA_2019/baseline_2/
# os.system("python2 ibvs_controller_single.py 0")
# # python2 ibvs_controller_single.py 0

frames=0

# os.chdir("/home/harish/RRC/ICRA_2019/baseline_2/")
# ##################### change here when you change for iterations
# command="python2 photo_error.py"+frames.get() + "2"
# print(command)
# os.system(command)
#
# photo_error=0
# file_in=f.open('/home/harish/RRC/ICRA_2019/baseline_2/aaaaa/baseline_2_photo.txt')
# for line in file_in.readlines():
#   photo_error=(float(line))


harish=0
# i=0
# print(mat)
photo_error=250
Vx=float(0)
Vy=float(0)
Vz=float(0)
Wx=float(0)
Wy=float(0)
Wz=float(0)
# while photo_error>40:
# # i=0
# # for i in range(harish):
#     print(harish)
#     with open("/home/harish/RRC/ICRA_2019/baseline_2/aaaaa/baseline_2_velocities_single_1_new.txt") as f:
#         content = f.readlines()
#     # print(content)
#     content = [x.strip() for x in content]
#     mat = []
#     for line in content:
#         s = line.split(' ')
#         print(s)
#         if  len(s) == 6:
#             mat.append(s)
#     Vx=float(mat[harish][0])
#     Vy=float(mat[harish][1])
#     Vz=float(mat[harish][2])
#     Wx=float(mat[harish][3])
#     Wy=float(mat[harish][4])
#     Wz=float(mat[harish][5])
#     frames=frames+1
#     if(frames<2):
#         perf=demo_runner.example(Vx,Vy,Vz,Wx,Wy,Wz,frames)
#     else:
#         perf=demo_runner_2.example(Vx,Vy,Vz,Wx,Wy,Wz,frames)
#
#     # frames=frames+1
#
#     ######################################## change here for the change in the destination and the desired output
#     # foo = subprocess.Popen(["/bin/sh", "/home/harish/RRC/ICRA_2019/Flow_Net/flownet2/set-env.sh"])
#     # shell_source("/home/harish/RRC/ICRA_2019/Flow_Net/flownet2/set-env.sh")
#     os.chdir("/home/harish/RRC/ICRA_2019/Flow_Net/flownet2/scripts")
#
#     command= "python2 run-flownet_baseline_2.py /home/harish/RRC/ICRA_2019/Flow_Net/flownet2/models/FlowNet2/FlowNet2_weights.caffemodel.h5 /home/harish/RRC/ICRA_2019/Flow_Net/flownet2/models/FlowNet2/FlowNet2_deploy.prototxt.template "+ str(frames)+" /home/harish/RRC/ICRA_2019/habitat-sim/image_baseline_2/test.rgba.00019.png out_single_1.flo"
#     os.system(command)
#
#     os.chdir("/home/harish/RRC/ICRA_2019/baseline_2/")
#     command="python2 ibvs_controller_single.py "+str(frames)
#     os.system(command)
#
#     os.chdir("/home/harish/RRC/ICRA_2019/baseline_2/")
#     ##################### change here when you change for iterations
#     command="python2 photo_error.py "+str(frames) + " 20"
#     os.system(command)
#
#     file_in=open("/home/harish/RRC/ICRA_2019/baseline_2/aaaaa/baseline_2_photo.txt","r")
#     for line in file_in.readlines():
#       photo_error=(float(line))
#     print(photo_error)






    # photo_error=50
# print(Vx)
# Vx,Vy,Vz,Wz=[0.00360888861229238410, 0.01565492554614493145, -0.01483013368933373821]
# if(harish==0):
# harish=settings["total_frames"]

# else:
# vard= demo_runner.example(Vx,Vy,Vz,Wz,harish,vard)
# harish=harish+1
demo_runner.example(Vx,Vy,Vz,Wx,Wy,Wz,frames)



print(" ============ Performance ======================== ")
print(
    " %d x %d, total time: %0.3f sec."
    # % (settings["width"], settings["height"], perf["total_time"]),
    # "FPS: %0.1f" % perf["fps"],
)
print(" ================================================= ")
