#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import demo_runner_me as dr
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--width", type=int, default=620)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--scene", type=str, default="test.glb")
parser.add_argument("--max_frames", type=int, default=19)
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
# with open("/home/harish/RRC/ICRA_2019/visual_servo/src/data.txt") as f:
with open("/home/yvsharish/working/aaaaa/baseline_2_velocities_single_1.txt") as f:
    content = f.readlines()
# print(content)
content = [x.strip() for x in content]
mat = []
for line in content:
    s = line.split(' ')
    print(s)
    if  len(s) == 6:
        mat.append(s)
harish=0
# i=0
print(mat)

Vx=float(0)
Vy=float(0)
Vz=float(0)
Wx=float(0)
Wy=float(0)
Wz=float(0)
while harish<settings["total_frames"]:
# i=0
# for i in range(harish):
    print(harish)
    Vx=Vx+float(mat[harish][0])
    Vy=Vy+float(mat[harish][1])
    Vz=Vz+float(mat[harish][2])
    Wx=Wx+float(mat[harish][3])
    Wy=Wy+float(mat[harish][4])
    Wz=Wz+float(mat[harish][5])
    harish=harish+1
# print(Vx)
# Vx,Vy,Vz,Wz=[0.00360888861229238410, 0.01565492554614493145, -0.01483013368933373821]
# if(harish==0):
harish=settings["total_frames"]
perf=demo_runner.example(Vx,Vy,Vz,Wx,Wy,Wz,harish)
# else:
# vard= demo_runner.example(Vx,Vy,Vz,Wz,harish,vard)
# harish=harish+1

print(" ============ Performance ======================== ")
print(
    " %d x %d, total time: %0.3f sec."
    % (settings["width"], settings["height"], perf["total_time"]),
    "FPS: %0.1f" % perf["fps"],
)
print(" ================================================= ")
