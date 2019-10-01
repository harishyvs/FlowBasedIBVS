# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


default_sim_settings = {
    # settings shared by example.py and benchmark.py
    "max_frames": 19,
    "total_frames":0,
    "width": 640,
    "height": 480,
    "scene": "Kerrtown.glb",  # default scene: test.glb
    "default_agent": 0,
    "sensor_height": 1.5,
    "color_sensor": True,  # RGB sensor (default: ON)
    "semantic_sensor": False,  # semantic sensor (default: OFF)
    "depth_sensor": False,  # depth sensor (default: OFF)
    "seed": 1,
    "silent": False,  # do not print log info (default: OFF)
    # settings exclusive to example.py
    "save_png": True,  # save the pngs to disk (default: OFF)
    "print_semantic_scene": False,
    "print_semantic_mask_stats": False,
    "compute_shortest_path": True,
    "compute_action_shortest_path": False,
    # "goal_position": [-1.7020688,0.511734, -0.6207685],
    # "goal_headings": [[0.0, -0.471395, 0.0, 0.881922], [0.0, 1.0, 0.0, 0.0]],
}
