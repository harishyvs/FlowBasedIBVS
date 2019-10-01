# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import random
import sys
import time

import numpy as np
from enum import Enum
from PIL import Image

import habitat_sim
import habitat_sim.agent

from habitat_sim.utils import d3_40_colors_rgb
import habitat_sim.bindings as hsim
from settings_me import default_sim_settings


class DemoRunnerType(Enum):
    BENCHMARK = 1
    EXAMPLE = 2


class DemoRunner:
    def __init__(self, sim_settings, simulator_demo_type):
        if simulator_demo_type == DemoRunnerType.EXAMPLE:
            self.set_sim_settings(sim_settings)

    def set_sim_settings(self, sim_settings):
        self._sim_settings = sim_settings.copy()

    # build SimulatorConfiguration
    def make_cfg(self, settings,Vx,Vy,Vz,Wx,Wy,Wz):
        sim_cfg = hsim.SimulatorConfiguration()
        sim_cfg.gpu_device_id = 0
        sim_cfg.scene.id = settings["scene"]

        # define default sensor parameters (see src/esp/Sensor/Sensor.h)
        sensors = {
            "color_sensor": {  # active if sim_settings["color_sensor"]
                "sensor_type": hsim.SensorType.COLOR,
                "resolution": [settings["height"], settings["width"]],
                "position": [0.0, settings["sensor_height"], 0.0],
            },
            "depth_sensor": {  # active if sim_settings["depth_sensor"]
                "sensor_type": hsim.SensorType.DEPTH,
                "resolution": [settings["height"], settings["width"]],
                "position": [0.0, settings["sensor_height"], 0.0],
            },
            "semantic_sensor": {  # active if sim_settings["semantic_sensor"]
                "sensor_type": hsim.SensorType.SEMANTIC,
                "resolution": [settings["height"], settings["width"]],
                "position": [0.0, settings["sensor_height"], 0.0],
            },
        }

        # create sensor specifications
        sensor_specs = []
        for sensor_uuid, sensor_params in sensors.items():
            if settings[sensor_uuid]:
                sensor_spec = hsim.SensorSpec()
                sensor_spec.uuid = sensor_uuid
                sensor_spec.sensor_type = sensor_params["sensor_type"]
                sensor_spec.resolution = sensor_params["resolution"]
                sensor_spec.position = sensor_params["position"]
                if not self._sim_settings["silent"]:
                    print("==== Initialized Sensor Spec: =====")
                    print("Sensor uuid: ", sensor_spec.uuid)
                    print("Sensor type: ", sensor_spec.sensor_type)
                    print("Sensor position: ", sensor_spec.position)
                    print("===================================")

                sensor_specs.append(sensor_spec)

        # create agent specifications
        # agent_cfg = hsim.AgentConfiguration()
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.action_space = {
            # 14.1447326288
            # "move_right": hsim.ActionSpec("moveRight", {"amount": 14.1447326288*Vx}),
            # "move_left": hsim.ActionSpec("moveLeft", {"amount": 0}),
            # "move_up": hsim.ActionSpec("moveUp", {"amount": 19.9351346819*Vy}),
            # "move_down": hsim.ActionSpec("moveDown", {"amount": 0}),
            # "move_forward": hsim.ActionSpec("moveForward", {"amount": 0}),
            # "move_backward": hsim.ActionSpec("moveBackward", {"amount": 3.77832260507*Vz}),

            # new_process
            # "move_right": hsim.ActionSpec("moveRight", {"amount": -0.09883337089*Vx}),
            "move_right": habitat_sim.agent.ActionSpec("moveRight", {"amount": -0.1*Vx}),
            # "move_right": hsim.ActionSpec("moveRight", {"amount": -0.08*Vx}),

            "move_left": habitat_sim.agent.ActionSpec("moveLeft", {"amount": 0}),
            "move_up":habitat_sim.agent.ActionSpec("moveUp", {"amount": -0.09883337089*Vy}),
            # "move_up": hsim.ActionSpec("moveUp", {"amount": -0.1*Vy}),

            "move_down": habitat_sim.agent.ActionSpec("moveDown", {"amount": 0}),
            "move_forward": habitat_sim.agent.ActionSpec("moveForward", {"amount": 0}),
            # "move_backward": hsim.ActionSpec("moveBackward", {"amount": -0.09883337089*Vz}),
            # "move_backward": hsim.ActionSpec("moveBackward", {"amount": 0.01*Vz}),
            # "move_backward": hsim.ActionSpec("moveBackward", {"amount": 0.09*Vz}),
            # "move_backward": hsim.ActionSpec("moveBackward", {"amount": -0.09*Vz}),
            "move_backward": habitat_sim.agent.ActionSpec("moveBackward", {"amount": 0.01*Vz}),
            # "look_left": hsim.ActionSpec("lookLeft", {"amount": 0*Wy}),
            # "look_right": hsim.ActionSpec("lookRight", {"amount": 0}),
            # "look_up": hsim.ActionSpec("lookUp", {"amount": 0*Wx}),
            # "look_down": hsim.ActionSpec("lookDown", {"amount": 0}),
            # "look_anti": hsim.ActionSpec("lookAnti", {"amount": 0*Wz}),
            # "look_clock": hsim.ActionSpec("lookClock", {"amount": 0}),


            # "look_left": hsim.ActionSpec("lookLeft", {"amount": 26.8092404858*Wy}),
            # "look_right": hsim.ActionSpec("lookRight", {"amount": 0}),
            # "look_up": hsim.ActionSpec("lookUp", {"amount": -23.0150891661*Wx}),
            # "look_down": hsim.ActionSpec("lookDown", {"amount": 0}),
            # # "look_anti": hsim.ActionSpec("lookAnti", {"amount": 491.810451008*Wz}),
            # "look_anti": hsim.ActionSpec("lookAnti", {"amount": 24*Wz}),
            # "look_clock": hsim.ActionSpec("lookClock", {"amount": 0}),

            "look_left": habitat_sim.agent.ActionSpec("lookLeft", {"amount": 1*Wy}),
            "look_right": habitat_sim.agent.ActionSpec("lookRight", {"amount": 0}),
            "look_up": habitat_sim.agent.ActionSpec("lookUp", {"amount": -1*Wx}),
            "look_down": habitat_sim.agent.ActionSpec("lookDown", {"amount": 0}),
            # "look_anti": hsim.ActionSpec("lookAnti", {"amount": 491.810451008*Wz}),

            # "look_anti": hsim.ActionSpec("lookAnti", {"amount": 2*Wz}),
            "look_anti": habitat_sim.agent.ActionSpec("lookAnti", {"amount": -1*Wz}),
            # "look_anti": hsim.ActionSpec("lookAnti", {"amount": 0*Wz}),
            # "look_anti": hsim.ActionSpec("lookAnti", {"amount": 0*Wz}),
            "look_clock": habitat_sim.agent.ActionSpec("lookClock", {"amount": 0}),

        }
        # sim_cfg.agents = [agent_cfg]

        # return sim_cfg
        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def save_color_observation(self, obs, total_frames):
        color_obs = obs["color_sensor"]
        color_img = Image.fromarray(color_obs, mode="RGBA")
        color_img.save("/home/yvsharish/working/habitat-sim/image_baseline_2_output/test.rgba.%05d.png" % total_frames)

    def save_semantic_observation(self, obs, total_frames):
        semantic_obs = obs["semantic_sensor"]
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img.save("test.sem.%05d.png" % total_frames)

    def save_depth_observation(self, obs, total_frames):
        if self._sim_settings["depth_sensor"]:
            depth_obs = obs["depth_sensor"]
            depth_img = Image.fromarray(
                (depth_obs / 10 * 255).astype(np.uint8), mode="L"
            )
            depth_img.save("/home/yvsharish/working/habitat-sim/image_baseline_2_output/test.depth.%05d.png" % total_frames)

    def output_semantic_mask_stats(self, obs, total_frames):
        semantic_obs = obs["semantic_sensor"]
        counts = np.bincount(semantic_obs.flatten())
        total_count = np.sum(counts)
        print(f"Pixel statistics for frame {total_frames}")
        for object_i, count in enumerate(counts):
            sem_obj = self._sim.semantic_scene.objects[object_i]
            cat = sem_obj.category.name()
            pixel_ratio = count / total_count
            if pixel_ratio > 0.01:
                print(f"obj_id:{sem_obj.id},category:{cat},pixel_ratio:{pixel_ratio}")

    def init_agent_state(self, agent_id):
        # initialize the agent at a random start state
        agent = self._sim.initialize_agent(agent_id)
        # start_state = hsim.AgentState()
        start_state=agent.get_state()

        # force starting position on first floor (try 100 samples)
        num_start_tries = 0
        while start_state.position[1] > 0.5 and num_start_tries < 100:
            start_state.position = self._sim.pathfinder.get_random_navigable_point()
            num_start_tries += 1
        agent.set_state(start_state)

        if not self._sim_settings["silent"]:
            print(
                "start_state.position\t",
                start_state.position,
                "start_state.rotation\t",
                start_state.rotation,
            )

        return start_state

    def compute_shortest_path(self, start_pos, end_pos):
        self._shortest_path.requested_start = start_pos
        self._shortest_path.requested_end = end_pos
        self._sim.pathfinder.find_path(self._shortest_path)
        print("shortest_path.geodesic_distance", self._shortest_path.geodesic_distance)

    def do_time_steps(self,harish):
        total_frames = harish
        start_time = time.time()
        action_names = list(
            self._sim_cfg.agents[
                self._sim_settings["default_agent"]
            ].action_space.keys()
        )

        # while total_frames < self._sim_settings["max_frames"]:
        # action = random.choice(action_names)
        # p=self._sim.get_agent(0)

        action = "move_right"
        if not self._sim_settings["silent"]:
            print("action", action)
        observations = self._sim.step(action)
        state = self._sim.last_state()
        # p.get_state(state)
        # p.set_state(state)

        action="move_backward"
        if not self._sim_settings["silent"]:
            print("action", action)
        observations = self._sim.step(action)
        state = self._sim.last_state()
        # p.get_state(state)
        # p.set_state(state)

        action="move_up"
        if not self._sim_settings["silent"]:
            print("action", action)
        observations = self._sim.step(action)
        state = self._sim.last_state()
        # p.get_state(state)
        # p.set_state(state)

        action="look_up"
        if not self._sim_settings["silent"]:
            print("action", action)
        observations = self._sim.step(action)
        state = self._sim.last_state()
        # p.get_state(state)
        # p.set_state(state)

        action="look_left"
        if not self._sim_settings["silent"]:
            print("action", action)
        observations = self._sim.step(action)
        state = self._sim.last_state()
        # p.get_state(state)
        # p.set_state(state)

        action="look_anti"
        if not self._sim_settings["silent"]:
            print("action", action)
        observations = self._sim.step(action)
        # p.get_state(state)
        # p.set_state(state)
        # action="move_"
        if self._sim_settings["save_png"]:
            if self._sim_settings["color_sensor"]:
                self.save_color_observation(observations, total_frames)
            if self._sim_settings["depth_sensor"]:
                self.save_depth_observation(observations, total_frames)
            if self._sim_settings["semantic_sensor"]:
                self.save_semantic_observation(observations, total_frames)
        state = self._sim.last_state()

        if not self._sim_settings["silent"]:
            print("position\t", state.position, "\t", "rotation\t", state.rotation)
        f=open("/home/yvsharish/working/aaaaa/baseline_2_exp_pose.txt","a+")
        f.write("%0.8f %0.8f %0.8f %e %e %e %e\n" %(state.position[0],state.position[1],state.position[2],state.rotation[0],state.rotation[1],state.rotation[2],state.rotation[3]))
        f.close()
        # f.write("%0.8f %0.8f %0.8f %e %e %e %e\n" %(state.position[0],state.position[1],state.position[2],state.rotation[0],state.rotation[1],state.rotation[2],state.rotation[3]))


        if self._sim_settings["compute_shortest_path"]:
            self.compute_shortest_path(
                state.position, self._sim_settings["goal_position"]
            )

        if self._sim_settings["compute_action_shortest_path"]:
            self._action_shortest_path.requested_start.position = state.position
            self._action_shortest_path.requested_start.rotation = state.rotation
            self._action_pathfinder.find_path(self._action_shortest_path)
            print(
                "len(action_shortest_path.actions)",
                len(self._action_shortest_path.actions),
            )

        if (
            self._sim_settings["semantic_sensor"]
            and self._sim_settings["print_semantic_mask_stats"]
        ):
            self.output_semantic_mask_stats(observations, total_frames)

        # total_frames += 1

        end_time = time.time()
        perf = {}
        perf["total_time"] = end_time - start_time
        perf["fps"] = total_frames / perf["total_time"]

        return perf

    def print_semantic_scene(self):
        if self._sim_settings["print_semantic_scene"]:
            scene = self._sim.semantic_scene
            print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")
            for level in scene.levels:
                print(
                    f"Level id:{level.id}, center:{level.aabb.center},"
                    f" dims:{level.aabb.sizes}"
                )
                for region in level.regions:
                    print(
                        f"Region id:{region.id}, category:{region.category.name()},"
                        f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
                    )
                    for obj in region.objects:
                        print(
                            f"Object id:{obj.id}, category:{obj.category.name()},"
                            f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                        )
            input("Press Enter to continue...")

    def init_common(self,Vx,Vy,Vz,Wx,Wy,Wz,harish):
        # Vx,Vy,Vz=[0.00360888861229238410, 0.01565492554614493145, -0.01483013368933373821]
        self._sim_cfg = self.make_cfg(self._sim_settings,Vx,Vy,Vz,Wx,Wy,Wz)
        self._sim = habitat_sim.Simulator(self._sim_cfg)

        random.seed(self._sim_settings["seed"])
        self._sim.seed(self._sim_settings["seed"])

        # initialize the agent at a random start state
        start_state = self.init_agent_state(self._sim_settings["default_agent"])
    # print
    # start_state = [-0.589161,0.76511734, -1.6109103]
        return start_state

    def benchmark(self, settings,Vx,Vy,Vz,Wx,Wy,Wz):
        self.set_sim_settings(settings)
        self.init_common(Vx,Vy,Vz,Wx,Wy,Wz)

        perf = self.do_time_steps(harish)

        self._sim.close()
        del self._sim

        return perf

    def example(self,Vx,Vy,Vz,Wx,Wy,Wz,harish):

        start_state = self.init_common(Vx,Vy,Vz,Wx,Wy,Wz,harish)
        # print(start_state.position[0])
        # print("harish")
        # start_state.position[0]=[-1.661037   0.18145986  -1.1990095]
        # start_state.rotation=[0.00000000e+00 1.22929005e-05 0.00000000e+00 1.00000000e+00]

        #
        # # initialize and compute shortest path to goal
        # self._shortest_path = hsim.ShortestPath()
        # self.compute_shortest_path(
        #     start_state.position, self._sim_settings["goal_position"]
        # )
        #
        # # set the goal headings, and compute action shortest path
        # if self._sim_settings["compute_action_shortest_path"]:
        #     agent_id = self._sim_settings["default_agent"]
        #     goal_headings = self._sim_settings["goal_headings"]
        #     self._action_pathfinder = self._sim.make_action_pathfinder(agent_id)
        #
        #     self._action_shortest_path = hsim.MultiGoalActionSpaceShortestPath()
        #     self._action_shortest_path.requested_start.position = start_state.position
        #     self._action_shortest_path.requested_start.rotation = start_state.rotation
        #
        #     # explicitly reset the start position
        #     self._shortest_path.requested_start = start_state.position
        #
        #     # initialize the requested ends when computing the action shortest path
        #     next_goal_idx = 0
        #     while next_goal_idx < len(goal_headings):
        #         sampled_pos = self._sim.pathfinder.get_random_navigable_point()
        #         self._shortest_path.requested_end = sampled_pos
        #         if (
        #             self._sim.pathfinder.find_path(self._shortest_path)
        #             and self._shortest_path.geodesic_distance < 5.0
        #             and self._shortest_path.geodesic_distance > 2.5
        #         ):
        #             self._action_shortest_path.requested_ends.append(
        #                 hsim.ActionSpacePathLocation(
        #                     sampled_pos, goal_headings[next_goal_idx]
        #                 )
        #             )
        #             next_goal_idx += 1
        #
        #     self._shortest_path.requested_end = self._sim_settings["goal_position"]
        #     self._sim.pathfinder.find_path(self._shortest_path)
        #
        #     self._action_pathfinder.find_path(self._action_shortest_path)
        #     print(
        #         "len(action_shortest_path.actions)",
        #         len(self._action_shortest_path.actions),
        #     )
        #
        # # print semantic scene
        # self.print_semantic_scene()
        # f=open("/home/harish/RRC/ICRA_2019/baseline_2/aaaaa/baseline_2_exp_pose.txt","a+")
        perf = self.do_time_steps(harish)
        self._sim.close()
        del self._sim
        # f.close()
        return perf
