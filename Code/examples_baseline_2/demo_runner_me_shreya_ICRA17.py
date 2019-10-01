# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import cv2

from io import BytesIO
from PIL import Image, ImageFile
import os
import random
from interactionMatrix import interactionMatrix
import scipy.io
from numpy import linalg as LA
#
# from scipy.misc import imread, imsave

import sys
import time
import socket

import numpy as np
from enum import Enum
# from PIL import Image

import quaternion
import habitat_sim
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
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.action_space = {
            # 14.1447326288
            # "move_right": habitat_sim.agent.ActionSpec("moveRight", {"amount": 14.1447326288*Vx}),
            # "move_left": habitat_sim.agent.ActionSpec("moveLeft", {"amount": 0}),
            # "move_up": habitat_sim.agent.ActionSpec("moveUp", {"amount": 19.9351346819*Vy}),
            # "move_down": habitat_sim.agent.ActionSpec("moveDown", {"amount": 0}),
            # "move_forward": habitat_sim.agent.ActionSpec("moveForward", {"amount": 0}),
            # "move_backward": habitat_sim.agent.ActionSpec("moveBackward", {"amount": 3.77832260507*Vz}),

            # new_process
            # "move_right": habitat_sim.agent.ActionSpec("moveRight", {"amount": -0.09883337089*Vx}),
            "move_right": habitat_sim.agent.ActionSpec("moveRight", {"amount": -0.1*Vx}),
            # "move_right": habitat_sim.agent.ActionSpec("moveRight", {"amount": -0.08*Vx}),

            "move_left": habitat_sim.agent.ActionSpec("moveLeft", {"amount": 0}),
            "move_up": habitat_sim.agent.ActionSpec("moveUp", {"amount": -0.09883337089*Vy}),
            # "move_up": habitat_sim.agent.ActionSpec("moveUp", {"amount": -0.1*Vy}),

            "move_down": habitat_sim.agent.ActionSpec("moveDown", {"amount": 0}),
            "move_forward": habitat_sim.agent.ActionSpec("moveForward", {"amount": 0}),
            # "move_backward": habitat_sim.agent.ActionSpec("moveBackward", {"amount": -0.09883337089*Vz}),
            # "move_backward": habitat_sim.agent.ActionSpec("moveBackward", {"amount": 0.01*Vz}),
            # "move_backward": habitat_sim.agent.ActionSpec("moveBackward", {"amount": 0.09*Vz}),
            "move_backward": habitat_sim.agent.ActionSpec("moveBackward", {"amount": -0.09*Vz}),
            # "look_left": habitat_sim.agent.ActionSpec("lookLeft", {"amount": 0*Wy}),
            # "look_right": habitat_sim.agent.ActionSpec("lookRight", {"amount": 0}),
            # "look_up": habitat_sim.agent.ActionSpec("lookUp", {"amount": 0*Wx}),
            # "look_down": habitat_sim.agent.ActionSpec("lookDown", {"amount": 0}),
            # "look_anti": habitat_sim.agent.ActionSpec("lookAnti", {"amount": 0*Wz}),
            # "look_clock": habitat_sim.agent.ActionSpec("lookClock", {"amount": 0}),


            # "look_left": habitat_sim.agent.ActionSpec("lookLeft", {"amount": 26.8092404858*Wy}),
            # "look_right": habitat_sim.agent.ActionSpec("lookRight", {"amount": 0}),
            # "look_up": habitat_sim.agent.ActionSpec("lookUp", {"amount": -23.0150891661*Wx}),
            # "look_down": habitat_sim.agent.ActionSpec("lookDown", {"amount": 0}),
            # # "look_anti": habitat_sim.agent.ActionSpec("lookAnti", {"amount": 491.810451008*Wz}),
            # "look_anti": habitat_sim.agent.ActionSpec("lookAnti", {"amount": 24*Wz}),
            # "look_clock": habitat_sim.agent.ActionSpec("lookClock", {"amount": 0}),

            "look_left": habitat_sim.agent.ActionSpec("lookLeft", {"amount": 1*Wy}),
            "look_right": habitat_sim.agent.ActionSpec("lookRight", {"amount": 0}),
            "look_up": habitat_sim.agent.ActionSpec("lookUp", {"amount": -1*Wx}),
            "look_down": habitat_sim.agent.ActionSpec("lookDown", {"amount": 0}),
            # "look_anti": habitat_sim.agent.ActionSpec("lookAnti", {"amount": 491.810451008*Wz}),

            # "look_anti": habitat_sim.agent.ActionSpec("lookAnti", {"amount": 2*Wz}),
            "look_anti": habitat_sim.agent.ActionSpec("lookAnti", {"amount": 1*Wz}),
            # "look_anti": habitat_sim.agent.ActionSpec("lookAnti", {"amount": 0*Wz}),
            "look_clock": habitat_sim.agent.ActionSpec("lookClock", {"amount": 0}),

        }
        # sim_cfg.agents = [agent_cfg]
        #
        # return sim_cfg
        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def save_color_observation(self, obs, total_frames):
        color_obs = obs["color_sensor"]
        color_img = Image.fromarray(color_obs, mode="RGBA")
        color_img.save("/scratch/yvsharish/working/habitat-sim/image_baseline_2_output/test.rgba.%05d.png" % total_frames)

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
            depth_img.save("/scratch/yvsharish/working/habitat-sim/image_baseline_2_output_depth/test.depth.%05d.png" % total_frames)

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
        new_state =habitat_sim.agent.AgentState()
        #new_state.position = np.array([1.23351967 ,0.16511734 ,-0.00159764]).astype('float32')
        new_state.position=np.array([0,0,0]).astype('float32')
        #new_state.position=np.array([0.56720018, 0.20017648, -0.37433529]).astype('float32')
        # new_state.position = np.array([1.23351967 ,2,-0.00159764]).astype('float32')

        new_state.rotation = np.quaternion(1,0,0,0) #deenikosam import quaternion, import numpy as np
        #new_state.rotation=np.quaternion(9.993908e-01, 0.000000e+00, 0.000000e+00, 3.489950e-02)
        agent = self._sim.initialize_agent(agent_id, new_state)
        #agent.set_state(new_state)
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
        action = "move_right"
        if not self._sim_settings["silent"]:
            print("action", action)
        observations = self._sim.step(action)
        state = self._sim.last_state()

        action="move_backward"
        if not self._sim_settings["silent"]:
            print("action", action)
        observations = self._sim.step(action)
        state = self._sim.last_state()

        action="move_up"
        if not self._sim_settings["silent"]:
            print("action", action)
        observations = self._sim.step(action)
        state = self._sim.last_state()

        action="look_up"
        if not self._sim_settings["silent"]:
            print("action", action)
        observations = self._sim.step(action)
        state = self._sim.last_state()

        action="look_left"
        if not self._sim_settings["silent"]:
            print("action", action)
        observations = self._sim.step(action)
        state = self._sim.last_state()

        action="look_anti"
        if not self._sim_settings["silent"]:
            print("action", action)
        observations = self._sim.step(action)

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
        f=open("/scratch/yvsharish/working/aaaaa/baseline_2_exp_pose.txt","a+")
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

    def readFlow(self,name):
        if name.endswith('.pfm') or name.endswith('.PFM'):
            return readPFM(name)[0][:,:,0:2]

        f = open(name, 'rb')

        header = f.read(4)
        if header.decode("utf-8") != 'PIEH':
            raise Exception('Flow file header does not contain PIEH')

        width = np.fromfile(f, np.int32, 1).squeeze()
        height = np.fromfile(f, np.int32, 1).squeeze()

        flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

        return flow.astype(np.float32)

    def ibvs_controller(self,vik,Z,cam,error):
        #here centroid mean the image coordinates of intersted point in the form (number, 2)
        #cam is the numpy interaction matrix
        # here Z matrix should be taken from the ground truth images or should be proportional to the velocities of the points , Z - of the size #image

        w=0.15
        w=0.15*2
        w=0.5*0.15
        print(Z)
        print(Z.shape)
        Lsd=interactionMatrix(vik,cam,Z)
        # print(Lsd.shape)
        # print(error.shape)
        lamda=0.1
        mu=0.1
        H=np.matmul(Lsd.T,Lsd)
        vc=-lamda*np.matmul(np.matmul(np.linalg.pinv(H+mu*H.diagonal()),Lsd.T),error)
        #vc=-w*np.matmul(np.linalg.pinv(Lsd),error)
        # error
        return vc

    def ibvs_controller_single(self,frames,vik):
        max_frames=15
        total_frames=0
        # while total_frames < max_frames-1:
        f=open("/scratch/yvsharish/working/aaaaa/baseline_2_velocities_single_1.txt","a+")
        file_in=open("/scratch/yvsharish/working/aaaaa/baseline_2_velocities_single_1_new.txt","w")

        f1=open("/scratch/yvsharish/working/aaaaa/baseline_2_velocities_norm_1.txt","a+")
        f2=open("/scratch/yvsharish/working/aaaaa/baseline_2.txt","w")
        print('before getting flo')
        error= self.readFlow('/home/yvsharish/test/output_dir/output_img.flo' )
        print('after reading flo')
        print(error.shape)
        # print(error)
        nx, ny = (512,384)
        error=error.transpose(1,0,2)
        error=error.flatten()
        error=np.reshape(error,(nx*ny*2,-1))
        # error=
        # error=np.reshape(error,(512*384*2,1))
        # nx, ny = (512,384)
        x = np.linspace(0, nx-1, nx)
        y = np.linspace(0, ny-1, ny)
        # s=np.array([])
        # for i in range(nx):
        #     for j in range(ny):
        #         s=np.hstack((s,np.array([i,j])))
        # print(s)
        # print(sys.argv[1])
        # harish=("/home/harish/RRC/ICRA_2019/habitat/habitat-sim/image/test.depth.%05.png")
        alph='/scratch/yvsharish/working/habitat-sim/image_baseline_2_output/test.depth.' + str(frames).zfill(5) + '.png'
        print(alph)
        while(not os.path.isfile(alph)):
        # while(not os.path.isfile( '/scratch/yvsharish/working/habitat-sim/image_baseline_2_output/test.depth.00001.png')):
            print('harish')
            time.sleep(0.001)
        # ImageFile.LOAD_TRUNCATED_IMAGES = True

        if(frames==0):
            print("inside depth = 0")
            ### for true depth
            # harish="/home/yvsharish/working/habitat-sim/image_baseline_2/test.depth." + str(frames).zfill(5) +".png"
            #### for depth_network

# ImageFile.LOAD_TRUNCATED_IMAGES = True
            while(True):
                try:
                    cv2.imread(alph,cv2.IMREAD_GRAYSCALE).shape
                    break
                except:
                    time.sleep(0.001)


            harish="/scratch/yvsharish/working/habitat-sim/image_baseline_2_output/test.depth." + str(frames).zfill(5) +".png"
            # with open(harish, 'rb') as f:
            #     b = BytesIO()
            #     f.seek(15, 0)
            #
            #     b.write(f.read())
            #
            #     im = Image.open(b)
            # im.load()

        else:
            while(True):
                try:
                    cv2.imread(alph,cv2.IMREAD_GRAYSCALE).shape
                    break
                except:
                    time.sleep(0.001)

            print('shankar is great')
            harish="/scratch/yvsharish/working/habitat-sim/image_baseline_2_output/test.depth." + str(frames).zfill(5) +".png"
            print(alph)
        # harish="/home/harish/RRC/ICRA_2019/habitat-sim/image_baseline_2/test.depth.00000"+".png"
        # harish="/home/harish/RRC/ICRA_2019/habitat-sim/image_baseline_2/test.depth.00019"+".png"
        # print(harish)
        # im=Image.open(harish)
        im=cv2.imread(alph,cv2.IMREAD_GRAYSCALE)
        print(im)
        print(im.shape)
        Z=(np.array(im))
        print(Z)
        print(Z.shape)
        Z=Z.astype('float64')


        cam=np.asarray([[nx/2,0,nx/2],[0,ny/2, ny/2 ],[ 0, 0, 1]])
        # print(error)

        print('before ibvs_controller')
        vc=self.ibvs_controller(vik,Z,cam,error)
        print('after ibvs_controller')
        f1.write("%.20f\n"%(LA.norm(vc[0:3,0])))
        f1.close()
        f2.write("%.20f\n"%(LA.norm(vc[0:3,0])))
        f2.close()
        # print(vc[2,0])
        # f.write("%e %e %.20f %.20f %.20f %.20f\n" %(vc[0,0],-vc[1,0],-vc[2,0],vc[3,0],vc[4,0],vc[5,0]))
        # file_in.write("%e %e %.20f %.20f %.20f %.20f\n" %(vc[0,0],-vc[1,0],-vc[2,0],vc[3,0],vc[4,0],vc[5,0]))
        f.write("%e %e %.20f %.20f %.20f %.20f\n" %(-vc[0,0],vc[1,0],-vc[2,0],-vc[3,0],vc[4,0],-vc[5,0]))
        file_in.write("%e %e %.20f %.20f %.20f %.20f\n" %(-vc[0,0],vc[1,0],-vc[2,0],-vc[3,0],vc[4,0],-vc[5,0]))
        file_in.close()
        f.close()
        total_frames=total_frames+1


    def example(self,Vx,Vy,Vz,Wx,Wy,Wz,harish):

        HOST = '127.0.0.1'    # The remote host
        PORT = 50055              # The same port as used by the server
        #s_ = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #s_.connect(('10.42.0.1', PORT))

        start_state = self.init_common(Vx,Vy,Vz,Wx,Wy,Wz,harish)
        frames=0
        p=self._sim._default_agent
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('10.4.18.4', PORT))
        s.listen(1)
        conn, addr = s.accept()

        photo_error=2500
        nx, ny = (512,384)
        vik=np.array([])
        for i in range(nx):
            for j in range(ny):
                vik=np.hstack((vik,np.array([i,j])))
        print(vik)
        
        while photo_error>500:

            conn.send(str.encode(str(frames)))
            data = conn.recv(4096)
            print('Received', repr(data))
            print('Received data',data)
            print('Before IBVS')
            while(not os.path.isfile('/home/yvsharish/test/output_dir/output_img.flo') ):
                time.sleep(0.001)

            ####################--------FROM HERE TO HERE ibvs controller single calculates Vx Vy Vz Wx Wy Wz from flownet----
            self.ibvs_controller_single(frames,vik)

            print('After IBVS')

            f_1=open("/scratch/yvsharish/working/aaaaa/baseline_2_exp.txt","a+")
            f_1_indiv=open("/scratch/yvsharish/working/aaaaa/baseline_2_exp_indi.txt","w")
            print(harish)
            with open("/scratch/yvsharish/working/aaaaa/baseline_2_velocities_single_1_new.txt") as f:
                content = f.readlines()
            content = [x.strip() for x in content]
            mat = []
            for line in content:
                s = line.split(' ')
                print(s)
                if  len(s) == 6:
                    mat.append(s)
            Vx=float(mat[harish][0])
            Vy=float(mat[harish][1])
            Vz=float(mat[harish][2])
            Wx=float(mat[harish][3])
            Wy=float(mat[harish][4])
            Wz=float(mat[harish][5])
            #########################----------------------------------

          
            print('I am here')
            if(frames==0):
                if self._sim_settings["save_png"]:
                    if self._sim_settings["color_sensor"]:
                        self.save_color_observation(observations, frames)
                    if self._sim_settings["depth_sensor"]:
                        self.save_depth_observation(observations, frames)
                    if self._sim_settings["semantic_sensor"]:
                        self.save_semantic_observation(observations, frames)
            state=p.get_state()
            if not self._sim_settings["silent"]:
                print("-----------position\t", state.position, "\t", "rotation\t", state.rotation,"imgi",state.rotation.imag, "real",state.rotation.real,"sensorstate",state)
            frames = frames +1
            f_1_indiv.write("%0.8f %0.8f %.8f %e %e %e %e\n" %(state.position[0],state.position[1],state.position[2],state.rotation.imag[0],state.rotation.imag[1],state.rotation.imag[2],state.rotation.real))
            f_1_indiv.close()

            f_1.write("%0.8f %0.8f %.8f %e %e %e %e\n" %(state.position[0],state.position[1],state.position[2],state.rotation.imag[0],state.rotation.imag[1],state.rotation.imag[2],state.rotation.real))

            ################------------FROM HERE TO HERE homogeneous.py takes Vx Vy Vz and returns the new_pose------------------
            os.chdir("/home/yvsharish/working/baseline_2/")
            command="python2 homogeneous.py"
            os.system(command)
            with open("/scratch/yvsharish/working/aaaaa/baseline_2_after_change.txt") as f:
                content = f.readlines()
            content = [x.strip() for x in content]
            mat_ = []
            for line in content:
                s = line.split(' ')
                print(s)
                if  len(s) == 7:
                    mat_.append(s)
            ###########################-------------------


            state.position=np.array([float(mat_[harish][0]),float(mat_[harish][1]),float(mat_[harish][2])]).astype('float32')
            state.rotation=np.quaternion(float(mat_[harish][6]),float(mat_[harish][3]),float(mat_[harish][4]),float(mat_[harish][5]))
            print("position\t", state.position, "\t", "rotation\t", state.rotation)

            f_1.write("%0.8f %0.8f %.8f %e %e %e %e\n" %(state.position[0],state.position[1],state.position[2],state.rotation.imag[0],state.rotation.imag[1],state.rotation.imag[2],state.rotation.real))
            f_1.close()

            state.sensor_states={'color_sensor': habitat_sim.agent.SixDOFPose(position=np.array([state.position[0],state.position[1]+1.5,state.position[2]]), rotation= state.rotation), 'depth_sensor': habitat_sim.agent.SixDOFPose(position=np.array([state.position[0],state.position[1]+1.5,state.position[2]]), rotation=state.rotation)}
            p.set_state(state)
            self._sim._last_state=p.get_state()
            print("p.last_state",self._sim._last_state)
            observations=self._sim.get_sensor_observations()
            if self._sim_settings["save_png"]:
                if self._sim_settings["color_sensor"]:
                    self.save_color_observation(observations, frames)
                if self._sim_settings["depth_sensor"]:
                    self.save_depth_observation(observations, frames)
                if self._sim_settings["semantic_sensor"]:
                    self.save_semantic_observation(observations, frames)
            print("after observation - p.last_state",self._sim._last_state)

            ###########################CALCULATES THE photoerror and stores it in file 
            os.chdir("/home/yvsharish/working/baseline_2/")
            ##################### change here when you change for iterations
            command="python2 photo_error.py "+str(frames) + " 20"
            os.system(command)
            print('Before photo_error')
            file_in=open("/scratch/yvsharish/working/aaaaa/baseline_2_photo.txt","r")
            for line in file_in.readlines():
              photo_error=(float(line))
            print(photo_error)
            print('after_photo_error')
            os.remove('/home/yvsharish/test/output_dir/output_img.flo')


        conn.close()
        self._sim.close()
        del self._sim
