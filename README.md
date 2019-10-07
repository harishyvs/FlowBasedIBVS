# FlowBasedIBVS
![Pipeline](https://i.imgur.com/8VOqFsb.png)
## Contents 

This repository contains the code for running FlowBased IBVS and the results obtained.

1. Code : This folder contains the main code for running FlowBased IBVS.

	##### habitat_sim_client folder
		This folder contains the code that runs the habitat simulator and ibvs controller.

	##### flownet2-tf
		This folder contains the implementation of the Flownet2 in tensorflow taken from [https://github.com/sampepose/flownet2-tf]. We have made changes to the net.py files so as to use the flownet's output as part of our pipeline.

2. Data : This folder contains the results for various experiments performed. 

Please download the FlowNet2 folder from the following link and store it in Code/flownet2-tf/src/flownet2/

https://drive.google.com/open?id=13rvb_HCWc673C43YVZurwlxJTdX42UUd

Final path of accessing the downloaded folder should be Code/flownet2-tf/src/flownet2/FlowNet2

## Usage

Save the input(initial) and output(desired) images in the folder : Code/flownet2-tf/src/image_baseline_2 as initial_image.png and desired_image.png.

NOTE: For the initial depth image in case of flowdepth, store it as initial_depth.png in the same folder.

Run the below commands in 2 separate terminals in the following order .

[Server Code]
```
cd Code/flownet2-tf
python -m src.flownet2.test_flow_depth (for flow depth)
		(or)
python -m src.flownet2.test_depth_net (for depth network)
```

[Client Code]
```
cd Code/habitat_sim_client
python example_me_depth_net.py --width 512 --height 384 --scene path/to/habitat/scene/name.glb --save_png --depth_sensor --max_frames 20 
		(or)
python example_me_flow_depth.py --width 512 --height 384 --scene path/to/habitat/scene/name.glb --save_png --depth_sensor --max_frames 20
```

### Results

The quantitative results such as photoerror, velocities predicted in every iteration are stored in the folder : Code/flownet2-tf/src/aaaaa

The qualitative results : The images taken in the simulator, depth images and flo images are stored in the folder : Code/flownet2-tf/src/image_baseline_2_output

### Prerequisites

Tensorflow,
Habitat-sim (for running in the simulator, can also send the velocities predicted in a different simulator).

Here is the link to the Habitat-sim repo :
[https://github.com/facebookresearch/habitat-sim](https://github.com/facebookresearch/habitat-sim)
The repo also contains the information to download the dataset used (.glb files from gibson dataset).

### Project Page
[https://github.com/harishyvs/FlowBasedIBVS](https://github.com/harishyvs/FlowBasedIBVS)
