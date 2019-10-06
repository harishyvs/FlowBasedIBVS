# FlowBasedIBVS
![Pipeline](https://i.imgur.com/8VOqFsb.png)
## Contents 

This repository contains the code for running FlowBased IBVS and the results obtained.

1. Code : This folder contains the main code for running FlowBased IBVS.

	##### examples_baseline_2 folder
		This folder contains the code that runs the habitat simulator and ibvs controller.

	##### flownet2-tf
		This folder contains the implementation of the Flownet2 in tensorflow taken from [https://github.com/sampepose/flownet2-tf](https://github.com/sampepose/flownet2-tf). We have made some changes in the net.py files so as to use the flownet's output as part of our pipeline.

2. Data : This folder contains the results for various experiments performed. 

## Usage

Save the input and output images in a folder input-baseline-2

Run the below commands in 2 separate terminals.

```
cd Code/flownet2-tf
python -m src.flownet2.test_flow_depth (for flow depth)
		(or)
python -m src.flownet2.test_depth_net (for depth network)
```

```
cd Code/examples-baseline-2
python example_me_depth_net.py --width 512 --height 384 --scene path/to/habitat/scene/name.glb 
		(or)
python example_me_flow_depth.py --width 512 --height 384 --scene path/to/habitat/scene/name.glb
```

### Results

The velocities predicted, photoerror, the images as scene in the simulator are stored in output-baseline-2 output folder.

### Prerequisites

Tensorflow,
Habitat-sim (for running in the simulator, can also send the velocities predicted in a different simulator).

Here is the link to the Habitat-sim repo :
[https://github.com/facebookresearch/habitat-sim](https://github.com/facebookresearch/habitat-sim)
The repo also contains the information to download the dataset used (.glb files from gibson dataset).

### Project Page
[https://github.com/harishyvs/FlowBasedIBVS](https://github.com/harishyvs/FlowBasedIBVS)
