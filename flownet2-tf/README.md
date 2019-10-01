## FlowNet2 (TensorFlow)

This repo contains FlowNet2[1] for TensorFlow. It includes FlowNetC, S, CS, CSS, CSS-ft-sd, SD, and 2.

### Installation
```
pip install enum
pip install pypng
pip install matplotlib
pip install image
pip install scipy
pip install numpy
pip install tensorflow
```

Linux:
`sudo apt-get install python-tk`

You must have CUDA-8 installed:
`make all`


### Flow Generation (1 image pair)

```
python -m src.flownet2.test --input_a data/samples/0img0.ppm --input_b data/samples/0img1.ppm --out ./
```

### Flowdepth (As a part of the module)

```
python -m src.flownet2.test_new_2
```
