import argparse
import os
from ..net_depth_network_one import Mode
from .flownet2_depth_net import FlowNet2

FLAGS = None


def main():
    # Create a new network
    net = FlowNet2(mode=Mode.TEST)

    net.test(
        checkpoint='./src/flownet2/FlowNet2/flownet-2.ckpt-0',
        out_path = './src/output_dir'
    )


if __name__ == '__main__':
    main()
