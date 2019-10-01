#!/home/yvsharish/test/test_tf/bin/python
import argparse
import os
from ..net import Mode
from  .flownet2 import FlowNet2

FLAGS = None


def init():
    # Create a new network
    net = FlowNet2(mode=Mode.TEST)

init()
