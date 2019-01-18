import mxnet as mx
from argparse import ArgumentParser

net = mx.gluon.nn.Dense(10)
net.initialize()
a = mx.ndarray.ones((1, 10))
