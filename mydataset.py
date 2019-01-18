# coding=utf-8
from mxnet.gluon.data.vision import CIFAR10
import mxnet

# 目前读取的数据集是int8类型的，然而网络的输入要求是float类型的，
# 所以要进行转化,具体转化在
_train_data = CIFAR10(root="./dataset/cifar10", train=True)
_test_data = CIFAR10(root="./dataset/cifar10", train=False)


