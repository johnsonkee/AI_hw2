import mxnet as mx

net = mx.gluon.nn.Dense(10)
net.initialize()
a = mx.ndarray.ones((1,10))
print(a.dtype)
print(net(a))