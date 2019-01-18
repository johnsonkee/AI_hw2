# coding=utf-8
from mydataset import _train_data, _test_data
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms
from mxnet.gluon import Trainer
from myNet import resnet18
from mxnet import autograd
from mxnet.gluon import loss
from mxnet import init
import mxnet as mx
import time
import gluonbook as gb

BATCH_SIZE = 128
NUMS_EPOCHS = 10
LR = 0.1
USE_CUDA = True

def train(net, train_dataloader, batch_size, nums_epochs, lr, ctx=mx.cpu()):
    trainer = Trainer(net.collect_params(), 'adam',
                      {'learning_rate': lr})
    myloss = loss.SoftmaxCrossEntropyLoss()
    for epoch in range(nums_epochs):
        train_loss, train_acc, start = 0.0, 0.0, time.time()
        for X,y in train_dataloader:
            # 原先的数据都是int8类型的，现在将其转化为float32
            # 以便输入到网络里面
            y = y.astype('float32').as_in_context(ctx)
            X = X.astype('float32').as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = myloss(y_hat,y)
            l.backward()
            trainer.step(batch_size)
            train_loss += l.mean().asscalar()
            def compute_acc(y,y_hat):
                length = len(y)
                same = 0
                for i in range(length):
                    if y[i] == y_hat[i]:
                        same += 1
                return same
            # train_acc += compute_acc(y_hat, y)
        time_s = "time %.2f sec" % (time.time() - start)
        epoch_s = ("epoch %d, loss %f,  "
                   % (epoch+1,
                      train_loss/len(train_dataloader)))
        print(epoch_s + time_s + ', lr' + str(trainer.learning_rate))

def evaluate(net, test_dataloaders,ctx):
    print(gb.evaluate_accuracy(test_dataloader,net,ctx))


if __name__ == "__main__":
    if USE_CUDA:
        ctx = mx.gpu()
    else:
        ctx = mx.cpu()
    transform_format = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2023, 0.1994, 0.2010])
    ])
    train_dataloader = DataLoader(_train_data.transform_first(transform_format),
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  last_batch='keep')

    test_dataloader = DataLoader(_test_data.transform_first(transform_format),
                                 batch_size=BATCH_SIZE,
                                 shuffle=True,
                                 last_batch='keep')

    net = resnet18(num_classes=10)
    net.initialize(ctx=ctx, init=init.Xavier())
    print("====>train")
    # train(net, train_dataloader, BATCH_SIZE, NUMS_EPOCHS, LR,ctx)
    print("====>evaluate")
    evaluate(net, test_dataloader, ctx)
