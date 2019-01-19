# coding=utf-8
from mydataset import _train_data, _test_data
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms
from mxnet.gluon import Trainer
from myNet import resnet18
from mxnet import autograd
from mxnet.gluon import loss
from mxnet import init
from mxnet import cpu
from mxnet import gpu
import time
import gluonbook as gb

from argparse import ArgumentParser



def parse_args():
    parser = ArgumentParser(description="Train a resnet18 for"
                                        " cifar10 dataset")
    parser.add_argument('--data', type=str, default='./dataset/cifar10',
                        help='path to test and training data files')
    parser.add_argument('-e', '--epochs', type=int, default=20,
                        help='number of epochs for training')
    parser.add_argument('-b', '--batch_size', type=int, default=128,
                        help='number of examples for each iteration')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1,
                        help='learning rate for optimizer')
    parser.add_argument('-lp', '--learning_period', type=float, default=80,
                        help='after learning_period, lr = lr * lr_decay')
    parser.add_argument('-lc','--learning_decay',type=float, default=0.1,
                        help='after learning_period, lr = lr * lr_decay')
    parser.add_argument('-wd',type=float, default=5e-4,
                        help='weight decay, used in SGD optimization')
    parser.add_argument('--gpu', type=bool, default=False,
                        help='use available GPUs')
    parser.add_argument('--use_model_path', type=str, default='resnet18.params',
                        help='the path of the pre-trained model')
    parser.add_argument('--use_model', type=bool, default=False,
                        help='whether use a pre-trained model')
    parser.add_argument('--save_model_path', type=str, default='resnet18.params',
                        help='where to save the trained model')
    parser.add_argument('--save_model', type=bool, default=True,
                        help='whether save the model')

    return parser.parse_args()


def train(net,
          train_dataloader,
          test_dataloader,
          batch_size,
          nums_epochs,
          lr,
          ctx,
          wd,
          lr_period,
          lr_decay):
    trainer = Trainer(net.collect_params(), 'sgd',
                      {'learning_rate': lr, 'momentum':0.9, 'wd': wd})
    myloss = loss.SoftmaxCrossEntropyLoss()
    for epoch in range(nums_epochs):
        train_loss, train_acc, start = 0.0, 0.0, time.time()
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for X, y in train_dataloader:
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
        train_acc = evaluate(net,train_dataloader,ctx)
        time_s = "time %.2f sec" % (time.time() - start)
        test_acc = evaluate(net,test_dataloader,ctx)
        epoch_s = ("epoch %d, loss %f, train_acc %f, test_acc %f, "
                   % (epoch+1,
                      train_loss/len(train_dataloader),
                      train_acc,
                      test_acc))
        print(epoch_s + time_s + ', lr' + str(trainer.learning_rate))

def evaluate(net, test_dataloader, ctx):
    return gb.evaluate_accuracy(test_dataloader, net, ctx)

def main():
    args = parse_args()

    BATCH_SIZE = args.batch_size
    NUMS_EPOCHS = args.epochs
    LR = args.learning_rate
    USE_CUDA = args.gpu
    WD = args.wd
    LR_PERIOD = args.learning_period
    LR_DECAY = args.learning_decay
    MODEL_PATH = args.use_model_path
    USE_MODEL = args.use_model
    SAVE_MODEL = args.save_model

    if USE_CUDA:
        ctx = gpu()
    else:
        ctx = cpu()
    transform_train = transforms.Compose([
        transforms.Resize(40),
        transforms.RandomResizedCrop(32,scale=(0.64,1.0),
                                     ratio=(1.0, 1.0)),
        transforms.RandomFlipLeftRight(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2023, 0.1994, 0.2010])
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2023, 0.1994, 0.2010])
    ])
    train_dataloader = DataLoader(_train_data.transform_first(transform_train),
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  last_batch='keep')

    test_dataloader = DataLoader(_test_data.transform_first(transform_test),
                                 batch_size=BATCH_SIZE,
                                 shuffle=True,
                                 last_batch='keep')

    net = resnet18(num_classes=10)
    net.hybridize()

    if USE_MODEL:
        net.load_parameters(MODEL_PATH,ctx=ctx)
    else:
        net.initialize(ctx=ctx, init=init.Xavier())
    print("====>train and test")
    train(net, train_dataloader, test_dataloader,
          BATCH_SIZE, NUMS_EPOCHS, LR, ctx, WD,LR_PERIOD, LR_DECAY)
    if SAVE_MODEL:
        net.save_parameters(MODEL_PATH)


if __name__ == "__main__":
    main()

