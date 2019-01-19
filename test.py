from mydataset import _test_data
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.data import DataLoader
from myNet import resnet18
from mxnet import cpu, gpu
import gluonbook as gb


BATCH_SIZE = 128
MODEL_PATH = 'resnet18.params'
CTX = gpu()

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465],
                         [0.2023, 0.1994, 0.2010])
])

test_dataloader = DataLoader(_test_data.transform_first(transform_test),
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             last_batch='keep')

net = resnet18(10)
net.load_parameters(MODEL_PATH,ctx=CTX)

test_acc = gb.evaluate_accuracy(test_dataloader,net,ctx=CTX)
print(test_acc)

