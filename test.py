from mydataset import _test_data
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.data import DataLoader
from myNet import resnet18
from mxnet import cpu, gpu
from mxnet import ndarray as nd
import pandas as pd


BATCH_SIZE = 1
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

confusion_matrix = nd.zeros((10,10))

for data,label in test_dataloader:
    label_hat = net(data.as_in_context(CTX))
    label_number = label.astype('int8')
    hat_number = label_hat.argmax(axis=1).copyto(cpu())
    confusion_matrix[label_number][hat_number] += 1

confusion_matrix = confusion_matrix.asnumpy()

data = pd.DataFrame(confusion_matrix)
data.to_csv("confusion.csv",index=False, columns=False)




