from mydataset import _test_data
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.data import DataLoader
from myNet import resnet18
from mxnet import cpu, gpu
from mxnet import ndarray as nd
from mxnet.test_utils import list_gpus
import pandas as pd


BATCH_SIZE = 1
MODEL_PATH = 'resnet18.params'

if list_gpus():
    CTX = gpu()
else:
    CTX = cpu()

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
# net.load_parameters(MODEL_PATH,ctx=CTX)
net.initialize(ctx=CTX)

confusion_matrix = nd.zeros((10,10))

print("====>make confusion matrix")
for data,label in test_dataloader:
    label_hat = net(data.as_in_context(CTX))
    label_number = label.astype('int8').copyto(cpu()).asscalar()
    hat_number = label_hat.argmax(axis=1).astype('int8').copyto(cpu())
    hat_number = hat_number.asscalar()
    confusion_matrix[label_number-1][hat_number-1] += 1

confusion_matrix = confusion_matrix.asnumpy()

data = pd.DataFrame(confusion_matrix,dtype='int8')
data.to_csv("confusion.csv",index=False, header=False)




