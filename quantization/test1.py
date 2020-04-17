# @Time : 2020-04-16 20:59 
# @Author : Ben 
# @Version：V 0.1
# @File : test1.py
# @desc :静态量化

import torch
from torch import nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub
import os


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = x.view(-1, int(x.nelement() / x.shape[0]))
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # return x
        x = self.quant(x)
        x = self.maxpool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.maxpool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(self, [['conv1', 'bn1', 'relu1'],
                                               ['conv2', 'bn2', 'relu2'],
                                               ['fc1', 'relu3'], ['fc2', 'relu4']], inplace=True)
        # if self.downsample:
        #     torch.quantization.fuse_modules(self.downsample, ['0', '1'], inplace=True)

    # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
    # This operation does not change the numerics
    # def fuse_model(self):
    #     for m in self.modules():
    #         if type(m) == ConvBNReLU:
    #             torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
    #         if type(m) == InvertedResidual:
    #             for idx in range(len(m.conv)):
    #                 if type(m.conv[idx]) == nn.Conv2d:
    #                     torch.quantization.fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)


model = LeNet()
print(model)

model.eval()

model.fuse_model()

torch.save(model.state_dict(), "temp.p")
print('Size (MB):', os.path.getsize("temp.p") / 1e6)
os.remove('temp.p')

model.qconfig = torch.quantization.default_qconfig
# print(model.qconfig)

torch.quantization.prepare(model, inplace=True)

# # Convert to quantized model
torch.quantization.convert(model, inplace=True)
torch.save(model.state_dict(), "temp.p")
print('Size (MB):', os.path.getsize("temp.p") / 1e6)
os.remove('temp.p')
