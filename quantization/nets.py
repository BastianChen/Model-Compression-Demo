# @Time : 2020-04-17 10:28 
# @Author : Ben 
# @Version：V 0.1
# @File : nets.py
# @desc :

import torch
from torch import nn
from torch.quantization import QuantStub, DeQuantStub


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.linear1 = nn.Linear(7 * 7 * 64, 128)
        self.relu4 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(128, 10)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.skip_add = nn.quantized.FloatFunctional()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, data):
        data = self.quant(data)
        out = self.relu1(self.bn1(self.conv1(data)))
        # data跟out的shape必须一致
        # out = self.skip_add.add(data, out)
        out = self.maxpool1(out)
        out = self.maxpool2(self.relu2(self.bn2(self.conv2(out))))
        out = self.relu3(self.bn3(self.conv3(out)))
        out = out.reshape(out.size(0), -1)
        out = self.relu4(self.linear1(out))
        out = self.linear2(out)
        out = self.dequant(out)
        return out

    def get_loss(self, output, label):
        return self.loss(output, label)

    def fuse_model(self):
        torch.quantization.fuse_modules(self, [['conv1', 'bn1', 'relu1'],
                                               ['conv2', 'bn2', 'relu2'],
                                               ['conv3', 'bn3', 'relu3'], ['linear1', 'relu4']], inplace=True)
