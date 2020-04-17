# @Time : 2020-04-17 10:28 
# @Author : Ben 
# @Version：V 0.1
# @File : nets.py
# @desc :

import torch
from torch import nn
from torch.quantization import QuantStub, DeQuantStub


class Conv(nn.Sequential):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, groups=groups, bias=bias),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )


class ResBlock(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.layer = nn.Sequential(
            Conv(input_channels, input_channels // 2, 1),
            Conv(input_channels // 2, input_channels // 2, 3, 1, 1),
            Conv(input_channels // 2, input_channels, 1)
        )
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, data):
        return self.skip_add.add(data, self.layer(data))


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            Conv(1, 32, 3, 1, 1),
            ResBlock(32),
            nn.MaxPool2d(2),
            Conv(32, 64, 3, 1, 1),
            ResBlock(64),
            ResBlock(64),
            nn.MaxPool2d(2),
            Conv(64, 64, 3, 1, 1),
        )

        self.linear1 = nn.Linear(7 * 7 * 64, 128)
        self.relu4 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(128, 10)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.loss = nn.CrossEntropyLoss()

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, data):
        data = self.quant(data)
        out = self.layer(data)
        out = out.reshape(out.size(0), -1)
        out = self.relu4(self.linear1(out))
        out = self.linear2(out)
        out = self.dequant(out)
        return out

    def get_loss(self, output, label):
        return self.loss(output, label)

    def fuse_model(self):
        for m in self.modules():
            if type(m) == Conv:
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
        torch.quantization.fuse_modules(self, ['linear1', 'relu4'], inplace=True)


if __name__ == '__main__':
    net = MyNet()
    net.fuse_model()
