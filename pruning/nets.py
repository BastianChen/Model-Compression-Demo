# @Time : 2020-04-10 11:48 
# @Author : Ben 
# @Version：V 0.1
# @File : nets.py
# @desc :编写网络以及剪枝方面的调用函数

import torch
from torch import nn
import torch.nn.functional as F
from pruning.utils import to_var


class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_flag = False

    def set_mask(self, mask):
        self.mask = to_var(mask, requires_grad=False)
        self.weight.data = self.weight.data * self.mask.data
        self.mask_flag = True

    def get_mask(self):
        print(self.mask_flag)
        return self.mask

    def forward(self, data):
        if self.mask_flag:
            weight = self.weight * self.mask
            return F.conv2d(data, weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(data, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)


class MaskedLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__(in_channels, out_channels, bias)
        self.mask_flag = False

    def set_mask(self, mask):
        self.mask = to_var(mask, requires_grad=False)
        self.weight.data = self.weight.data * self.mask.data
        self.mask_flag = True

    def get_mask(self):
        print(self.mask_flag)
        return self.mask

    def forward(self, data):
        if self.mask_flag:
            weight = self.weight * self.mask
            return F.linear(data, weight, self.bias)
        else:
            return F.linear(data, self.weight, self.bias)


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = MaskedConv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2 = MaskedConv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv3 = MaskedConv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.linear1 = MaskedLinear(7 * 7 * 64, 128)
        self.linear2 = MaskedLinear(128, 10)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, data):
        out = self.maxpool1(self.relu1(self.conv1(data)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = self.relu3(self.conv3(out))
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out

    def get_loss(self, output, label):
        return self.loss(output, label)

    def set_masks(self, masks, isLinear=False):
        # Should be a less manual way to set masks
        # Leave it for the future
        if isLinear:
            self.linear1.set_mask(masks[0])
            self.linear2.set_mask(masks[1])
        else:
            self.conv1.set_mask(torch.from_numpy(masks[0]))
            self.conv2.set_mask(torch.from_numpy(masks[1]))
            self.conv3.set_mask(torch.from_numpy(masks[2]))


if __name__ == '__main__':
    net = MyNet()
    for p in net.conv1.parameters():
        print(p.data.size())
    for p in net.linear1.parameters():
        print(p.data.size())
