# @Time : 2020-04-15 15:18
# @Author : Ben
# @Version：V 0.1
# @File : test.py
# @desc :使用pytorch实现动态量化实验

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import quantize_dynamic
import os


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = LeNet()

quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
)
print(quantized_model)


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')


print_size_of_model(model)
print_size_of_model(quantized_model)