# @Time : 2020-04-17 17:06 
# @Author : Ben 
# @Version：V 0.1
# @File : qat.py
# @desc :Quantization-aware training(量化感知训练提高量化后模型的精度)

import torch
from torch import nn
from quantization.nets import MyNet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import os

import torch
from torch import nn
from quantization.nets import MyNet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import os


class QAT:
    def __init__(self, net_path):
        self.device = "cpu" if torch.cuda.is_available() else "cpu"
        self.net = MyNet().to(self.device)
        self.net.load_state_dict(torch.load(net_path, map_location=self.device))
        self.net.eval()
        self.net.fuse_model()
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.train_data = DataLoader(datasets.MNIST("../datasets/", train=True, transform=self.trans, download=False),
                                     batch_size=100, shuffle=True, drop_last=True)
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.net.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

    def train(self):
        torch.quantization.prepare_qat(self.net, inplace=True)
        for epoch in range(1, 3):
            total = 0
            for i, (data, label) in enumerate(self.train_data):
                data, label = data.to(self.device), label.to(self.device)
                output = self.net(data)
                loss = self.net.get_loss(output, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total += len(data)
                progress = math.ceil(i / len(self.train_data) * 50)
                print("\rTrain epoch %d: %d/%d, [%-51s] %d%%" %
                      (epoch, total, len(self.train_data.dataset),
                       '-' * progress + '>', progress * 2), end='')
        # Check the accuracy after each epoch
        self.net = torch.quantization.convert(self.net.eval(), inplace=False)
        torch.jit.save(torch.jit.script(self.net), "models/net_convert_qat.pth")
        # torch.jit.save(torch.jit.trace(self.net, example), "models/net_convert.pth")
        print('\nSize (MB):', os.path.getsize("models/net_convert.pth") / 1e6)


if __name__ == '__main__':
    qat = QAT("models/net.pth")
    qat.train()
