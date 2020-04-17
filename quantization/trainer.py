# @Time : 2020-04-17 10:39 
# @Author : Ben 
# @Version：V 0.1
# @File : trainer.py
# @desc :

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
from quantization.nets import MyNet


class Trainer:
    def __init__(self, save_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_path = save_path
        self.net = MyNet().to(self.device)
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.train_data = DataLoader(datasets.MNIST("../datasets/", train=True, transform=self.trans, download=False),
                                     batch_size=100, shuffle=True, drop_last=True)
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.net.train()

    def train(self):
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
            torch.save(self.net.state_dict(), self.save_path)


if __name__ == '__main__':
    trainer = Trainer("models/net.pth")
    trainer.train()
