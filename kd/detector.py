# @Time : 2020-04-19 14:28 
# @Author : Ben 
# @Version：V 0.1
# @File : detector.py
# @desc :测试老师网络跟学生网络的性能差异

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
from kd.nets import TeacherNet, StudentNet
from torch import nn
import time


class Detector:
    def __init__(self, net_path, isTeacher=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if isTeacher:
            self.net = TeacherNet().to(self.device)
        else:
            self.net = StudentNet().to(self.device)
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.1307], [0.3081])
        ])
        self.test_data = DataLoader(datasets.MNIST("../datasets/", train=False, transform=self.trans, download=False),
                                    batch_size=100, shuffle=True)
        self.net.load_state_dict(torch.load(net_path, map_location=self.device))
        self.net.eval()

    def detect(self):
        correct = 0
        start = time.time()
        with torch.no_grad():
            for data, label in self.test_data:
                data, label = data.to(self.device), label.to(self.device)
                output = self.net(data)
                pred = output.argmax(dim=-1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(label.view_as(pred)).sum().item()
        end = time.time()
        print(f"total time:{end - start}")

        print('Test: average  accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(self.test_data.dataset),
                                                                  100. * correct / len(self.test_data.dataset)))


if __name__ == '__main__':
    print("teacher_net")
    detector = Detector("models/teacher_net.pth")
    detector.detect()
    print("student_net")
    detector = Detector("models/student_net.pth", False)
    detector.detect()
