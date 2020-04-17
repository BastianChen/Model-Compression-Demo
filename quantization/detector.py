# @Time : 2020-04-17 10:42 
# @Author : Ben 
# @Version：V 0.1
# @File : detector.py
# @desc :

from quantization.nets import MyNet
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os


class Detector:
    def __init__(self, net_path, isQuantize=False):
        self.device = "cpu" if torch.cuda.is_available() else "cpu"
        self.net = MyNet().to(self.device)
        if isQuantize:
            self.net = torch.jit.load(net_path)
        else:
            self.net.load_state_dict(torch.load(net_path, map_location=self.device))
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.test_data = DataLoader(datasets.MNIST("../datasets/", train=False, transform=self.trans, download=False),
                                    batch_size=100, shuffle=True, drop_last=True)
        self.net.eval()

    def detect(self):
        self.print_size_of_model()
        test_loss = 0
        correct = 0
        start = time.time()
        with torch.no_grad():
            for data, label in self.test_data:
                data, label = data.to(self.device), label.to(self.device)
                output = self.net(data)
                # test_loss += self.net.get_loss(output, label)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(label.view_as(pred)).sum().item()

        end = time.time()
        print(f"total time:{end - start}")
        # test_loss /= len(self.test_data.dataset)

        print('Test: aaccuracy: {}/{} ({:.0f}%)\n'.format(correct, len(self.test_data.dataset),
                                                          100. * correct / len(self.test_data.dataset)))

    def quantize(self):
        print("=====quantize start=====")
        self.net.fuse_model()
        self.net.qconfig = torch.quantization.default_qconfig
        torch.quantization.prepare(self.net, inplace=True)

        # Convert to quantized model
        torch.quantization.convert(self.net, inplace=True)
        self.detect()

    def print_size_of_model(self):
        torch.save(self.net.state_dict(), "temp.p")
        print('Size (MB):', os.path.getsize("temp.p") / 1e6)
        os.remove('temp.p')


if __name__ == '__main__':
    print("models/net.pth")
    detector = Detector("models/net.pth")
    detector.detect()
    print("models/net_fuse.pth")
    detector = Detector("models/net_fuse.pth", isQuantize=True)
    detector.detect()
    print("models/net_convert.pth")
    detector = Detector("models/net_convert.pth", isQuantize=True)
    detector.detect()
    # 量化后又进行了量化感知训练来提升精度
    print("models/net_convert_qat.pth")
    detector = Detector("models/net_convert_qat.pth", isQuantize=True)
    detector.detect()