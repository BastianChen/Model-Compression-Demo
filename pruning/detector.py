# @Time : 2020-04-10 11:48 
# @Author : Ben 
# @Version：V 0.1
# @File : detector.py
# @desc :测试类,加载网络模型并进行剪枝

from pruning.nets import MyNet
import torch
from copy import deepcopy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pruning.utils import weight_prune, plot_weights, filter_prune
import time


class Detector:
    def __init__(self, net_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = MyNet().to(self.device)
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.test_data = DataLoader(datasets.MNIST("../datasets/", train=False, transform=self.trans, download=False),
                                    batch_size=100, shuffle=True, drop_last=True)
        # 如果没有GPU的话把在GPU上训练的参数放在CPU上运行，cpu-->gpu 1:lambda storage, loc: storage.cuda(1)
        self.map_location = None if torch.cuda.is_available() else lambda storage, loc: storage
        self.net.load_state_dict(torch.load(net_path, map_location=self.map_location))
        self.net.eval()

    def detect(self):
        test_loss = 0
        correct = 0
        start = time.time()
        with torch.no_grad():
            for data, label in self.test_data:
                data, label = data.to(self.device), label.to(self.device)
                output = self.net(data)
                test_loss += self.net.get_loss(output, label)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(label.view_as(pred)).sum().item()

        end = time.time()
        print(f"total time:{end - start}")
        test_loss /= len(self.test_data.dataset)

        print('\nTest: average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(self.test_data.dataset),
            100. * correct / len(self.test_data.dataset)))

        # print("\n=====Pruning=======\n")
        # pruned_net = deepcopy(self.net)
        # self.pruning(pruned_net)
        # return self.net, pruned_net

    # def pruning(self, pruned_net):
    #     mask = weight_prune(pruned_net, 60)
    #     pruned_net.set_masks(mask, True)
    #     filter_prune(pruned_net, 50)
    #     test_loss = 0
    #     correct = 0
    #     start = time.time()
    #     with torch.no_grad():
    #         for data, label in self.test_data:
    #             data, label = data.to(self.device), label.to(self.device)
    #             output = pruned_net(data)
    #             test_loss += pruned_net.get_loss(output, label)
    #             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    #             correct += pred.eq(label.view_as(pred)).sum().item()
    #
    #     end = time.time()
    #     print(f"total time:{end - start}")
    #     test_loss /= len(self.test_data.dataset)
    #
    #     print('\nTest: average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)'.format(
    #         test_loss, correct, len(self.test_data.dataset),
    #         100. * correct / len(self.test_data.dataset)))


if __name__ == '__main__':
    print("models/net.pth")
    detector1 = Detector("models/net.pth")
    detector1.detect()
    # detector2 = Detector("models/pruned_net_without_conv.pth")
    # detector2.detect()
    # detector3 = Detector("models/pruned_net_with_conv.pth")
    # detector3.detect()
    # net, pruned_net = detector.detect()
    # torch.save(pruned_net.state_dict(), "models/pruned_net.pth")
    # plot_weights(net)
    # plot_weights(pruned_net)

    for i in range(1, 10):
        amount = 0.1 * i
        print(f"models/pruned_net_with_torch_{amount:.1f}.pth")
        detector1 = Detector(f"models/pruned_net_with_torch_{amount:.1f}.pth")
        detector1.detect()
