# @Time : 2020-04-11 15:11 
# @Author : Ben 
# @Version：V 0.1
# @File : pruning.py
# @desc :对训练好的模型进行剪枝并保存网络模型

import torch
from pruning.nets import MyNet
from pruning.utils import weight_prune, filter_prune


class Pruning:
    def __init__(self, net_path):
        self.net = MyNet()
        self.net.load_state_dict(torch.load(net_path))

    def pruning(self):
        mask = weight_prune(self.net, 60)
        self.net.set_masks(mask, True)
        torch.save(self.net.state_dict(), "models/pruned_net_without_conv.pth")
        filter_prune(self.net, 50)
        torch.save(self.net.state_dict(), "models/pruned_net_with_conv.pth")


if __name__ == '__main__':
    pruning = Pruning("models/net.pth")
    pruning.pruning()
