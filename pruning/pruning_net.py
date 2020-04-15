# @Time : 2020-04-11 15:11 
# @Author : Ben 
# @Version：V 0.1
# @File : pruning.py
# @desc :对训练好的模型进行剪枝并保存网络模型

import torch
from pruning.nets import MyNet
from pruning.utils import weight_prune, filter_prune
import torch.nn.utils.prune as prune


class Pruning:
    def __init__(self, net_path, amount):
        self.net = MyNet()
        self.net.load_state_dict(torch.load(net_path))
        self.parameters_to_prune = (
            (self.net.conv1, 'weight'),
            (self.net.conv2, 'weight'),
            (self.net.conv3, 'weight'),
            (self.net.linear1, 'weight'),
            (self.net.linear2, 'weight'),
        )
        self.amount = amount

    def pruning(self):
        prune.global_unstructured(
            self.parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.amount,
        )
        # print(self.net.state_dict().keys())
        # 删除weight_orig 、weight_mask以及forward_pre_hook
        prune.remove(self.net.conv1, 'weight')
        prune.remove(self.net.conv2, 'weight')
        prune.remove(self.net.conv3, 'weight')
        prune.remove(self.net.linear1, 'weight')
        prune.remove(self.net.linear2, 'weight')
        # print(self.net.linear1.weight)
        # mask = weight_prune(self.net, 60)
        # self.net.set_masks(mask, True)
        # torch.save(self.net.state_dict(), "self.nets/pruned_net_without_conv.pth")
        # filter_prune(self.net, 50)
        print(
            "Sparsity in conv1.weight: {:.2f}%".format(
                100. * float(torch.sum(self.net.conv1.weight == 0))
                / float(self.net.conv1.weight.nelement())
            )
        )
        print(
            "Sparsity in conv2.weight: {:.2f}%".format(
                100. * float(torch.sum(self.net.conv2.weight == 0))
                / float(self.net.conv2.weight.nelement())
            )
        )
        print(
            "Sparsity in conv3.weight: {:.2f}%".format(
                100. * float(torch.sum(self.net.conv3.weight == 0))
                / float(self.net.conv3.weight.nelement())
            )
        )
        print(
            "Sparsity in linear1.weight: {:.2f}%".format(
                100. * float(torch.sum(self.net.linear1.weight == 0))
                / float(self.net.linear1.weight.nelement())
            )
        )
        print(
            "Sparsity in linear2.weight: {:.2f}%".format(
                100. * float(torch.sum(self.net.linear2.weight == 0))
                / float(self.net.linear2.weight.nelement())
            )
        )
        print(
            "Global sparsity: {:.2f}%".format(
                100. * float(
                    torch.sum(self.net.conv1.weight == 0)
                    + torch.sum(self.net.conv2.weight == 0)
                    + torch.sum(self.net.conv3.weight == 0)
                    + torch.sum(self.net.linear1.weight == 0)
                    + torch.sum(self.net.linear2.weight == 0)
                )
                / float(
                    self.net.conv1.weight.nelement()
                    + self.net.conv2.weight.nelement()
                    + self.net.conv3.weight.nelement()
                    + self.net.linear1.weight.nelement()
                    + self.net.linear2.weight.nelement()
                )
            )
        )
        # torch.save(self.net.state_dict(), "models/pruned_net_with_conv.pth")
        torch.save(self.net.state_dict(), f"models/pruned_net_with_torch_{self.amount:.1f}_l1.pth")


if __name__ == '__main__':
    for i in range(1, 10):
        pruning = Pruning("models/net.pth", 0.1 * i)
        pruning.pruning()
