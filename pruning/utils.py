# @Time : 2020-04-10 13:59 
# @Author : Ben 
# @Version：V 0.1
# @File : utils.py
# @desc :工具类

import torch
import numpy as np
from matplotlib import pyplot as plt


def to_var(x, requires_grad=False):
    """
    Automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return x.clone().detach().requires_grad_(requires_grad)


def weight_prune(model, pruning_perc):
    '''
    Prune pruning_perc % weights layer-wise
    '''
    threshold_list = []
    for p in model.parameters():
        if len(p.data.size()) != 1:  # bias
            weight = p.cpu().data.abs().numpy().flatten()
            threshold = np.percentile(weight, pruning_perc)
            threshold_list.append(threshold)

    # generate mask
    masks = []
    idx = 0
    for p in model.parameters():
        if len(p.data.size()) != 1:
            pruned_inds = p.data.abs() > threshold_list[idx]
            masks.append(pruned_inds.float())
            idx += 1
    return masks


def plot_weights(model):
    modules = [module for module in model.modules()]
    num_sub_plot = 0
    for i, layer in enumerate(modules):
        print(i)
        if hasattr(layer, 'weight'):
            plt.subplot(131 + num_sub_plot)
            w = layer.weight.data
            w_one_dim = w.cpu().numpy().flatten()
            plt.hist(w_one_dim[w_one_dim != 0], bins=50)
            num_sub_plot += 1
    plt.show()
