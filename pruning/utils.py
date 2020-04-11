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
            if len(p.data.size()) != 4:
                weight = p.cpu().data.abs().numpy().flatten()
                threshold = np.percentile(weight, pruning_perc)
                threshold_list.append(threshold)

    # generate mask
    masks = []
    idx = 0
    for p in model.parameters():
        if len(p.data.size()) != 1:
            if len(p.data.size()) != 4:
                pruned_inds = p.data.abs() > threshold_list[idx]
                masks.append(pruned_inds.float())
                idx += 1
    return masks


"""Reference https://github.com/zepx/pytorch-weight-prune/"""


def prune_rate(model, verbose=False):
    """
    Print out prune rate for each layer and the whole network
    """
    total_nb_param = 0
    nb_zero_param = 0

    layer_id = 0

    for parameter in model.parameters():

        param_this_layer = 1
        for dim in parameter.data.size():
            param_this_layer *= dim
        total_nb_param += param_this_layer

        # only pruning linear and conv layers
        if len(parameter.data.size()) != 1:
            layer_id += 1
            zero_param_this_layer = np.count_nonzero(parameter.cpu().data.numpy() == 0)
            nb_zero_param += zero_param_this_layer

            if verbose:
                print("Layer {} | {} layer | {:.2f}% parameters pruned".format(
                    layer_id,
                    'Conv' if len(parameter.data.size()) == 4 \
                        else 'Linear',
                    100. * zero_param_this_layer / param_this_layer,
                ))
    pruning_perc = 100. * nb_zero_param / total_nb_param
    if verbose:
        print("Final pruning rate: {:.2f}%".format(pruning_perc))
    return pruning_perc


def arg_nonzero_min(a):
    """
    nonzero argmin of a non-negative array
    """

    if not a:
        return

    min_ix, min_v = None, None
    # find the starting value (should be nonzero)
    for i, e in enumerate(a):
        if e != 0:
            min_ix = i
            min_v = e
    if not min_ix:
        print('Warning: all zero')
        return np.inf, np.inf

    # search for the smallest nonzero
    for i, e in enumerate(a):
        if e < min_v and e != 0:
            min_v = e
            min_ix = i

    return min_v, min_ix


def prune_one_filter(model, masks):
    '''
    Pruning one least ``important'' feature map by the scaled l2norm of
    kernel weights
    arXiv:1611.06440
    '''
    NO_MASKS = False
    # construct masks if there is not yet
    if not masks:
        masks = []
        NO_MASKS = True

    values = []
    for p in model.parameters():

        if len(p.data.size()) == 4:  # nasty way of selecting conv layer
            p_np = p.data.cpu().numpy()

            # construct masks if there is not
            if NO_MASKS:
                masks.append(np.ones(p_np.shape).astype('float32'))

            # find the scaled l2 norm for each filter this layer
            value_this_layer = np.square(p_np).sum(axis=1).sum(axis=1).sum(axis=1) / (
                    p_np.shape[1] * p_np.shape[2] * p_np.shape[3])
            # normalization (important)
            value_this_layer = value_this_layer / np.sqrt(np.square(value_this_layer).sum())
            # 找到通道值的最小值以及索引
            min_value, min_ind = arg_nonzero_min(list(value_this_layer))
            values.append([min_value, min_ind])

    assert len(masks) == len(values), "something wrong here"

    values = np.array(values)

    # set mask corresponding to the filter to prune
    to_prune_layer_ind = np.argmin(values[:, 0])
    to_prune_filter_ind = int(values[to_prune_layer_ind, 1])
    masks[to_prune_layer_ind][to_prune_filter_ind] = 0.

    #     print('Prune filter #{} in layer #{}'.format(
    #         to_prune_filter_ind,
    #         to_prune_layer_ind))

    return masks


def filter_prune(model, pruning_perc):
    '''
    Prune filters one by one until reach pruning_perc
    (not iterative pruning)
    '''
    masks = []
    current_pruning_perc = 0.

    while current_pruning_perc < pruning_perc:
        masks = prune_one_filter(model, masks)
        model.set_masks(masks)
        current_pruning_perc = prune_rate(model, verbose=False)
    #         print('{:.2f} pruned'.format(current_pruning_perc))
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


if __name__ == '__main__':
    a = np.random.randn(3, 5)
    # print(a)
    # print(abs(a))
    print(abs(a).flatten())
    b = np.percentile(abs(a).flatten(), 20)
    print(b)
    b = np.percentile(abs(a).flatten(), 40)
    print(b)
    b = np.percentile(abs(a).flatten(), 60)
    print(b)
    b = np.percentile(abs(a).flatten(), 80)
    print(b)
