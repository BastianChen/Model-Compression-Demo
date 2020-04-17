# @Time : 2020-04-15 15:18
# @Author : Ben
# @Version：V 0.1
# @File : test.py
# @desc :使用pytorch实现量化实验

# imports

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.quantization import quantize_dynamic


# class LSTMModel(nn.Module):
#     """Container module with an encoder, a recurrent module, and a decoder."""
#
#     def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
#         super(LSTMModel, self).__init__()
#         self.drop = nn.Dropout(dropout)
#         self.encoder = nn.Embedding(ntoken, ninp)
#         self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
#         self.decoder = nn.Linear(nhid, ntoken)
#
#         self.init_weights()
#
#         self.nhid = nhid
#         self.nlayers = nlayers
#
#     def init_weights(self):
#         initrange = 0.1
#         self.encoder.weight.data.uniform_(-initrange, initrange)
#         self.decoder.bias.data.zero_()
#         self.decoder.weight.data.uniform_(-initrange, initrange)
#
#     def forward(self, input, hidden):
#         emb = self.drop(self.encoder(input))
#         output, hidden = self.rnn(emb, hidden)
#         output = self.drop(output)
#         decoded = self.decoder(output)
#         return decoded, hidden
#
#     def init_hidden(self, bsz):
#         weight = next(self.parameters())
#         return (weight.new_zeros(self.nlayers, bsz, self.nhid),
#                 weight.new_zeros(self.nlayers, bsz, self.nhid))


# 动态量化

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

# 静态量化

# import numpy as np
# import torch
# import torch.nn as nn
# import torchvision
# from torch.utils.data import DataLoader
# from torchvision import datasets
# import torchvision.transforms as transforms
# import os
# import time
# import sys
# import torch.quantization
#
# # # Setup warnings
# import warnings
#
# warnings.filterwarnings(
#     action='ignore',
#     category=DeprecationWarning,
#     module=r'.*'
# )
# warnings.filterwarnings(
#     action='default',
#     module=r'torch.quantization'
# )
#
# # Specify random seed for repeatable results
# torch.manual_seed(191009)
#
# from torch.quantization import QuantStub, DeQuantStub
#
#
# def _make_divisible(v, divisor, min_value=None):
#     """
#     This function is taken from the original tf repo.
#     It ensures that all layers have a channel number that is divisible by 8
#     It can be seen here:
#     https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
#     :param v:
#     :param divisor:
#     :param min_value:
#     :return:
#     """
#     if min_value is None:
#         min_value = divisor
#     new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
#     # Make sure that round down does not go down by more than 10%.
#     if new_v < 0.9 * v:
#         new_v += divisor
#     return new_v
#
#
# class ConvBNReLU(nn.Sequential):
#     def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
#         padding = (kernel_size - 1) // 2
#         super(ConvBNReLU, self).__init__(
#             nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
#             nn.BatchNorm2d(out_planes, momentum=0.1),
#             # Replace with ReLU
#             nn.ReLU(inplace=False)
#         )
#
#
# class InvertedResidual(nn.Module):
#     def __init__(self, inp, oup, stride, expand_ratio):
#         super(InvertedResidual, self).__init__()
#         self.stride = stride
#         assert stride in [1, 2]
#
#         hidden_dim = int(round(inp * expand_ratio))
#         self.use_res_connect = self.stride == 1 and inp == oup
#
#         layers = []
#         if expand_ratio != 1:
#             # pw
#             layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
#         layers.extend([
#             # dw
#             ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
#             # pw-linear
#             nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(oup, momentum=0.1),
#         ])
#         self.conv = nn.Sequential(*layers)
#         # Replace torch.add with floatfunctional
#         self.skip_add = nn.quantized.FloatFunctional()
#
#     def forward(self, x):
#         if self.use_res_connect:
#             return self.skip_add.add(x, self.conv(x))
#         else:
#             return self.conv(x)
#
#
# class MobileNetV2(nn.Module):
#     def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
#         """
#         MobileNet V2 main class
#
#         Args:
#             num_classes (int): Number of classes
#             width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
#             inverted_residual_setting: Network structure
#             round_nearest (int): Round the number of channels in each layer to be a multiple of this number
#             Set to 1 to turn off rounding
#         """
#         super(MobileNetV2, self).__init__()
#         block = InvertedResidual
#         input_channel = 32
#         last_channel = 1280
#
#         if inverted_residual_setting is None:
#             inverted_residual_setting = [
#                 # t, c, n, s
#                 [1, 16, 1, 1],
#                 [6, 24, 2, 2],
#                 [6, 32, 3, 2],
#                 [6, 64, 4, 2],
#                 [6, 96, 3, 1],
#                 [6, 160, 3, 2],
#                 [6, 320, 1, 1],
#             ]
#
#         # only check the first element, assuming user knows t,c,n,s are required
#         if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
#             raise ValueError("inverted_residual_setting should be non-empty "
#                              "or a 4-element list, got {}".format(inverted_residual_setting))
#
#         # building first layer
#         input_channel = _make_divisible(input_channel * width_mult, round_nearest)
#         self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
#         features = [ConvBNReLU(3, input_channel, stride=2)]
#         # building inverted residual blocks
#         for t, c, n, s in inverted_residual_setting:
#             output_channel = _make_divisible(c * width_mult, round_nearest)
#             for i in range(n):
#                 stride = s if i == 0 else 1
#                 features.append(block(input_channel, output_channel, stride, expand_ratio=t))
#                 input_channel = output_channel
#         # building last several layers
#         features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
#         # make it nn.Sequential
#         self.features = nn.Sequential(*features)
#         self.quant = QuantStub()
#         self.dequant = DeQuantStub()
#         # building classifier
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(self.last_channel, num_classes),
#         )
#
#         # weight initialization
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.zeros_(m.bias)
#
#     def forward(self, x):
#
#         x = self.quant(x)
#
#         x = self.features(x)
#         x = x.mean([2, 3])
#         x = self.classifier(x)
#         x = self.dequant(x)
#         return x
#
#     # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
#     # This operation does not change the numerics
#     def fuse_model(self):
#         for m in self.modules():
#             if type(m) == ConvBNReLU:
#                 torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
#             if type(m) == InvertedResidual:
#                 for idx in range(len(m.conv)):
#                     if type(m.conv[idx]) == nn.Conv2d:
#                         torch.quantization.fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)
#
#
# # import requests
# #
# # url = "https://s3.amazonaws.com/pytorch-tutorial-assets/imagenet_1k.zip"
# # filename = '~/Downloads/imagenet_1k_data.zip'
# #
# # r = requests.get(url)
# #
# # with open(filename, 'wb') as f:
# #     f.write(r.content)
#
# import torchvision
# import torchvision.transforms as transforms
#
# imagenet_dataset = torchvision.datasets.ImageNet(
#     '~/.data/imagenet',
#     split='train',
#     download=True,
#     transform=transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225]),
#     ]))
#
# def prepare_data_loaders(data_path):
#
#     traindir = os.path.join(data_path, 'train')
#     valdir = os.path.join(data_path, 'val')
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#
#     dataset = torchvision.datasets.ImageFolder(
#         traindir,
#         transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize,
#         ]))
#
#     dataset_test = torchvision.datasets.ImageFolder(
#         valdir,
#         transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             normalize,
#         ]))
#
#     train_sampler = torch.utils.data.RandomSampler(dataset)
#     test_sampler = torch.utils.data.SequentialSampler(dataset_test)
#
#     data_loader = torch.utils.data.DataLoader(
#         dataset, batch_size=train_batch_size,
#         sampler=train_sampler)
#
#     data_loader_test = torch.utils.data.DataLoader(
#         dataset_test, batch_size=eval_batch_size,
#         sampler=test_sampler)
#
#     return data_loader, data_loader_test
#
# data_path = 'data/imagenet_1k'
# saved_model_dir = 'data/'
# float_model_file = 'mobilenet_pretrained_float.pth'
# scripted_float_model_file = 'mobilenet_quantization_scripted.pth'
# scripted_quantized_model_file = 'mobilenet_quantization_scripted_quantized.pth'
#
# train_batch_size = 30
# eval_batch_size = 30
#
# data_loader, data_loader_test = prepare_data_loaders(data_path)
# criterion = nn.CrossEntropyLoss()
# float_model = torch.load(saved_model_dir + float_model_file).to('cpu')
#
# print('\n Inverted Residual Block: Before fusion \n\n', float_model.features[1].conv)
# float_model.eval()
#
# # Fuses modules
# float_model.fuse_model()
#
# # Note fusion of Conv+BN+Relu and Conv+Relu
# print('\n Inverted Residual Block: After fusion\n\n',float_model.features[1].conv)
#
# num_eval_batches = 10
#
# print("Size of baseline model")
# print_size_of_model(float_model)
#
# top1, top5 = evaluate(float_model, criterion, data_loader_test, neval_batches=num_eval_batches)
# print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
# torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)