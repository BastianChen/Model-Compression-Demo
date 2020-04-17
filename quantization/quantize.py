# @Time : 2020-04-17 10:43 
# @Author : Ben 
# @Version：V 0.1
# @File : quantization.py
# @desc :

import torch
from quantization.nets import MyNet
import os


class Quantize:
    def __init__(self, net_path):
        self.device = "cpu" if torch.cuda.is_available() else "cpu"
        self.net = MyNet().to(self.device)
        self.net.load_state_dict(torch.load(net_path, map_location=self.device))
        self.net.eval()
        self.net.fuse_model()

    def quantize(self):
        # example = torch.Tensor(1, 1, 28, 28).to(self.device)
        # torch.jit.save(torch.jit.trace(self.net, example), "models/net_fuse.pth")
        torch.jit.save(torch.jit.script(self.net), "models/net_fuse.pth")
        print('Size (MB):', os.path.getsize("models/net_fuse.pth") / 1e6)

        self.net.qconfig = torch.quantization.default_qconfig
        # print(model.qconfig)
        # only cpu
        torch.quantization.prepare(self.net, inplace=True)

        # Convert to quantized model
        torch.quantization.convert(self.net, inplace=True)
        torch.jit.save(torch.jit.script(self.net), "models/net_convert.pth")
        # torch.jit.save(torch.jit.trace(self.net, example), "models/net_convert.pth")
        print('Size (MB):', os.path.getsize("models/net_convert.pth") / 1e6)

if __name__ == '__main__':
    quantize = Quantize("models/net.pth")
    quantize.quantize()
