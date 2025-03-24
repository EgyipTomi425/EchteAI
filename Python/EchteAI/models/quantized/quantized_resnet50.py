import torch
import torch.nn as nn
import torch.ao.nn.quantized as nnq
import torch.ao.quantization as quant
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from torch import Tensor
from collections import OrderedDict

class QuantizedBasicBlock(BasicBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_add = nnq.FloatFunctional()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.skip_add.add(out, identity)
        out = self.relu(out)

        return out


class QuantizedBottleneck(Bottleneck):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_add = nnq.FloatFunctional()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.skip_add.add(out, identity)
        out = self.relu(out)

        return out


class QuantizedResNet(ResNet):
    def __init__(self, block, layers, **kwargs):
        super().__init__(block, layers, **kwargs)
        del self.avgpool
        del self.fc
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        out0 = self.layer1(x)
        out1 = self.layer2(out0)
        out2 = self.layer3(out1)
        out3 = self.layer4(out2)

        out = OrderedDict()
        out["0"] = out0
        out["1"] = out1
        out["2"] = out2
        out["3"] = out3

        return out

    def forward(self, x):
        x = self.quant(x)
        x = self._forward_impl(x)
        for key in x:
            x[key] = self.dequant(x[key])
        return x


def quantized_resnet50():
    return QuantizedResNet(QuantizedBottleneck, [3, 4, 6, 3])
