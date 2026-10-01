"""A custom network that is not a torchvision architecture: SiLU stem, depthwise-separable blocks with
squeeze-and-excitation and Hardswish, a ReLU residual bottleneck, one convolution without BN, a GroupNorm
layer and a linear head."""
import torch
import torch.nn as nn


class SE(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.fc1, self.fc2 = nn.Conv2d(c, c // r, 1), nn.Conv2d(c // r, c, 1)
        self.act, self.gate = nn.SiLU(), nn.Sigmoid()

    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)
        return x * self.gate(self.fc2(self.act(self.fc1(s))))


class DWBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False)
        self.bn1, self.act1 = nn.BatchNorm2d(cin), nn.Hardswish()
        self.se = SE(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn2(self.pw(self.se(self.act1(self.bn1(self.dw(x))))))


class Bottleneck(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1, self.bn1, self.relu1 = nn.Conv2d(c, c // 2, 1, bias=False), nn.BatchNorm2d(c // 2), nn.ReLU()
        self.conv2, self.bn2, self.relu2 = nn.Conv2d(c // 2, c // 2, 3, 1, 1, bias=False), nn.BatchNorm2d(c // 2), nn.ReLU()
        self.conv3, self.bn3 = nn.Conv2d(c // 2, c, 1, bias=False), nn.BatchNorm2d(c)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu1(self.bn1(self.conv1(x)))
        y = self.relu2(self.bn2(self.conv2(y)))
        return self.relu(x + self.bn3(self.conv3(y)))


class MyNet(nn.Module):
    def __init__(self, n_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.SiLU())
        self.b1 = DWBlock(32, 64, 2)
        self.r1 = Bottleneck(64)
        self.b2 = DWBlock(64, 128, 2)
        self.r2 = Bottleneck(128)
        self.b3 = DWBlock(128, 256, 2)
        self.plain = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU())        # no BN: invisible to the model
        self.gn = nn.Sequential(nn.Conv2d(256, 384, 1, bias=False), nn.GroupNorm(8, 384), nn.SiLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(384, n_classes)

    def forward(self, x):
        x = self.b3(self.r2(self.b2(self.r1(self.b1(self.stem(x))))))
        return self.fc(self.pool(self.gn(self.plain(x))).flatten(1))
