# file: src/modeling/backbones/yolo11_backbone.py
import math, torch, torch.nn as nn, torch.nn.functional as F
from ...core import register

def autopad(k, p=None):
    return (k // 2) if p is None else p

def make_divisible(v, d=8):
    return int(math.ceil(v / d) * d)

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act='silu'):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act == 'silu' else nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class C2f(nn.Module):
    """YOLOv8/11 style C2f: split→stack bottlenecks→concat+1x1"""
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, act='silu'):
        super().__init__()
        c_ = make_divisible(int(c2 * e))
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c1, c_, 1, 1, act=act)
        m = []
        for _ in range(n):
            m += [nn.Sequential(Conv(c_, c_, 3, 1, act=act),
                                Conv(c_, c_, 3, 1, act=act))]
        self.m = nn.Sequential(*m)
        self.cv3 = Conv(2 * c_, c2, 1, 1, act=act)
        self.shortcut = shortcut

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        y1 = self.m(y1)
        y = torch.cat([y1, y2], dim=1)
        return self.cv3(y)

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5, act='silu'):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c_ * 4, c2, 1, 1, act=act)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

@register()
class YOLO11Backbone(nn.Module):
    __share__ = ['width_multi', 'depth_multi']
    def __init__(self,
                 in_channels=3,
                 width_multi=1.0,
                 depth_multi=1.0,
                 act='silu',
                 return_idx=[2, 3, 4]):  # -> P3,P4,P5 (stride 8/16/32)
        super().__init__()
        w = width_multi; d = depth_multi
        # base channels per stage (stem, p2, p3, p4, p5)
        base = [64, 128, 256, 512, 768]  # you can tweak last to 1024 for larger models
        ch = [make_divisible(c * w, 8) for c in base]
        # depth per stage
        depths = [1, 2, 3, 1]  # p2..p5 C2f repeat; scale by d
        depths = [max(1, round(x * d)) for x in depths]

        self.stem = Conv(in_channels, ch[0], 3, 2, act=act)     # stride=2 (P1)
        # P2
        self.p2_down = Conv(ch[0], ch[1], 3, 2, act=act)        # stride=4
        self.p2_c2f = C2f(ch[1], ch[1], n=depths[0], act=act)
        # P3
        self.p3_down = Conv(ch[1], ch[2], 3, 2, act=act)        # stride=8
        self.p3_c2f = C2f(ch[2], ch[2], n=depths[1], act=act)
        # P4
        self.p4_down = Conv(ch[2], ch[3], 3, 2, act=act)        # stride=16
        self.p4_c2f = C2f(ch[3], ch[3], n=depths[2], act=act)
        # P5
        self.p5_down = Conv(ch[3], ch[4], 3, 2, act=act)        # stride=32
        self.p5_c2f = C2f(ch[4], ch[4], n=depths[3], act=act)
        self.head_sppf = SPPF(ch[4], ch[4], k=5, act=act)

        self.return_idx = return_idx
        self.out_channels = [ch[2], ch[3], ch[4]]      # P3,P4,P5
        self.strides = [8, 16, 32]

    def forward(self, x):
        outs = []
        x = self.stem(x)                  # P1
        x = self.p2_down(x); x = self.p2_c2f(x)  # P2
        p3 = self.p3_c2f(self.p3_down(x))        # P3
        p4 = self.p4_c2f(self.p4_down(p3))       # P4
        p5 = self.p5_c2f(self.p5_down(p4))       # P5
        p5 = self.head_sppf(p5)

        # idx 기준 저장(여기선 고정적으로 2:P3, 3:P4, 4:P5로 가정)
        feats = [None, None, p3, p4, p5]
        return [feats[i] for i in self.return_idx]
