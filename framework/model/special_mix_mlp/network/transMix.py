# _*_ coding: utf-8 _*_

"""
    @Time : 2023/2/19 10:42 
    @Author : smile 笑
    @File : transMix.py
    @desc :
"""


import torch
from timm.data.mixup import Mixup, one_hot
import numpy as np
from torch import nn


def mixup_target(target, num_classes, lam=1., smoothing=0.0, device='cuda', return_y1y2=False):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value, device=device)  # flip翻转

    if return_y1y2:
        return y1 * lam + y2 * (1. - lam), y1.clone(), y2.clone()
    else:
        return y1 * lam + y2 * (1. - lam)


class Mixup_transmix(nn.Module):
    def __init__(self, mixup_alpha=5., mixup_beta=1., prob=1.0, label_smoothing=0.1, num_classes=1000):
        super(Mixup_transmix, self).__init__()
        self.mixup_alpha = mixup_alpha
        self.mixup_beta = mixup_beta
        self.mix_prob = prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop)

    def self_params_per_batch(self):
        lam = 1.
        if self.mixup_enabled and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_beta)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = float(lam_mix)
        return lam

    def multi_mix_batch(self, x, y):
        lam = self.self_params_per_batch()

        if lam == 1.:
            return 1.

        x_flipped = x.flip(0).mul_(1. - lam)
        x.mul_(lam).add_(x_flipped)

        y_flipped = y.flip(0).mul_(1. - lam)
        y.mul_(lam).add_(y_flipped)

        return lam

    def step_mix_batch(self, x, y, lam):
        if np.random.rand() < 0.3:
            x_flipped = x.flip(0).mul_(1. - lam)
            x.mul_(lam).add_(x_flipped)

            y_flipped = y.flip(0).mul_(1. - lam)
            y.mul_(lam).add_(y_flipped)

    def __call__(self, x, y, target):
        assert len(x) % 2 == 0, 'Batch size should be even when using this'
        lam = self.multi_mix_batch(x, y)  # tuple or value

        mixed_target, y1, y2 = mixup_target(target, self.num_classes, lam, self.label_smoothing, x.device,
                                            return_y1y2=True)  # tuple or tensor

        return mixed_target, lam


if __name__ == '__main__':
    from copy import deepcopy
    model = Mixup_transmix()
    a = torch.randn([4, 16, 768])
    a2 = deepcopy(a)
    t = torch.tensor([4, 2, 12, 23])
    b = torch.randn([4, 1000])
    x, m, lam = model(a, t)
    y = torch.randn([4, 10, 768])
    y2 = deepcopy(y)

    # print(y)
    # print(y2)

    import torch.nn.functional as F
    from torch import nn

    # kl = F.kl_div(m.softmax(dim=-1).log(), b.softmax(dim=-1), reduction='sum')  # 第一个预测分布，第二个真实分布
    # cr = nn.BCEWithLogitsLoss()(b, m)
    #
    # print(x.shape)
    # print(m.shape)
    # print(kl)
    # print(cr)


