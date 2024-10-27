# _*_ coding: utf-8 _*_

"""
    @Time : 2023/3/29 10:26 
    @Author : smile 笑
    @File : content_replace_mix.py
    @desc :
"""


import torch
import numpy as np
from torch import nn


def rand_qus_box(img_len, qus_len):
    cl = np.random.randint(img_len-qus_len)
    bbz1 = int(cl)
    bbz2 = int(cl+qus_len)

    return bbz1, bbz2


# 如果对文本内容全部与图像内容进行mixup是soft_mix
class RandContentReplaceMix(nn.Module):
    def __init__(self, mixup_alpha=5., mixup_beta=1., prob=1.0):
        super(RandContentReplaceMix, self).__init__()
        self.mixup_alpha = mixup_alpha
        self.mixup_beta = mixup_beta
        self.mix_prob = prob
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
        x_bbz1, x_bbz2 = rand_qus_box(x.size(1), y.size(1))
        x_flipped = y.mul(1. - lam)
        x[:, x_bbz1:x_bbz2].mul_(lam).add_(x_flipped)

        return lam

    def step_mix_batch(self, x, y, lam):
        # if np.random.rand() < 0.3:
        x_bbz1, x_bbz2 = rand_qus_box(x.size(1), y.size(1))
        x_flipped = y.mul(1. - lam)
        x[:, x_bbz1:x_bbz2].mul_(lam).add_(x_flipped)

    def __call__(self, x, y):
        lam = self.multi_mix_batch(x, y)  # tuple or value

        return lam



