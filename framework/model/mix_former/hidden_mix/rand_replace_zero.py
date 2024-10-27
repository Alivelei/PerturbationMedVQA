# _*_ coding: utf-8 _*_

"""
    @Time : 2023/3/29 10:27 
    @Author : smile 笑
    @File : rand_replace_zero.py
    @desc :
"""


import torch
import numpy as np
from torch import nn


def rand_qus_box(size, lam):
    L = size[1]

    cut_l = np.int(L * (1. - lam))
    cl = np.random.randint(L)
    bbz1 = np.clip(cl - cut_l // 2, 0, L)
    bbz2 = np.clip(cl + cut_l // 2, 0, L)

    return bbz1, bbz2


class RandReplaceZeroMix(nn.Module):
    def __init__(self, mixup_alpha=5., mixup_beta=1., prob=1.0):
        super(RandReplaceZeroMix, self).__init__()
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
        x_bbz1, x_bbz2 = rand_qus_box(x.size(), lam)

        x[:, x_bbz1:x_bbz2] = 0

        y_bbz1, y_bbz2 = rand_qus_box(y.size(), lam)

        y[:, y_bbz1:y_bbz2] = 0

        return lam

    def step_mix_batch(self, x, y, lam):
        # if np.random.rand() < 0.3:
        x_bbz1, x_bbz2 = rand_qus_box(x.size(), lam)

        x[:, x_bbz1:x_bbz2] = 0

        y_bbz1, y_bbz2 = rand_qus_box(y.size(), lam)

        y[:, y_bbz1:y_bbz2] = 0

    def __call__(self, x, y):
        lam = self.multi_mix_batch(x, y)  # tuple or value

        return lam



