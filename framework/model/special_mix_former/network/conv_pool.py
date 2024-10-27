# _*_ coding: utf-8 _*_

"""
    @Time : 2022/7/11 14:41 
    @Author : smile 笑
    @File : conv_mlp.py
    @desc :
"""


import torch
from torch import nn


class ConvPoolEmbedding(nn.Module):
    def __init__(self, input_dim, channel_dim, output_dim):
        super(ConvPoolEmbedding, self).__init__()

        self.conv_pool = nn.Sequential(
            nn.Conv2d(input_dim, channel_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),

            nn.Conv2d(channel_dim, output_dim, kernel_size=1, stride=1, padding=0),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.conv_pool(x)


class ConvPatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super(ConvPatchEmbedding, self).__init__()
        self.num_patches = int(img_size // patch_size) ** 2
        self.patch_size = (patch_size, patch_size)

        self.patch_conv = nn.Sequential(
            ConvPoolEmbedding(in_chans, 16, 32),
            ConvPoolEmbedding(32, 32, 64),
            ConvPoolEmbedding(64, 64, 128),
            ConvPoolEmbedding(128, 128, embed_dim),
        )

    def forward(self, x):
        return self.patch_conv(x).flatten(2).transpose(1, 2)


if __name__ == '__main__':
    x = torch.randn([4, 3, 224, 224])
    model = ConvPatchEmbedding()
    x = model(x)
    print(x.shape)
    print(sum(x.numel() for x in model.parameters()))  # 76736
    # torch.save(model.state_dict(), "1.pth")  # 24M


