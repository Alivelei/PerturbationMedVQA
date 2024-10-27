# _*_ coding: utf-8 _*_

"""
    @Time : 2022/2/28 16:37
    @Author : smile 笑
    @File : standard_model.py
    @desc : framework.model.mix_mixer.
"""


import torch
from torch import nn
from einops.layers.torch import Rearrange, Reduce
from framework.model.mix_mixer.network.mlp_mixer import MLPMixerBlock
from framework.model.mix_mixer.network.word_embedding import WordEmbedding
from framework.model.mix_mixer.hidden_mix.rand_replace_zero import RandReplaceZeroMix
from framework.model.mix_mixer.hidden_mix.rand_gaoss_mix import BalancedRandGaossMix, HardRandGaossMix, SoftRandGaossMix
from framework.model.mix_mixer.hidden_mix.content_replace_mix import RandContentReplaceMix


class QusEmbeddingMap(nn.Module):
    def __init__(self, glove_path, word_size, embedding_dim, hidden_size):
        super(QusEmbeddingMap, self).__init__()

        self.embedding = WordEmbedding(word_size, embedding_dim, 0.0, False)
        self.embedding.init_embedding(glove_path)

        self.linear = nn.Linear(embedding_dim, hidden_size)

    def forward(self, qus):
        text_embedding = self.embedding(qus)

        text_x = self.linear(text_embedding)

        return text_x


class MaxLinearPooling(nn.Module):
    def __init__(self, in_chans=3, out_channels=768, conv_f=32):
        super(MaxLinearPooling, self).__init__()

        self.max_pooling = nn.Sequential(
            nn.Conv2d(in_chans, conv_f, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(conv_f, conv_f, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(conv_f, conv_f, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(conv_f, out_channels, 3, 1, 1),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.max_pooling(x)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            # Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x):
        x = self.projection(x)

        return x


class ImageToTokens(nn.Module):
    def __init__(self, in_channels=3, out_channels=768, conv_kernel=7, conv_stride=4, conv_padding=4, pool_kernel=4, pool_stride=4):
        super(ImageToTokens, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, conv_kernel, conv_stride, conv_padding),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(pool_kernel, pool_stride)
        )

    def forward(self, x):
        img = self.conv(x)
        # b, c, _, _ = img.shape
        # img_to_token = img.view(b, c, -1).transpose(1, 2)

        return img


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


class MLPMixerVQASystem(nn.Module):
    def __init__(self, depth=12, emb_size=768, drop_out=.2, qus_embedding_dim=300, num_patch=216,
                 glove_path="../../../save/embedding/slake_qus_glove_emb_300d.npy", ans_size=223,
                 mixup_alpha=5, mixup_beta=1, prob=1, word_size=305, select_hidden_mix="rand_content_replace_mix"):
        super(MLPMixerVQASystem, self).__init__()

        token_dim, channel_dim = int(emb_size // 2), int(emb_size * 4)

        self.text_embedding_linear = QusEmbeddingMap(glove_path, word_size, qus_embedding_dim, emb_size)

        self.img_to_tokens = ConvPatchEmbedding(embed_dim=emb_size)

        self.img_mod_embed = nn.Parameter(torch.zeros(1, 1, emb_size), requires_grad=True)  # img modality cls
        self.qus_mod_embed = nn.Parameter(torch.zeros(1, 1, emb_size), requires_grad=True)  # qus modality cls

        if select_hidden_mix == "rand_replace_zero_mix":
            self.trans_mix = RandReplaceZeroMix(mixup_alpha=mixup_alpha, mixup_beta=mixup_beta, prob=prob)
        elif select_hidden_mix == "rand_content_replace_mix":
            self.trans_mix = RandContentReplaceMix(mixup_alpha=mixup_alpha, mixup_beta=mixup_beta, prob=prob)
        elif select_hidden_mix == "hard_rand_gauss_mix":
            self.trans_mix = HardRandGaossMix(embed_dim=emb_size, mixup_alpha=mixup_alpha, mixup_beta=mixup_beta, prob=prob)
        elif select_hidden_mix == "soft_rand_gauss_mix":
            self.trans_mix = SoftRandGaossMix(embed_dim=emb_size, mixup_alpha=mixup_alpha, mixup_beta=mixup_beta, prob=prob)
        elif select_hidden_mix == "bal_rand_gauss_mix":
            self.trans_mix = BalancedRandGaossMix(embed_dim=emb_size, mixup_alpha=mixup_alpha, mixup_beta=mixup_beta, prob=prob)

        self.mlp_mixer = nn.ModuleList([MLPMixerBlock(input_dim=emb_size, num_patch=num_patch, token_dim=token_dim,
                                           channel_dim=channel_dim, dropout=drop_out) for i in range(depth)])

        self.cls_forward = nn.Sequential(
            Reduce("b s c -> b c", reduction="mean"),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, emb_size),
            nn.GELU(),
            nn.Linear(emb_size, ans_size)
        )

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.img_mod_embed, std=.02)
        torch.nn.init.normal_(self.qus_mod_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, y, mode="test"):
        img_tokens = self.img_to_tokens(x)

        qus_embedding = self.text_embedding_linear(y)

        img_tokens = img_tokens + self.img_mod_embed
        qus_embedding = qus_embedding + self.qus_mod_embed

        x_len = img_tokens.size(1)
        z = torch.cat([img_tokens, qus_embedding], dim=1)

        for i, block in enumerate(self.mlp_mixer):
            if mode == "train":
                x, y = z[:, :x_len], z[:, x_len:]

                if i == 0:
                    lam = self.trans_mix(x, y)
                else:
                    self.trans_mix.step_mix_batch(x, y, lam)
                z[:, :x_len], z[:, x_len:] = x, y
            z = block(z)

        res = self.cls_forward(z)  # 使用单层输出

        return res


def posi_mlp_mixer_small(**kwargs):
    model = MLPMixerVQASystem(
        depth=8, emb_size=512, drop_out=.1, qus_embedding_dim=300, num_patch=216,
        glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], ans_size=kwargs["ans_size"],
        mixup_alpha=kwargs["mix_alpha_1"], mixup_beta=kwargs["mix_alpha_2"],
        prob=kwargs["mix_probability"], select_hidden_mix=kwargs["select_hidden_mix"],
    )
    return model


def posi_mlp_mixer_base(**kwargs):
    model = MLPMixerVQASystem(
        depth=12, emb_size=768, drop_out=.2, qus_embedding_dim=300, num_patch=216,
        glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], ans_size=kwargs["ans_size"],
        mixup_alpha=kwargs["mix_alpha_1"], mixup_beta=kwargs["mix_alpha_2"],
        prob=kwargs["mix_probability"], select_hidden_mix=kwargs["select_hidden_mix"],
    )
    return model


def posi_mlp_mixer_large(**kwargs):
    model = MLPMixerVQASystem(
        depth=24, emb_size=1024, drop_out=.3, qus_embedding_dim=300, num_patch=216,
        glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], ans_size=kwargs["ans_size"],
        mixup_alpha=kwargs["mix_alpha_1"], mixup_beta=kwargs["mix_alpha_2"],
        prob=kwargs["mix_probability"], select_hidden_mix=kwargs["select_hidden_mix"],
    )
    return model


def posi_mlp_mixer_huge(**kwargs):
    model = MLPMixerVQASystem(
        depth=32, emb_size=1280, drop_out=.0, qus_embedding_dim=300, num_patch=216,
        glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], ans_size=kwargs["ans_size"],
        mixup_alpha=kwargs["mix_alpha_1"], mixup_beta=kwargs["mix_alpha_2"],
        prob=kwargs["mix_probability"], select_hidden_mix=kwargs["select_hidden_mix"],
    )
    return model


if __name__ == '__main__':
    a = torch.randn([2, 3, 224, 224]).cuda()
    b = torch.randint(0, 20, [2, 20]).cuda()
    model = MLPMixerVQASystem().cuda()
    # torch.save(model.state_dict(), "1.pth")
    print(model(a, b).shape)
    print(sum(x.numel() for x in model.parameters()))  # 60095979

