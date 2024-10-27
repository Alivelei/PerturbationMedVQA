# _*_ coding: utf-8 _*_

"""
    @Time : 2022/6/25 10:55
    @Author : smile 笑
    @File : model.py
    @desc : framework.model.special_mix_former.
"""


import torch
import torch.nn as nn
from einops.layers.torch import Reduce
import numpy as np
from timm.models.vision_transformer import Block
from framework.model.special_mix_former.network.conv_pool import ConvPatchEmbedding
from framework.model.special_mix_former.network.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed
from framework.model.special_mix_former.network.word_embedding import WordEmbedding
from framework.model.special_mix_former.hidden_mix.trans_mix import BalancedTransMix, SoftTransMix, HardTransMix


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


class BertM3AEVQAModel(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, depth=8, num_heads=12, ans_size=223,
                 mlp_ratio=4, glove_path="../../../save/embedding/all_word_glove_emb_300d.npy",
                 word_size=14745, qus_embedding_dim=300, qus_seq_len=20, mixup_alpha=5, mixup_beta=1, prob=1,
                 norm_layer=nn.LayerNorm, select_hidden_mix="hard_trans_mix",
                 select_layer_mix=(1, 4, 7, 10), random_layer_mix=None):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = ConvPatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.img_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.img_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)  # img modality cls
        self.img_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)

        self.qus_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.qus_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)  # qus modality cls
        self.qus_pos_embed = nn.Parameter(torch.zeros(1, qus_seq_len + 1, embed_dim), requires_grad=False)
        self.qus_seq_len = qus_seq_len

        self.qus_embedding = QusEmbeddingMap(glove_path, word_size, qus_embedding_dim, embed_dim)  # 4645900

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])

        if select_hidden_mix == "soft_trans_mix":
            self.trans_mix = SoftTransMix(mixup_alpha=mixup_alpha, mixup_beta=mixup_beta, prob=prob, num_classes=ans_size)
        elif select_hidden_mix == "hard_trans_mix":
            self.trans_mix = HardTransMix(mixup_alpha=mixup_alpha, mixup_beta=mixup_beta, prob=prob, num_classes=ans_size)
        elif select_hidden_mix == "bal_trans_mix":
            self.trans_mix = BalancedTransMix(mixup_alpha=mixup_alpha, mixup_beta=mixup_beta, prob=prob, num_classes=ans_size)

        self.select_layer_mix = select_layer_mix
        self.random_layer_mix = random_layer_mix

        self.cls_linear = nn.Sequential(
            Reduce("b s c -> b c", reduction="mean"),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, ans_size)
        )

        # initialize x and y
        self.initialize_weights_x()
        self.initialize_weights_y()

    def initialize_weights_x(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding  (197, 768)
        img_pos_embed = get_2d_sincos_pos_embed(self.img_pos_embed.shape[-1], int(self.num_patches ** .5),
                                                cls_token=True)

        self.img_pos_embed.data.copy_(torch.from_numpy(img_pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.img_cls_token, std=.02)
        torch.nn.init.normal_(self.img_mod_embed, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def initialize_weights_y(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding  (20 768)
        qus_pos_embed = get_1d_sincos_pos_embed(self.qus_pos_embed.shape[-1], int(self.qus_seq_len),
                                                cls_token=True)

        self.qus_pos_embed.data.copy_(torch.from_numpy(qus_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.qus_cls_token, std=.02)
        torch.nn.init.normal_(self.qus_mod_embed, std=.02)

        # initialize nn.Linear and nn.LayerNorm
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

    def forward(self, x, y, label=None, mode="test"):
        # embed patches
        x = self.patch_embed(x)

        # embedding to qus
        y = self.qus_embedding(y)

        # add pos embed w/o cls token
        x = x + self.img_pos_embed[:, 1:, :]

        # append cls token
        img_cls_token = self.img_cls_token + self.img_pos_embed[:, :1, :]
        img_cls_tokens = img_cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((img_cls_tokens, x), dim=1)

        # add pos embed for question
        y = y + self.qus_pos_embed[:, 1:, :]
        # append cls token
        qus_cls_token = self.qus_cls_token + self.qus_pos_embed[:, :1, :]

        qus_cls_tokens = qus_cls_token.expand(y.shape[0], -1, -1)
        y = torch.cat((qus_cls_tokens, y), dim=1)

        x = x + self.img_mod_embed
        y = y + self.qus_mod_embed

        # concat img embedding and qus embedding in token dim
        z = torch.cat([x, y], dim=1)
        x_len = x.size(1)

        # apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            if mode == "train":
                x, y = z[:, 1:x_len], z[:, x_len + 1:]

                if i == 0:
                    mix_label, lam = self.trans_mix(x, y, label)
                elif self.select_layer_mix is not None:
                    if i + 1 in self.select_layer_mix:
                        self.trans_mix.step_mix_batch(x, y, lam)
                elif self.random_layer_mix is not None:
                    if np.random.rand() < self.random_layer_mix:
                        self.trans_mix.step_mix_batch(x, y, lam)
                z[:, 1:x_len], z[:, x_len + 1:] = x, y

            z = blk(z)

        res = self.cls_linear(z)

        if mode == "train":
            return res, mix_label
        else:
            return res


def special_trans_former_small(**kwargs):
    model = BertM3AEVQAModel(
        img_size=224, patch_size=16, in_chans=3, embed_dim=512, depth=6, num_heads=8, mlp_ratio=4,
        glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], ans_size=kwargs["ans_size"],
        qus_embedding_dim=300, qus_seq_len=20, mixup_alpha=kwargs["mix_alpha_1"], mixup_beta=kwargs["mix_alpha_2"],
        prob=kwargs["mix_probability"], select_hidden_mix=kwargs["select_hidden_mix"],
        select_layer_mix=kwargs["select_layer_mix"], random_layer_mix=kwargs["random_layer_mix"]
    )
    return model


def special_trans_former_base(**kwargs):
    model = BertM3AEVQAModel(
        img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], ans_size=kwargs["ans_size"],
        qus_embedding_dim=300, qus_seq_len=20, mixup_alpha=kwargs["mix_alpha_1"], mixup_beta=kwargs["mix_alpha_2"],
        prob=kwargs["mix_probability"], select_hidden_mix=kwargs["select_hidden_mix"],
        select_layer_mix=kwargs["select_layer_mix"], random_layer_mix=kwargs["random_layer_mix"]
    )
    return model


def special_trans_former_large(**kwargs):
    model = BertM3AEVQAModel(
        img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=16,
        glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], ans_size=kwargs["ans_size"],
        qus_embedding_dim=300, qus_seq_len=20, mixup_alpha=kwargs["mix_alpha_1"], mixup_beta=kwargs["mix_alpha_2"],
        prob=kwargs["mix_probability"], select_hidden_mix=kwargs["select_hidden_mix"],
        select_layer_mix=kwargs["select_layer_mix"], random_layer_mix=kwargs["random_layer_mix"]
    )
    return model


def special_trans_former_huge(**kwargs):
    model = BertM3AEVQAModel(
        img_size=224, patch_size=16, in_chans=3, embed_dim=1280, depth=36, num_heads=16, mlp_ratio=4,
        glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], ans_size=kwargs["ans_size"],
        qus_embedding_dim=300, qus_seq_len=20, mixup_alpha=kwargs["mix_alpha_1"], mixup_beta=kwargs["mix_alpha_2"],
        prob=kwargs["mix_probability"], select_hidden_mix=kwargs["select_hidden_mix"],
        select_layer_mix=kwargs["select_layer_mix"], random_layer_mix=kwargs["random_layer_mix"]
    )
    return model


if __name__ == '__main__':
    a = torch.randn([2, 3, 224, 224]).cuda()
    b = torch.ones([2, 20], dtype=torch.int64).cuda()
    l = torch.tensor([4, 2]).cuda()

    model = BertM3AEVQAModel().cuda()

    res, m_l = model(a, b, l, "train")
    print(res.shape, m_l.shape)
    print(sum(x.numel() for x in model.parameters()))  # 62596011
    # torch.save(model.state_dict(), "1.pth")  # 67M

    # nn.BinaryCrossEntropy()
    ls = nn.BCEWithLogitsLoss(reduction="mean")
    print(ls(res, m_l))

    # kl = nn.KLDivLoss(reduction="sum")(res.softmax(dim=-1).log(), m_l.softmax(dim=-1))
    # print(kl)
    # import torch.nn.functional as F
    # kl = F.kl_div(res.softmax(dim=-1).log(), m_l.softmax(dim=-1), reduction='sum')  # 第一个预测分布，第二个真实分布
    # print(kl)
    #
    mse = nn.MSELoss()
    print(mse(res, m_l))



