# _*_ coding: utf-8 _*_

"""
    @Time : 2022/6/25 10:55
    @Author : smile 笑
    @File : model.py
    @desc : framework.model.mix_former.
"""


import torch
import torch.nn as nn
from einops.layers.torch import Reduce
from timm.models.vision_transformer import Block
from framework.model.mix_former.network.conv_pool import ConvPatchEmbedding
from framework.model.mix_former.network.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed
from framework.model.mix_former.network.word_embedding import WordEmbedding


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
                 embed_dim=768, depth=12, num_heads=12, ans_size=223,
                 mlp_ratio=4, glove_path="../../../save/embedding/all_word_glove_emb_300d.npy",
                 word_size=14745, qus_embedding_dim=300, qus_seq_len=20,
                 norm_layer=nn.LayerNorm):
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

    def forward(self, x, y):
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

        # apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            z = blk(z)

        res = self.cls_linear(z)

        return res


def general_former_small(**kwargs):
    model = BertM3AEVQAModel(
        img_size=224, patch_size=16, in_chans=3, embed_dim=512, depth=4, num_heads=8, mlp_ratio=4,
        glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], ans_size=kwargs["ans_size"],
        qus_embedding_dim=300, qus_seq_len=20,
    )
    return model


def general_former_base(**kwargs):
    model = BertM3AEVQAModel(
        img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], ans_size=kwargs["ans_size"],
        qus_embedding_dim=300, qus_seq_len=20,
    )
    return model


def general_former_large(**kwargs):
    model = BertM3AEVQAModel(
        img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], ans_size=kwargs["ans_size"],
        qus_embedding_dim=300, qus_seq_len=20,
    )
    return model


def general_former_huge(**kwargs):
    model = BertM3AEVQAModel(
        img_size=224, patch_size=16, in_chans=3, embed_dim=1280, depth=36, num_heads=16, mlp_ratio=4,
        glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], ans_size=kwargs["ans_size"],
        qus_embedding_dim=300, qus_seq_len=20,
    )
    return model


if __name__ == '__main__':
    a = torch.randn([2, 3, 224, 224])
    b = torch.ones([2, 20], dtype=torch.int64)
    model = BertM3AEVQAModel()

    import time
    t1 = time.time()
    res = model(a, b)
    t2 = time.time()
    print(t2 - t1)
    print(res.shape)  # 0.54520
    print(sum(x.numel() for x in model.parameters()))  # 62596011



