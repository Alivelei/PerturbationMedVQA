# _*_ coding: utf-8 _*_

"""
    @Time : 2023/2/7 8:55 
    @Author : smile 笑
    @File : __init__.py
    @desc :
"""


from .model_interface import ModelInterfaceModule


def get_model_module(model_name):
    if model_name == "general_former_small":
        from .model.mix_former.model import general_former_small
        return general_former_small
    if model_name == "trans_former_small":
        from .model.mix_former.trans_model import trans_former_small
        return trans_former_small
    if model_name == "posi_former_small":
        from .model.mix_former.posi_model import posi_former_small
        return posi_former_small

    if model_name == "general_former_base":
        from .model.mix_former.model import general_former_base
        return general_former_base
    if model_name == "trans_former_base":
        from .model.mix_former.trans_model import trans_former_base
        return trans_former_base
    if model_name == "posi_former_base":
        from .model.mix_former.posi_model import posi_former_base
        return posi_former_base

    if model_name == "general_former_large":
        from .model.mix_former.model import general_former_large
        return general_former_large
    if model_name == "trans_former_large":
        from .model.mix_former.trans_model import trans_former_large
        return trans_former_large
    if model_name == "posi_former_large":
        from .model.mix_former.posi_model import posi_former_large
        return posi_former_large

    if model_name == "general_mlp_struc_small":
        from .model.mix_mlp.model import general_mlp_struc_small
        return general_mlp_struc_small
    if model_name == "trans_mlp_struc_small":
        from .model.mix_mlp.trans_model import trans_mlp_struc_small
        return trans_mlp_struc_small
    if model_name == "posi_mlp_struc_small":
        from .model.mix_mlp.posi_model import posi_mlp_struc_small
        return posi_mlp_struc_small

    if model_name == "general_mlp_struc_base":
        from .model.mix_mlp.model import general_mlp_struc_base
        return general_mlp_struc_base
    if model_name == "trans_mlp_struc_base":
        from .model.mix_mlp.trans_model import trans_mlp_struc_base
        return trans_mlp_struc_base
    if model_name == "posi_mlp_struc_base":
        from .model.mix_mlp.posi_model import posi_mlp_struc_base
        return posi_mlp_struc_base

    if model_name == "general_mlp_struc_large":
        from .model.mix_mlp.model import general_mlp_struc_large
        return general_mlp_struc_large
    if model_name == "trans_mlp_struc_large":
        from .model.mix_mlp.trans_model import trans_mlp_struc_large
        return trans_mlp_struc_large
    if model_name == "posi_mlp_struc_large":
        from .model.mix_mlp.posi_model import posi_mlp_struc_large
        return posi_mlp_struc_large

    if model_name == "general_mlp_mixer_base":
        from .model.mix_mixer.model import general_mlp_mixer_base
        return general_mlp_mixer_base
    if model_name == "trans_mlp_mixer_base":
        from .model.mix_mixer.trans_model import trans_mlp_mixer_base
        return trans_mlp_mixer_base
    if model_name == "posi_mlp_mixer_base":
        from .model.mix_mixer.posi_model import posi_mlp_mixer_base
        return posi_mlp_mixer_base

    if model_name == "general_lin_former_base":
        from .model.mix_linformer.model import general_lin_former_base
        return general_lin_former_base
    if model_name == "trans_lin_former_base":
        from .model.mix_linformer.trans_model import trans_lin_former_base
        return trans_lin_former_base
    if model_name == "posi_lin_former_base":
        from .model.mix_linformer.posi_model import posi_lin_former_base
        return posi_lin_former_base

    if model_name == "special_trans_former_base":
        from .model.special_mix_former.trans_model import special_trans_former_base
        return special_trans_former_base
    if model_name == "special_posi_former_base":
        from .model.special_mix_former.posi_model import special_posi_former_base
        return special_posi_former_base

    if model_name == "special_trans_mlp_struc_base":
        from .model.special_mix_mlp.trans_model import special_trans_mlp_struc_base
        return special_trans_mlp_struc_base
    if model_name == "special_posi_mlp_struc_base":
        from .model.special_mix_mlp.posi_model import special_posi_mlp_struc_base
        return special_posi_mlp_struc_base


