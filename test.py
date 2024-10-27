# _*_ coding: utf-8 _*_

"""
    @Time : 2023/2/4 11:25 
    @Author : smile 笑
    @File : test.py
    @desc :
"""


import argparse
from data import DataInterfaceModule, SlakeDatasetModule, RadDatasetModule, PathVQADatasetModule, OVQADatasetModule
from framework import ModelInterfaceModule, get_model_module
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything
import os


def mkdir_println(dir_path, println):
    if os.path.exists(dir_path):
        print(println + "文件夹已创建.")
    else:
        os.mkdir(dir_path)
        print(println + "文件夹创建成功.")


def create_model_module(args):
    model_name = args.model_select + "_" + args.model_size
    model_func = get_model_module(model_name)

    model = ModelInterfaceModule.load_from_checkpoint(args.test_best_model_path, model=model_func, args=args, trict=False)
    print("test_best_model load success!")

    args.default_root_dir = os.path.join(args.default_root_dir, model_name + "/")

    return model, args


def dataset_select(args):
    if args.select_data == "slake":
        db = DataInterfaceModule(SlakeDatasetModule, args)
    if args.select_data == "rad":
        db = DataInterfaceModule(RadDatasetModule, args)
    if args.select_data == "path_vqa":
        db = DataInterfaceModule(PathVQADatasetModule, args)
    if args.select_data == "ovqa":
        db = DataInterfaceModule(OVQADatasetModule, args)

    # 用来获取是哪个版本的模型
    logger = TensorBoardLogger(
        save_dir=args.default_root_dir,
        version=args.model_select + "_" + args.model_size + "_" + args.select_data + "_" + str(args.version),
        name="train_logs"
    )

    return db, logger


def main(args):
    seed_everything(args.random_seed, True)  # 设置随机数种子

    model, args = create_model_module(args)
    db, logger = dataset_select(args)

    # 构建json保存路径
    epoch_effect_path = os.path.join(args.train_epoch_effect_path, str(logger.version))
    mkdir_println(epoch_effect_path, "model_param_version")  # 创建param下的version文件夹
    args.test_epoch_effect_path = os.path.join(epoch_effect_path, "test_epoch_effect.json")

    trainer = Trainer(
        gpus=args.device_ids,
        strategy="ddp",
        logger=logger,
        default_root_dir=args.default_root_dir,
        resume_from_checkpoint=args.resume_from_checkpoint if os.path.exists(args.resume_from_checkpoint) else None,
    )

    trainer.fit(model, db)  # ckpt_path=resume_from_checkpoint

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default="train")
    parser.add_argument("--model_select", default="posi_former", choices=["general_former", "trans_former", "posi_former",
                                                                          "general_mlp_struc", "trans_mlp_struc", "posi_mlp_struc",
                                                                          "general_mlp_mixer", "trans_mlp_mixer", "posi_mlp_mixer",
                                                                          "general_lin_former", "trans_lin_former", "posi_lin_former",
                                                                          "special_trans_former", "special_posi_former",
                                                                          "special_trans_mlp_struc", "special_posi_mlp_struc"])
    parser.add_argument("--model_size", default="base", choices=["small", "base", "large", "huge"])
    parser.add_argument("--select_data", default="slake", choices=["slake", "rad", "path_vqa", "ovqa"])
    parser.add_argument("--select_mix", default="cut_img_qus_mixup", choices=["cut_img_qus_mixup", "img_qus_mixup"])
    parser.add_argument("--mix_flag", default=None, choices=["trans_mix", "posi_mix", "special_trans_mix", "special_posi_mix", None], help="mixup Flag.")
    parser.add_argument("--random_layer_mix", default=None, choices=[0.25, 0.5, 0.75])
    parser.add_argument("--select_layer_mix", default=None, choices=[(1, 4, 7, 10), (1, 3, 5, 7, 9, 11)])  # mlp自行增加
    parser.add_argument("--train_data_aug", default=False, choices=[True, False])
    parser.add_argument("--select_hidden_mix", default="soft_trans_mix", choices=[
        "hard_trans_mix", "soft_trans_mix", "bal_trans_mix",  # 自己设计的mix  trans_mix
        "hard_rand_gauss_mix", "soft_rand_gauss_mix", "bal_rand_gauss_mix",  # 随机加入噪声  posi_mix
        "rand_content_replace_mix",  # 文本到图像内容替换  posi_mix
        "rand_replace_zero_mix",  # 随机置为0  posi_mix
    ])
    parser.add_argument("--trans_mix_model_loss", default="bce", choices=["bce", "kl", "mse", "bce_mse"])
    parser.add_argument("--load_pre", default=False, choices=[True, False])
    parser.add_argument("--use_pretrain", default=False, choices=[True, False], help="control dataset choice")
    parser.add_argument("--version", default="lam_5_1")

    # configure
    parser.add_argument("--batch_size", default=48, type=int)
    parser.add_argument("--device_ids", default=[0, 1])
    parser.add_argument("--num_workers", default=4, type=int)

    # constant_image
    parser.add_argument("--img_rotation", default=15, type=int)
    parser.add_argument("--resized_crop_left", default=0.6, type=float)
    parser.add_argument("--resized_crop_right", default=1.0, type=float)
    parser.add_argument("--blur", default=[0.1, 2.0])
    parser.add_argument("--b_size", default=[5, 5])
    parser.add_argument("--blur_p", default=0.5, type=float)
    parser.add_argument("--apply_p", default=0.8, type=float)
    parser.add_argument("--img_flip", default=0.5, type=float)
    parser.add_argument("--brightness", default=0.4, type=float)
    parser.add_argument("--contrast", default=0.4, type=float)
    parser.add_argument("--saturation", default=0.4, type=float)
    parser.add_argument("--hue", default=0.4, type=float)
    parser.add_argument("--grayscale", default=0.2, type=float)

    # rand aug image
    parser.add_argument("--general_rand_aug", default=False)
    parser.add_argument("--resized_crop_scale_left", default=0.6, type=float)
    parser.add_argument("--resized_crop_scale_right", default=1, type=float)
    parser.add_argument("--ra_n", default=2)
    parser.add_argument("--ra_m", default=12)
    parser.add_argument("--img_jitter", default=0.2, type=float)
    parser.add_argument("--reprob", default=0.2)

    # configure
    parser.add_argument("--epochs", default=15000, type=int)
    parser.add_argument("--qus_seq_len", default=20, type=int)
    parser.add_argument("--answer_open", default=0, type=int)
    parser.add_argument("--answer_close", default=1, type=int)
    parser.add_argument("--train_epoch_effect_path", default="param")
    parser.add_argument("--test_epoch_effect_path", default="param")

    parser.add_argument("--best_model_path", default="best_model")
    parser.add_argument("--test_best_model_path", default="test_best_model")
    parser.add_argument("--default_root_dir", default="./save/")
    parser.add_argument("--resume_from_checkpoint", default="./save/model/best_model/last.ckpt")
    parser.add_argument("--pre_best_model_path", default="./save/model/pre_best_model/0/train_loss=0.2651.ckpt")

    # model
    parser.add_argument("--random_seed", default=1024, type=int)

    # slake dataset
    parser.add_argument("--slake_qus_ws_path", default="./save/ws/slake_qus_ws.pkl")
    parser.add_argument("--slake_ans_ws_path", default="./save/ws/slake_ans_ws.pkl")
    parser.add_argument("--slake_qus_glove_path", default="./save/embedding/slake_qus_glove_emb_300d.npy")
    parser.add_argument("--slake_qus_word_size", default=305, type=int)
    parser.add_argument("--slake_ans_word_size", default=222, type=int)
    parser.add_argument("--slake_train_dataset_path", default="./data/ref/Slake1.0/train.json")
    parser.add_argument("--slake_test_dataset_path", default="./data/ref/Slake1.0/test.json")
    parser.add_argument("--slake_dataset_xm_path", default="./data/ref/Slake1.0/imgs/xmlab")

    # rad dataset
    parser.add_argument("--rad_qus_word_size", default=1231, type=int)
    parser.add_argument("--rad_ans_word_size", default=475, type=int)
    parser.add_argument("--rad_qus_ws_path", default="./save/ws/rad_qus_ws.pkl")
    parser.add_argument("--rad_ans_ws_path", default="./save/ws/rad_ans_ws.pkl")
    parser.add_argument("--rad_qus_glove_path", default="./save/embedding/rad_qus_glove_emb_300d.npy")
    parser.add_argument("--rad_images_path", default="./data/ref/rad/images")
    parser.add_argument("--rad_train_dataset_path", default="./data/ref/rad/trainset.json")
    parser.add_argument("--rad_test_dataset_path", default="./data/ref/rad/testset.json")

    # path_vqa dataset
    parser.add_argument("--path_vqa_qus_word_size", default=4631, type=int)
    parser.add_argument("--path_vqa_ans_word_size", default=4092, type=int)
    parser.add_argument("--path_vqa_qus_ws_path", default="./save/ws/path_vqa_qus_ws.pkl")
    parser.add_argument("--path_vqa_ans_ws_path", default="./save/ws/path_vqa_ans_ws.pkl")
    parser.add_argument("--path_qus_glove_path", default="./save/embedding/path_vqa_qus_glove_300d.npy")
    parser.add_argument("--path_train_img_folder_path", default="./data/ref/PathVQA/images/train")
    parser.add_argument("--path_test_img_folder_path", default="./data/ref/PathVQA/images/test")
    parser.add_argument("--path_train_dataset_text_path", default="./data/ref/PathVQA/qas/train/train_qa.pkl")
    parser.add_argument("--path_test_dataset_text_path", default="./data/ref/PathVQA/qas/test/test_qa.pkl")

    # ovqa dataset
    parser.add_argument("--ovqa_qus_word_size", default=969, type=int)
    parser.add_argument("--ovqa_ans_word_size", default=707, type=int)
    parser.add_argument("--ovqa_qus_ws_path", default="./save/ws/ovqa_qus_ws.pkl")
    parser.add_argument("--ovqa_ans_ws_path", default="./save/ws/ovqa_ans_ws.pkl")
    parser.add_argument("--ovqa_qus_glove_path", default="./save/embedding/ovqa_qus_glove_emb_300d.npy")
    parser.add_argument("--ovqa_images_path", default="./data/ref/OVQA_publish/img")
    parser.add_argument("--ovqa_train_dataset_path", default="./data/ref/OVQA_publish/trainset.json")
    parser.add_argument("--ovqa_test_dataset_path", default="./data/ref/OVQA_publish/testset.json")

    # image
    parser.add_argument("--img_height", default=224, type=int)
    parser.add_argument("--img_width", default=224, type=int)
    parser.add_argument("--slake_img_mean", default=[0.38026, 0.38026, 0.38026])
    parser.add_argument("--slake_img_std", default=[0.2979, 0.2979, 0.2979])
    parser.add_argument("--rad_img_mean", default=[0.33640, 0.33630, 0.33610])
    parser.add_argument("--rad_img_std", default=[0.29664, 0.29659, 0.29642])
    parser.add_argument("--path_vqa_img_mean", default=[0.6755, 0.5576, 0.6504])
    parser.add_argument("--path_vqa_img_std", default=[0.3275, 0.3081, 0.3212])
    parser.add_argument("--ovqa_img_mean", default=[0.2016, 0.1895, 0.1793])
    parser.add_argument("--ovqa_img_std", default=[0.3169, 0.3032, 0.2927])

    args = parser.parse_args()

    main(args)
