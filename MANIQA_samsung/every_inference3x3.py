import os
import torch
import numpy as np
import logging
import time
import torch.nn as nn
import random

from torchvision import transforms
from torch.utils.data import DataLoader
from models.maniqa import MANIQA_5x5, MANIQA_3x3
from config import Config
from utils.process import RandCrop_inference, ToTensor_inference, Normalize_infernece, five_point_crop
from utils.process import split_dataset_kadid10k, split_dataset_koniq10k_for_test
from utils.process import RandRotation, RandHorizontalFlip
from scipy.stats import spearmanr, pearsonr
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def eval_epoch(config, net, test_loader):
    with torch.no_grad():
        losses = []
        net.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []

        for data in tqdm(test_loader):
            pred = 0
            for i in range(config.num_avg_val):
                x_d = data['d_img_org'].cuda()
                x_d = five_point_crop(i, d_img=x_d, config=config)
                pred += net(x_d)

            pred /= config.num_avg_val
            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()*10
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)


        return pred_epoch



if __name__ == '__main__':
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    setup_seed(20)

    config = Config({
        # dataset path
        "dataset_name": "koniq10k",

        # PIPAL
        "train_dis_path": "/mnt/IQA_dataset/PIPAL22/Train_dis/",
        "val_dis_path": "/mnt/IQA_dataset/PIPAL22/Val_dis/",
        "pipal22_train_label": "./data/PIPAL22/pipal22_train.txt",
        "pipal22_val_txt_label": "./data/PIPAL22/pipal22_val.txt",

        # KADID-10K
        "kadid10k_path": "/mnt/IQA_dataset/kadid10k/images/",
        "kadid10k_label": "./data/kadid10k/kadid10k_label.txt",

        # KONIQ-10K
        "koniq10k_path": "./IQA_dataset/test/",
        "koniq10k_HZ_path": "./IQA_dataset/test_mscn/",
        "koniq10k_label": "./data/koniq10k/test_jpg.txt",

        # optimization
        "batch_size": 4,
        "learning_rate": 1e-6,
        "weight_decay": 1e-5,
        "n_epoch": 300,
        "val_freq": 1,
        "T_max": 50,
        "eta_min": 0,
        "num_avg_val": 1,  # if training koniq10k, num_avg_val is set to 1
        "num_workers": 8,  # -> cpu core

        # data
        "split_seed": 20,
        "train_keep_ratio": 1.0,
        "val_keep_ratio": 1.0,
        "crop_size": 224,
        "prob_aug": 0.5,

        # model
        "patch_size": 8,
        "img_size": 224,
        "embed_dim": 768,
        "dim_mlp": 768,
        "num_heads": [4, 4],
        "window_size": 4,
        "depths": [2, 2],
        "num_outputs": 1,
        "num_tab": 2,
        "scale": 0.8,

        "ckpt_path": "./output/learning_rate_6_mscn_RGB_3x3_seed78/Koniq10k/koniq10k-base_s20/epoch8.pt",  # directory for saving checkpoint
        "log_path": "./output/log/",
        "log_file": "koniq10k-base_s206.log",
        "tensorboard_path": "./output/tensorboard/"
    })
    if config.dataset_name == 'koniq10k':
        from data.koniq10k.koniq10k import Koniq10k_inference
        val_name = split_dataset_koniq10k_for_test(
            txt_file_name=config.koniq10k_label,
            split_seed=config.split_seed
        )
        dis_train_path = config.koniq10k_path
        dis_val_path = config.koniq10k_path
        label_train_path = config.koniq10k_label
        label_val_path = config.koniq10k_label
        dis_train_HZ_path = config.koniq10k_HZ_path
        Dataset = Koniq10k_inference



    val_dataset = Dataset(
        dis_path=dis_val_path,
        txt_file_name=label_val_path,
        list_name=val_name,
        dis_HZ_path=dis_train_HZ_path,
        transform=transforms.Compose([RandCrop_inference(patch_size=config.crop_size),
            Normalize_infernece(0.5, 0.5), ToTensor_inference()]),
        keep_ratio=config.val_keep_ratio
    )

    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size,
                            num_workers=config.num_workers, drop_last=True, shuffle=False)

    net = MANIQA_3x3(embed_dim=config.embed_dim, num_outputs=config.num_outputs, dim_mlp=config.dim_mlp,
        patch_size=config.patch_size, img_size=config.img_size, window_size=config.window_size,
        depths=config.depths, num_heads=config.num_heads, num_tab=config.num_tab, scale=config.scale)

    net.load_state_dict(torch.load(config.ckpt_path), strict=False)
    net = net.cuda()

    logging.info('{} : {} [M]'.format('#Params', sum(map(lambda x: x.numel(), net.parameters())) / 10 ** 6))

    logging.info('Starting eval...')
    pred = eval_epoch(config, net, val_loader)
    logging.info('Eval done...')
    result_df = pd.DataFrame({
        'img_name': val_name,
        'mos': pred,

    })
    result_df.to_csv('./result/learning_rate_6_mscn_RGB_3x3_seed78.csv', index=False)

    print("Inference completed and results saved to submit.csv.")