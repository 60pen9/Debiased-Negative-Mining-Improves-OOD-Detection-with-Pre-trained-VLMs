# Copyright (c) Alibaba Group
import argparse
import torch
import torchvision.datasets as datasets
import torch.nn.functional as F
import os
import math
import numpy as np
import random
import logging

from sympy.abc import alpha
from tqdm import tqdm

from utils.detection_util import print_measures, get_and_print_results
from utils.file_ops import save_as_dataframe, setup_log
from utils.plot_util import plot_distribution

import scipy.optimize as sopt

parser = argparse.ArgumentParser(description='OOD scoring for ImageNet')

parser.add_argument('--seed', default=50, type=int, help="random seed") #50

# hyper-parameters
parser.add_argument('--temp', default=0.01, type=float) # A 0.013

parser.add_argument('--ngroups', default=100, type=int)

parser.add_argument('--tau_ood', default=0.5, type=float, help="between 0 and 1") #B16 L14 RN50 0.5 #B32 0.12 #L14_336 0.12

# parser.add_argument('--tau_ood', default=0.12, type=float, help="between 0 and 1") #A R V2 S 1.0


# end

torch.cuda.set_device(0)

to_np = lambda x: x.data.cpu().numpy()
concat = lambda x: np.concatenate(x, axis=0)

# 'ImageNet', 'ImageNet-A', 'ImageNet-R', 'ImageNet-S', 'ImageNetV2'
in_dataset = 'ImageNet'
out_datasets = ['iNaturalist','SUN', 'places365', 'dtd']


def setup_log(args):
    log = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(".\ood_eval_info.log", mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    log.setLevel(logging.DEBUG)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)
    log.debug(f"#########eval_ood############")
    return log

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_score(image_features, text_features_pos, text_features_neg, text_features_pos_ens, args):

    ngroup = args.ngroups
    temp = args.temp
    tau_ood = args.tau_ood

    text_features_neg = text_features_neg

    drop = text_features_neg.shape[0] % ngroup

    if ngroup>1:
        random_permute = True
    else:
        random_permute = False

    if drop > 0:
        text_features_neg = text_features_neg[:-drop,:]

    if random_permute:
        idx = torch.randperm(text_features_neg.size(0)).cuda()
        text_features_neg = text_features_neg[idx]


    text_features_neg = torch.reshape(text_features_neg, (ngroup, -1, text_features_neg.size(1)))

    print(text_features_neg.shape)
    print(text_features_pos.shape)


    text_features_neg_pos = text_features_pos

    text_features_pos = text_features_pos / text_features_pos.norm(dim=-1, keepdim=True)
    text_features_neg = text_features_neg / text_features_neg.norm(dim=-1, keepdim=True)
    text_features_neg_pos = text_features_neg_pos / text_features_neg_pos.norm(dim=-1, keepdim=True)

    Q = text_features_neg.size(1)

    score = 0

    for j in range(ngroup):

        sim_pos = image_features @ text_features_pos.T

        sim_neg_j = image_features @ text_features_neg[j].T

        sim_neg_pos_j = image_features @ text_features_neg_pos.T

        output_pos = torch.exp(sim_pos / temp)

        Z_j_1 = torch.exp(sim_pos / temp).sum(-1, keepdim=True)

        Z_j_2 = (1 / tau_ood) * torch.mean(torch.exp(sim_neg_j / temp), dim=-1, keepdim=True)

        Z_j_3 = (1 - tau_ood) / tau_ood * torch.mean(torch.exp(sim_neg_pos_j / temp), dim=-1, keepdim=True)

        Z_j = Z_j_1 + Q * (Z_j_2 - Z_j_3)

        output = output_pos / Z_j

        score_j = output[:, 0:text_features_pos.shape[0]]

        score = score + score_j / ngroup

    score = to_np(score)

    score = np.mean(score, axis=1)

    return score

def main():

    args = parser.parse_args()

    setup_seed(args.seed)

    log = setup_log(args)

    # image_feature_dict = torch.load('./CLIP_features_L14.pth')
    # image_feature_dict = torch.load('./CLIP_features_B32.pth')
    image_feature_dict = torch.load('./CLIP_features_B16.pth')
    # image_feature_dict = torch.load('./CLIP_features_RN50.pth')
    # image_feature_dict = torch.load('./CLIP_features_L14_336.pth')

    # image_feature_dict = torch.load('./CLIP_features_B16_A.pth')
    # image_feature_dict = torch.load('./CLIP_features_B16_R.pth')
    # image_feature_dict = torch.load('./CLIP_features_B16_V2.pth')
    # image_feature_dict = torch.load('./CLIP_features_B16_S.pth')

    print(image_feature_dict.keys())

    # text_feature_dict = torch.load('./neg_dump_new_L14.pth')
    text_feature_dict = torch.load('./neg_dump_new_B16.pth')
    # text_feature_dict = torch.load('./neg_dump_new_B32.pth')
    # text_feature_dict = torch.load('./neg_dump_new_RN50.pth')
    # text_feature_dict = torch.load('./neg_dump_new_L14_336.pth')





    print('load pre-trained CLIP text features')
    text_features_pos = text_feature_dict['pos_emb'].cuda()
    id_class_num = text_features_pos.shape[0]

    text_features_neg = text_feature_dict['neg_emb_selected'].cuda()
    ood_class_num = text_features_neg.shape[0]

    text_features_pos_ens = torch.load('./pos_dump_new_ens.pth')['pos_emb'].cuda()
    print(text_features_pos_ens.shape)


    print(f'ID_length: {id_class_num}')
    print(f'total_selected_neg_labels: {ood_class_num}')

    auroc_list, aupr_list, fpr_list = [], [], []

    log.debug(f"Evaluting OOD dataset {in_dataset}")

    image_features_in = image_feature_dict[in_dataset].cuda()
    id_sample_num = image_features_in.shape[0]
    print(f'ID_sample_number: {id_sample_num}')



    # in_scores = get_score(image_features_in, text_features_pos, text_features_neg, args)

    for out_dataset in out_datasets:

        log.debug(f"Evaluting OOD dataset {out_dataset}")

        out_scores = []

        image_features_out = image_feature_dict[out_dataset].cuda()
        out_sample_num = image_features_out.shape[0]
        print(f'OOD_feature_length: {out_sample_num}')

        # in_scores = get_score(image_features_in, text_features_pos, text_features_neg, args)

        # out_scores = get_score(image_features_out, text_features_pos, text_features_neg, args)

        image_features = torch.cat((image_features_in, image_features_out), dim=0)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        scores = get_score(image_features, text_features_pos, text_features_neg, text_features_pos_ens, args)

        in_scores = scores[0:id_sample_num]
        out_scores = scores[id_sample_num:]

        get_and_print_results(args, log, -in_scores, -out_scores,
                              auroc_list, aupr_list, fpr_list)


    print_measures(log, np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list))


if __name__ == '__main__':
    main()

