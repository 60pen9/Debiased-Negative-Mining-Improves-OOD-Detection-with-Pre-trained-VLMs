import os
import argparse
import numpy as np
import torch
from scipy import stats

import faiss

from CLIP import clip

import torch.nn.functional as F

import random

from utils.common import setup_seed, get_num_cls, get_test_labels
from utils.detection_util import get_Mahalanobis_score, get_mean_prec, print_measures, get_and_print_results, get_ood_scores_clip
from utils.file_ops import save_as_dataframe, setup_log
from utils.plot_util import plot_distribution
from utils.train_eval_util import  set_model_clip, set_train_loader, set_val_loader, set_ood_loader_ImageNet
# sys.path.append(os.path.dirname(__file__))

import main_online

import time
import warnings
warnings.filterwarnings("ignore")

to_np = lambda x: x.data.cpu().numpy()

def process_args():
    parser = argparse.ArgumentParser(description='Feature Extraction for CLIP',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # setting for each run
    parser.add_argument('--in_dataset', default='ImageNet', type=str,
                        choices=['ImageNet', 'ImageNet100', 'bird200', 'car196', 'food101', 'pet37','ImageNet-A','ImageNet-R','ImageNet-S','ImageNet-V2'],
                        help='in-distribution dataset')
    parser.add_argument('--root-dir', default="datasets", type=str,
                        help='root dir of datasets')
    parser.add_argument('--name', default="eval_ood",
                        type=str, help="unique ID for the run")
    parser.add_argument('--seed', default=5, type=int, help="random seed") #5
    parser.add_argument('--gpu', default=1, type = int,
                        help='the GPU indice to use')

    parser.add_argument('--model', default='CLIP', type=str, help='model architecture')
    parser.add_argument('--CLIP_ckpt', type=str, default='ViT-B/16',
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px', 'RN50'], help='which pretrained img encoder to use')
    parser.add_argument('--score', default='ours', type=str, choices=[
        'MCM', 'energy', 'max-logit', 'entropy', 'var', 'maha', 'ours'], help='score options')

    parser.add_argument(
        "--mc_global_number",
        type=int,
        default=1,
        help="Number of random global crops.",
    )
    parser.add_argument(
        "--mc_global_scale",
        type=float,
        nargs="+",
        default=(0.6, 1.0),
        help="Scale range for global crops.",
    )    
    
    parser.add_argument('-b', '--batch-size', default=1024, type=int,
                        help='mini-batch size')
                        
    # hyper-parameters

    parser.add_argument('--ngroup', type=int, default=100,
                        help='number of grouping')

    parser.add_argument('--alpha', type=float, default=0.25,
                        help='temperature parameter ')
                        
    parser.add_argument('--beta', type=float, default=1.0,
                        help='temperature parameter')

    parser.add_argument('--t1', type=float, default=0.01,
                        help='temperature parameter')

    parser.add_argument('--t2', type=float, default=0.02,
                        help='temperature parameter')

    parser.add_argument('--t3', type=float, default=50,
                        help='temperature parameter')

    parser.add_argument(
        "--mc_local_number",
        type=int,
        default=0,
        help="Number of random local crops.",

    )

    parser.add_argument(
        "--mc_local_scale",
        type=float,
        nargs="+",
        default=(0.55, 1.0),
        help="Scale range for local crops.",
    )

    parser.add_argument('--mask_ratio', type=float, default=0.1,
                        help='temperature parameter ')

    # end


    # for Mahalanobis score
    parser.add_argument('--feat_dim', type=int, default=512, help='feat dim； 512 for ViT-B and 768 for ViT-L')
    parser.add_argument('--normalize', type = bool, default = False, help='whether use normalized features for Maha score')
    parser.add_argument('--generate', type = bool, default = True, help='whether to generate class-wise means or read from files for Maha score')
    parser.add_argument('--template_dir', type = str, default = 'img_templates', help='the loc of stored classwise mean and precision matrix')
    parser.add_argument('--subset', default = False, type =bool, help = "whether uses a subset of samples in the training set")
    parser.add_argument('--max_count', default = 250, type =int, help = "how many samples are used to estimate classwise mean and precision matrix")
    args = parser.parse_args()

    args.n_cls = get_num_cls(args)
    args.log_directory = f"results/{args.in_dataset}/{args.score}/{args.model}_{args.CLIP_ckpt}_ID_{args.name}"
    os.makedirs(args.log_directory, exist_ok=True)

    return args

def main():
    args = process_args()
    setup_seed(args.seed)
    log = setup_log(args)
    assert torch.cuda.is_available()
    torch.cuda.set_device(args.gpu)

    net, preprocess = set_model_clip(args)
    net.eval()

    if args.in_dataset in ['ImageNet10']: 
        out_datasets = ['ImageNet20']
    elif args.in_dataset in ['ImageNet20']: 
        out_datasets = ['ImageNet10']
    elif args.in_dataset in ['ImageNet', 'ImageNet100', 'bird200', 'car196', 'food101', 'pet37','ImageNet-A','ImageNet-R','ImageNet-S','ImageNet-V2']:
         out_datasets = ['iNaturalist','SUN', 'places365', 'dtd']

    test_loader = set_val_loader(args, preprocess)
    test_labels = get_test_labels(args, test_loader)
    # print(test_labels)

    emb_batchsize = 5000

    # from transformers import CLIPTokenizer

    # tokenizer = CLIPTokenizer.from_pretrained(args.ckpt)

    # dump_dict = torch.load('./selected_neg_labels/neg_dump.pth')
    # dump_dict = torch.load('./selected_neg_labels/neg_dump_new2.pth')
    #
    # text_features_neg = dump_dict['neg_emb_selected'].cuda().to(torch.float32)
    # text_features_pos = dump_dict['pos_emb'].cuda().to(torch.float32)
    # noun_length = dump_dict['noun_length']
    # adj_length = dump_dict['adj_length']
    # print(text_features_neg.size(), noun_length, adj_length)

    noun_prompt_templates = [
        'the nice {}',
    ]

    adj_prompt_templates = [
        'This is a {} photo'
    ]



    with (torch.no_grad()):

        # text_inputs_pos = tokenizer([f"the nice {c}" for c in test_labels], padding=True, return_tensors="pt")
        # text_features_pos = net.get_text_features(input_ids=text_inputs_pos['input_ids'].cuda(),
        #                                             attention_mask=text_inputs_pos['attention_mask'].cuda()).float()
        # text_features_pos /= text_features_pos.norm(dim=-1, keepdim=True)

        text_features_pos = []
        for template in noun_prompt_templates:
            text_inputs_pos = clip.tokenize([f"{template.format(c)}"for c in test_labels]).cuda() #tokenize
            text_feature_pos = net.encode_text(text_inputs_pos).float() #embed with text encoder
            text_features_pos.append(torch.unsqueeze(text_feature_pos, dim=1))

        text_features_pos = torch.cat(text_features_pos, dim=1)
        dump_dict = dict(pos_emb=text_features_pos.cpu())
        torch.save(dump_dict, './pos_dump_new_ens.pth')

        text_features_pos = text_features_pos.mean(dim=1)



        wordnet_database = './txtfiles'

        txtfiles = os.listdir(wordnet_database)

        words_noun = []
        words_adj = []
        prompt_templete = dict(
            adj= adj_prompt_templates,
            noun=noun_prompt_templates,
        )

        dedup = dict()
        noun_length = 0
        adj_length = 0
        for file in txtfiles:
            filetype = file.split('.')[0]
            if filetype not in prompt_templete:
                continue
            if file in ['noun.person.txt', 'noun.quantity.txt', 'noun.group.txt', 'adj.pert.txt']:
                continue
            with open(os.path.join(wordnet_database, file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.replace('_', ' ')
                    if line.strip() in dedup:
                        continue
                    dedup[line.strip()] = None
                    if filetype == 'noun':
                        noun_length += 1
                        words_noun.append(line.strip())
                        # for template in prompt_templete[filetype]:
                        #     words_noun.append(template.format(line.strip()))
                    elif filetype == 'adj':
                        adj_length += 1
                        candidate = line.strip()
                        # candidate = random.choice(csp_templates).format(candidate)
                        words_adj.append(candidate)
                        # for template in prompt_templete[filetype]:
                        #     words_adj.append(template.format(line.strip()))
                    else:
                        raise TypeError

        text_features_neg_noun = []
        for template in noun_prompt_templates:
            text_features_neg_noun_template = []
            for i in range(0, noun_length, emb_batchsize):
                x = words_noun[i: i + emb_batchsize]
                x = clip.tokenize([f"{template.format(c)}" for c in x]).cuda()  # tokenize
                text_feature_neg_noun = net.encode_text(x).float()  # embed with text encoder
                text_features_neg_noun_template.append(torch.unsqueeze(text_feature_neg_noun, dim=1))

            text_features_neg_noun_template = torch.cat(text_features_neg_noun_template, dim=0)
            text_features_neg_noun.append(text_features_neg_noun_template)

        text_features_neg_noun = torch.cat(text_features_neg_noun, dim=1).mean(dim=1)

        if adj_length>0:

            text_features_neg_adj = []
            for template in adj_prompt_templates:
                text_features_neg_adj_template = []
                for i in range(0, adj_length, emb_batchsize):
                    x = words_adj[i: i + emb_batchsize]
                    x = clip.tokenize([f"{template.format(c)}" for c in x]).cuda()  # tokenize
                    text_feature_neg_adj = net.encode_text(x).float()  # embed with text encoder
                    text_features_neg_adj_template.append(torch.unsqueeze(text_feature_neg_adj, dim=1))

                text_features_neg_adj_template = torch.cat(text_features_neg_adj_template, dim=0)
                text_features_neg_adj.append(text_features_neg_adj_template)

            text_features_neg_adj = torch.cat(text_features_neg_adj, dim=1).mean(dim=1)

            # text_inputs_neg_adj = clip.tokenize([f"{adj_prompt_templates[0].format(c)}" for c in words_adj]).cuda()  # tokenize
            # text_features_neg_adj = net.encode_text(text_inputs_neg_adj).float()  # embed with text encoder

            text_features_neg = torch.cat([text_features_neg_noun, text_features_neg_adj], dim=0)
        else:
            text_features_neg = text_features_neg_noun

        ensemble_noun_length = len(text_features_neg)

        print(f'Candidate pool size: {ensemble_noun_length}')
        print(f'Noun candidates: {noun_length}')
        print(f'Adj candidates: {adj_length}')


    neg_sim = []

    # B16 L14
    neg_topk_n = 0.23
    neg_topk_a = 0.13
    k1 = 100

    # B32
    # neg_topk_n = 0.25
    # neg_topk_a = 0.13
    # k1 = 10

    # RN50
    # neg_topk_n = 0.25
    # neg_topk_a = 0.13
    # k1 = 100

    # L14_336
    # neg_topk_n = 0.25
    # neg_topk_a = 0.08
    # k1 = 100


    norm = text_features_neg.norm(dim=-1, keepdim=True)
    text_features_neg_norm = text_features_neg / text_features_neg.norm(dim=-1, keepdim=True)

    sim = norm-2*text_features_neg_norm @ text_features_neg_norm.T+norm.T

    neg_knn = torch.topk(-sim, k=k1+1, dim=1)[0]
    neg_knn = torch.log(-neg_knn).mean(dim=1)

    neg_sim = neg_knn

    neg_sim_noun_ori = neg_sim[:noun_length]
    neg_sim_adj_ori = neg_sim[noun_length:]

    text_features_neg_noun = text_features_neg[:noun_length]
    text_features_neg_adj = text_features_neg[noun_length:]

    ind_noun = torch.argsort(neg_sim_noun_ori)
    ind_adj = torch.argsort(neg_sim_adj_ori)

    text_features_neg_noun_selected = text_features_neg_noun[ind_noun[0:int(len(ind_noun) * neg_topk_n)]]
    text_features_neg_adj_selected = text_features_neg_adj[ind_adj[0:int(len(ind_adj) * neg_topk_a)]]

    text_features_neg = torch.cat([text_features_neg_noun_selected, text_features_neg_adj_selected], dim=0)


    dump_dict = dict(pos_emb=text_features_pos.cpu(), neg_emb_selected=text_features_neg.cpu(), noun_length=noun_length,
                     adj_length=adj_length)

    torch.save(dump_dict, './neg_dump_new_B16.pth')
    # torch.save(dump_dict, './neg_dump_new_B32.pth')
    # torch.save(dump_dict, './neg_dump_new_L14.pth')
    # torch.save(dump_dict, './neg_dump_new_RN50.pth')
    # torch.save(dump_dict, './neg_dump_new_L14_336.pth')


    print(f'ID_length: {text_features_pos.shape[0]}')
    # print(f'selected_noun_length: {text_features_neg_noun_selected.shape[0]}')
    # print(f'selected_adj_length: {text_features_neg_adj_selected.shape[0]}')
    print(f'total_selected_neg_labels: {text_features_neg.shape[0]}')

    # main_online.main()


    # text_features_neg = text_features_neg[0:10000]
    # print(text_features_neg.size())
    #
    # dump_dict=dict(neg_emb=text_features_neg.cpu())
    # torch.save(dump_dict, './selected_neg_labels/neg_dump_sorted.pth')


    drop = text_features_neg.shape[0] % args.ngroup

    if args.ngroup > 1:
        random_permute = True
    else:
        random_permute = False

    if drop > 0:
        text_features_neg = text_features_neg[:-drop,:]
    if random_permute:
        idx = torch.randperm(text_features_neg.size(0)).cuda()
        text_features_neg = text_features_neg[idx]

    text_features_neg = torch.reshape(text_features_neg, (args.ngroup, -1, text_features_neg.size(1)))

    feature_dict = dict()
    if args.score == 'maha':
        os.makedirs(args.template_dir, exist_ok = True)
        train_loader = set_train_loader(args, preprocess, subset = args.subset)
        if args.generate:
            classwise_mean, precision = get_mean_prec(args, net, train_loader)
        classwise_mean = torch.load(os.path.join(args.template_dir, f'{args.model}_classwise_mean_{args.in_dataset}_{args.max_count}_{args.normalize}.pt'), map_location= 'cpu').cuda()
        precision = torch.load(os.path.join(args.template_dir,  f'{args.model}_precision_{args.in_dataset}_{args.max_count}_{args.normalize}.pt'), map_location= 'cpu').cuda()
        in_score = get_Mahalanobis_score(args, net, test_loader, classwise_mean, precision, in_dist = True)
    else:
        in_score, global_features = get_ood_scores_clip(args, net, test_loader, test_labels, text_features_neg, text_features_pos, in_dist=True)
        print(global_features.shape)
        feature_dict[args.in_dataset] = global_features

    auroc_list, aupr_list, fpr_list = [], [], []
    for out_dataset in out_datasets:
        log.debug(f"Evaluting OOD dataset {out_dataset}")
        ood_loader = set_ood_loader_ImageNet(args, out_dataset, preprocess, root=os.path.join(args.root_dir, 'ImageNet_OOD_dataset'))
        if args.score == 'maha':
            out_score = get_Mahalanobis_score(args, net, ood_loader, classwise_mean, precision, in_dist = False)
        else:
            out_score, global_features = get_ood_scores_clip(args, net, ood_loader, test_labels, text_features_neg, text_features_pos)
            feature_dict[out_dataset] = global_features
            print(global_features.shape)


    # torch.save(feature_dict, './CLIP_features_L14_336.pth')
    # torch.save(feature_dict, './CLIP_features_RN50.pth')
    # torch.save(feature_dict, './CLIP_features_L14.pth')
    # torch.save(feature_dict, './CLIP_features_B32.pth')
    torch.save(feature_dict, './CLIP_features_B16.pth')

    # torch.save(feature_dict, './CLIP_features_B16_A.pth')
    # torch.save(feature_dict, './CLIP_features_B16_R.pth')
    # torch.save(feature_dict, './CLIP_features_B16_S.pth')
    # torch.save(feature_dict, './CLIP_features_B16_V2.pth')




if __name__ == '__main__':
    main()
