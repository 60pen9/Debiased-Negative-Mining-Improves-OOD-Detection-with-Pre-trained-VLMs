import os
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import entropy
import torchvision
import sklearn.metrics as sk
from transformers import CLIPTokenizer
from torchvision import datasets
import torch.nn.functional as F
import torchvision
import time
import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import ImageFilter, ImageOps


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, net, head=None):
        super().__init__()
        # disable layers dedicated to ImageNet labels classification
        self.net = net

    def forward(self, x, mask_ratio):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        
        global_crops = x[0]
        local_crops = x[1]

        global_crops = [inp.cuda() for inp in global_crops]
        global_output = self.net.encode_image(torch.cat(global_crops), mask_ratio=0.0).float()

        if len(local_crops) == 0:
            local_output = None
        else:
            local_crops = [inp.cuda() for inp in local_crops]
            local_output = self.net.encode_image(torch.cat(local_crops), mask_ratio=args.mask_ratio).float()

        return global_output, local_output

def set_ood_loader_ImageNet(args, out_dataset, preprocess, root):
    '''
    set OOD loader for ImageNet scale datasets
    '''
    if out_dataset == 'iNaturalist':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'iNaturalist'), transform=preprocess)
    elif out_dataset == 'SUN':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'SUN'), transform=preprocess)
    elif out_dataset == 'places365': # filtered places
        testsetout = torchvision.datasets.ImageFolder(root= os.path.join(root, 'Places'),transform=preprocess)  
    elif out_dataset == 'placesbg': 
        testsetout = torchvision.datasets.ImageFolder(root= os.path.join(root, 'placesbg'),transform=preprocess)  
    elif out_dataset == 'dtd':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'dtd', 'images'),
                                        transform=preprocess)
    elif out_dataset == 'ImageNet10': # the train split is used due to larger and comparable size with ID dataset
        testsetout = datasets.ImageFolder(os.path.join(args.root_dir, 'ImageNet10', 'train'), transform=preprocess)
    elif out_dataset == 'ImageNet20':
        testsetout = datasets.ImageFolder(os.path.join(args.root_dir, 'ImageNet20', 'val'), transform=preprocess)
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
                                            shuffle=False, num_workers=4)
    return testloaderOut

def print_measures(log, auroc, aupr, fpr, method_name='Ours', recall_level=0.95):
    if log == None: 
        print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
        print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
        print('AUPR:  \t\t\t{:.2f}'.format(100 * aupr))
    else:
        log.debug('\t\t\t\t' + method_name)
        log.debug('  FPR{:d} AUROC AUPR'.format(int(100*recall_level)))
        log.debug('& {:.2f} & {:.2f} & {:.2f}'.format(100*fpr, 100*auroc, 100*aupr))

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr


def input_preprocessing(args, net, images, text_features = None, classifier = None):
    criterion = torch.nn.CrossEntropyLoss()
    if args.model == 'vit-Linear':
        image_features = net(pixel_values = images.float()).last_hidden_state
        image_features = image_features[:, 0, :]
    elif args.model == 'CLIP-Linear':
        image_features = net.encode_image(images).float()
    if classifier:
        outputs = classifier(image_features) / args.T
    else: 
        image_features = image_features/ image_features.norm(dim=-1, keepdim=True) 
        outputs = image_features @ text_features.T / args.T
    pseudo_labels = torch.argmax(outputs.detach(), dim=1)
    loss = criterion(outputs, pseudo_labels) # loss is NEGATIVE log likelihood
    loss.backward()

    sign_grad =  torch.ge(images.grad.data, 0) # sign of grad 0 (False) or 1 (True)
    sign_grad = (sign_grad.float() - 0.5) * 2  # convert to -1 or 1

    std=(0.26862954, 0.26130258, 0.27577711) # for CLIP model
    for i in range(3):
        sign_grad[:,i] = sign_grad[:,i]/std[i]

    processed_inputs = images.data  - args.noiseMagnitude * sign_grad # because of nll, here sign_grad is actually: -sign of gradient
    return processed_inputs
  
def get_mean_prec(args, net, train_loader):
    '''
    used for Mahalanobis score. Calculate class-wise mean and inverse covariance matrix
    '''
    classwise_mean = torch.empty(args.n_cls, args.feat_dim, device =args.gpu)
    all_features = []
    # classwise_features = []
    from collections import defaultdict
    classwise_idx = defaultdict(list)
    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.cuda()
            if args.model == 'CLIP': 
                features = net.get_image_features(pixel_values = images).float()
            if args.normalize: 
                features /= features.norm(dim=-1, keepdim=True)
            for label in labels:
                classwise_idx[label.item()].append(idx)
            all_features.append(features.cpu()) #for vit
    all_features = torch.cat(all_features)
    for cls in range(args.n_cls):
        classwise_mean[cls] = torch.mean(all_features[classwise_idx[cls]].float(), dim = 0)
        if args.normalize: 
            classwise_mean[cls] /= classwise_mean[cls].norm(dim=-1, keepdim=True)
    cov = torch.cov(all_features.T.double()) 
    precision = torch.linalg.inv(cov).float()
    print(f'cond number: {torch.linalg.cond(precision)}')
    torch.save(classwise_mean, os.path.join(args.template_dir,f'{args.model}_classwise_mean_{args.in_dataset}_{args.max_count}_{args.normalize}.pt'))
    torch.save(precision, os.path.join(args.template_dir,f'{args.model}_precision_{args.in_dataset}_{args.max_count}_{args.normalize}.pt'))
    return classwise_mean, precision

def get_Mahalanobis_score(args, net, test_loader, classwise_mean, precision, in_dist = True):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    '''
    # net.eval()
    Mahalanobis_score_all = []
    total_len = len(test_loader.dataset)
    tqdm_object = tqdm(test_loader, total=len(test_loader))
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm_object):
            if (batch_idx >= total_len // args.batch_size) and in_dist is False:
                break   
            images, labels = images.cuda(), labels.cuda()
            if args.model == 'CLIP':
                features = net.get_image_features(pixel_values = images).float()
            if args.normalize: 
                features /= features.norm(dim=-1, keepdim=True)
            for i in range(args.n_cls):
                class_mean = classwise_mean[i]
                zero_f = features - class_mean
                Mahalanobis_dist = -0.5*torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
                if i == 0:
                    Mahalanobis_score = Mahalanobis_dist.view(-1,1)
                else:
                    Mahalanobis_score = torch.cat((Mahalanobis_score, Mahalanobis_dist.view(-1,1)), 1)      
            Mahalanobis_score, _ = torch.max(Mahalanobis_score, dim=1)
            Mahalanobis_score_all.extend(-Mahalanobis_score.cpu().numpy())
        
    return np.asarray(Mahalanobis_score_all, dtype=np.float32)

def get_ood_scores_clip(args, net, loader, test_labels, text_features_neg, text_features_pos, in_dist=False):
    '''
    used for scores based on img-caption product inner products: MIP, entropy, energy score. 
    '''

    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)

    _score = []
    _global_features = []

    # tokenizer = CLIPTokenizer.from_pretrained(args.ckpt)

    # model = MultiCropWrapper(net)

    model = net

    tqdm_object = tqdm(loader, total=len(loader))


    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm_object):
            # bz = images.size(0)
            # labels = labels.long().cuda()
            images = images.cuda()
  
            if args.model == 'CLIP':

                global_features = model.encode_image(images, mask_ratio=0.0).float()
                global_features /= global_features.norm(dim=-1, keepdim=True)


                if args.mc_local_number == 0:

                    text_features_pos = text_features_pos.cuda()
                    #
                    # output_pos = global_features @ text_features_pos.T / args.t1
                    #
                    # score = torch.exp(output_pos)/torch.sum(torch.exp(output_pos), dim=1, keepdim=True)
                    #
                    # score = np.max(to_np(score), axis=1)


                    score = 0

                    for j in range(args.ngroup):

                        text_features = torch.cat((text_features_pos, text_features_neg[j]), dim=0)

                        output_pos = global_features @ text_features_pos.T/args.t1

                        output = global_features @ text_features.T/args.t1

                        score_j = torch.exp(output_pos)/torch.sum(torch.exp(output), dim=1, keepdim=True)

                        score = score + score_j/args.ngroup

                    score = args.beta*np.log(np.sum(np.exp(to_np(torch.log(score)) / args.beta), axis=1))

                    # score = np.max(score, axis=1)


                else:

                    ######___V4.1___###########

                    score = 0

                    text_features_neg_merged = torch.reshape(text_features_neg, (text_features_neg.size(0)*text_features_neg.size(1), args.feat_dim))

                    text_features_merged = torch.cat((text_features_pos, text_features_neg_merged), dim=0)

                    for i in range(args.mc_local_number):

                        local_features = model.encode_image(images, mask_ratio=0.0).float()

                        local_features /= local_features.norm(dim=-1, keepdim=True)
                        
                        local_output_merged = local_features @ text_features_merged.T/args.t3

                        w_xp_y = F.softmax(local_output_merged, dim=1)

                        # global_local_features = args.alpha * global_features + (1-args.alpha) * local_features[i]

                        global_local_features = global_features + args.alpha * (global_features-local_features[i])

                        global_local_features /= global_local_features.norm(dim=-1, keepdim=True)

                        global_local_output_pos = global_local_features @ text_features_pos.T/args.t2

                        global_local_output_merged = global_local_features @ text_features_merged.T/args.t2

                        z_xp_y = torch.sum(w_xp_y*torch.exp(global_local_output_merged), dim=1, keepdim=True)

                        score_x_y_I_xp = torch.exp(global_local_output_pos)/z_xp_y

                        score_i_xp_y = 0

                        for j in range(args.ngroup):

                            text_features = torch.cat((text_features_pos, text_features_neg[j]), dim=0)

                            local_output_pos = local_features @ text_features_pos.T/args.t1

                            local_output = local_features @ text_features.T/args.t1

                            z_xp = torch.sum(torch.exp(local_output), dim=1, keepdim=True)

                            score_i_xp_y_j = torch.exp(local_output_pos)/z_xp

                            score_i_xp_y = score_i_xp_y + score_i_xp_y_j/args.ngroup

                        score_i = torch.log(score_i_xp_y) + torch.log(score_x_y_I_xp)

                        score = score + score_i/args.mc_local_number

                    score = args.beta*np.log(np.sum(np.exp(to_np(score) / args.beta), axis=1))

                    ######___V4.1___###########

                    ######___V4.2___###########

                    # score = 0
                    #
                    # text_features_neg_merged = torch.reshape(text_features_neg, (text_features_neg.size(0)*text_features_neg.size(1), args.feat_dim))
                    #
                    # text_features_merged = torch.cat((text_features_pos, text_features_neg_merged), dim=0)
                    #
                    # for i in range(args.mc_local_number):
                    #
                    #     score_i = 0
                    #
                    #     score_i_xp_y = 0
                    #
                    #     w_i_xp = []
                    #
                    #     local_features = model.encode_image(images, mask_ratio=args.mask_ratio).float()
                    #     local_features /= local_features.norm(dim=-1, keepdim=True)
                    #
                    #     for j in range(args.ngroup):
                    #
                    #         text_features = torch.cat((text_features_pos, text_features_neg[j]), dim=0)
                    #
                    #         local_output_pos = local_features @ text_features_pos.T
                    #
                    #         local_output = local_features @ text_features.T
                    #
                    #         z_i_xp_j = torch.sum(torch.exp(local_output/args.t1), dim=1, keepdim=True)
                    #
                    #         score_i_xp_y_j = torch.exp(local_output_pos/args.t1)/z_i_xp_j
                    #
                    #         score_i_xp_y = score_i_xp_y + score_i_xp_y_j/args.ngroup
                    #
                    #         w_i_xp_j = torch.sum(torch.exp(local_output/args.t3), dim=1, keepdim=True)
                    #
                    #     w_i_xp.append(w_i_xp_j)
                    #
                    #     w_i_xp = torch.cat(w_i_xp, dim=1)
                    #
                    #     local_output_merged = local_features @ text_features_merged.T/args.t3
                    #
                    #     w_i_xp = w_i_xp.unsqueeze(dim=1) + torch.exp(local_output_merged).unsqueeze(dim=2)
                    #
                    #     w_i_xp = torch.mean(text_features_neg.size(1)/w_i_xp, dim=-1)
                    #
                    #     w_i_xp_y = torch.exp(local_output_merged)*w_i_xp
                    #
                    #     global_local_features = args.alpha * global_features + (1-args.alpha) * local_features
                    #
                    #     # global_local_features = global_features + args.alpha * (global_features-local_features)
                    #
                    #     global_local_features /= global_local_features.norm(dim=-1, keepdim=True)
                    #
                    #     global_local_output_pos = global_local_features @ text_features_pos.T/args.t2
                    #
                    #     global_local_output_merged = global_local_features @ text_features_merged.T/args.t2
                    #
                    #     z_xp_y = torch.sum(w_i_xp_y*torch.exp(global_local_output_merged), dim=1, keepdim=True)
                    #
                    #     score_x_y_I_xp = torch.exp(global_local_output_pos)/z_xp_y
                    #
                    #     score_i = torch.log(score_i_xp_y) + torch.log(score_x_y_I_xp)
                    #
                    #     score = score + score_i/args.mc_local_number
                    #
                    #
                    # score = args.beta*np.log(np.sum(np.exp(to_np(score) / args.beta), axis=1))
                    ######___V4.2___###########

            # score = to_np(score)
            # score = np.mean(score, axis=1)

            if args.score == 'energy':
                #Energy = - T * logsumexp(logit_k / T), by default T = 1 in https://arxiv.org/pdf/2010.03759.pdf
                _score.append(-to_np((args.T*torch.logsumexp(output / args.T, dim=1))))  #energy score is expected to be smaller for ID
            elif args.score == 'entropy':  
                # raw_value = entropy(smax)
                # filtered = raw_value[raw_value > -1e-5]
                _score.append(entropy(smax, axis = 1)) 
                # _score.append(filtered) 
            elif args.score == 'var':
                _score.append(-np.var(smax, axis = 1))
            elif args.score in ['MCM', 'max-logit']:
                _score.append(-np.max(smax, axis=1))
            elif args.score == 'ours':
                _score.append(-score)

            _global_features.append(global_features.cpu())

    return concat(_score)[:len(loader.dataset)].copy(), torch.cat(_global_features, dim=0)



def get_and_print_results(args, log, in_score, out_score, auroc_list, aupr_list, fpr_list):
    '''
    1) evaluate detection performance for a given OOD test set (loader)
    2) print results (FPR95, AUROC, AUPR)
    '''
    aurocs, auprs, fprs = [], [], []
    measures = get_measures(-in_score, -out_score)
    aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    # print(f'in score samples (random sampled): {in_score[:3]}, out score samples: {out_score[:3]}')
    # print(f'in score samples (min): {in_score[-3:]}, out score samples: {out_score[-3:]}')
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr) # used to calculate the avg over multiple OOD test sets
    print_measures(log, auroc, aupr, fpr)

class TextDataset(torch.utils.data.Dataset):
    '''
    used for MIPC score. wrap up the list of captions as Dataset to enable batch processing
    '''
    def __init__(self, texts, labels):
        self.labels = labels
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        # Load data and get label
        X = self.texts[index]
        y = self.labels[index]

        return X, y
