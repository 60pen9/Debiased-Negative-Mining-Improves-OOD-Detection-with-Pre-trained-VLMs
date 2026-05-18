import sys
import os
import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from transformers import CLIPModel
from torchvision import datasets, transforms
import torchvision.transforms as transforms
from dataloaders import StanfordCars, Food101, OxfordIIITPet, Cub2011
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import ImageFilter, ImageOps

from CLIP import clip

class MultiCropAugmentation(object):
    def __init__(self, global_number, global_scale, local_number, local_scale):
        assert (global_number > 0) or (local_number > 0)
        self.global_number = global_number
        self.local_number = local_number

        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP


        self.global_tfm = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        # transformation for the local small crops
        self.local_tfm = transforms.Compose(
            [
                # transforms.Resize(224),
                # transforms.CenterCrop(256),
                transforms.RandomResizedCrop(
                    224,
                    scale=local_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def __repr__(self) -> str:
        return (
            "global_number={}, local_number={}, \nglobal_tfm={}\nlocal_tfm={}".format(
                self.global_number, self.local_number, self.global_tfm, self.local_tfm
            )
        )

    def __call__(self, image):
        global_crops = []
        local_crops = []

        for _ in range(self.global_number):
            global_crops.append(self.global_tfm(image))

        for _ in range(self.local_number):
            local_crops.append(self.global_tfm(image))

        crops = [global_crops, local_crops]

        return crops




def set_model_clip(args):
    # '''
    # load Huggingface CLIP
    # '''
    # ckpt_mapping = {"ViT-B/16":"openai/clip-vit-base-patch16", 
    #                 "ViT-B/32":"openai/clip-vit-base-patch32",
    #                 "ViT-L/14":"openai/clip-vit-large-patch14"}
    # args.ckpt = ckpt_mapping[args.CLIP_ckpt]
    # model =  CLIPModel.from_pretrained(args.ckpt)
    # if args.model == 'CLIP-Linear':
    #     model.load_state_dict(torch.load(args.finetune_ckpt, map_location=torch.device(args.gpu)))

    model, _ = clip.load(args.CLIP_ckpt)
    model.eval()
    model = model.cuda()
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP

    val_preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

    # val_preprocess = transforms.Compose([
    #         transforms.Resize(336),
    #         transforms.CenterCrop(336),
    #         transforms.ToTensor(),
    #         normalize
    #     ])

    # val_preprocess = MultiCropAugmentation(
    #     args.mc_global_number,
    #     args.mc_global_scale,
    #     args.mc_local_number,
    #     args.mc_local_scale,
    # )
    
    return model, val_preprocess

def set_train_loader(args, preprocess=None, batch_size=None, shuffle=False, subset=False):
    root = args.root_dir
    if preprocess == None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        preprocess = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    kwargs = {'num_workers': 4, 'pin_memory': True}
    if batch_size is None:  # normal case: used for trainign
        batch_size = args.batch_size
        shuffle = True
    if args.in_dataset == "ImageNet":
        path = os.path.join(root, 'ImageNet', 'train')
        dataset = datasets.ImageFolder(path, transform=preprocess)
        if subset:
            from collections import defaultdict
            classwise_count = defaultdict(int)
            indices = []
            for i, label in enumerate(dataset.targets):
                if classwise_count[label] < args.max_count:
                    indices.append(i)
                    classwise_count[label] += 1
            dataset = torch.utils.data.Subset(dataset, indices)
        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset in ["ImageNet10", "ImageNet20", "ImageNet100"]:
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(
                root, args.in_dataset, 'train'), transform=preprocess),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset == "car196":
        train_loader = torch.utils.data.DataLoader(StanfordCars(root, split="train", download=True, transform=preprocess),
                                                   batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset == "food101":
        train_loader = torch.utils.data.DataLoader(Food101(root, split="train", download=True, transform=preprocess),
                                                   batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset == "pet37":
        train_loader = torch.utils.data.DataLoader(OxfordIIITPet(root, split="trainval", download=True, transform=preprocess),
                                                   batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset == "bird200":
        train_loader = torch.utils.data.DataLoader(Cub2011(root, train = True, transform=preprocess),
                    batch_size=batch_size, shuffle=shuffle, **kwargs)
    return train_loader


def set_val_loader(args, preprocess=None):



    root = args.root_dir

    root = "/data/bpeng2/bpeng2/pythonProject1"

    bs = args.batch_size


    if preprocess == None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        preprocess = transforms.Compose([
            # Resize(336, interpolation=BICUBIC),
            # CenterCrop(336),
            transforms.ToTensor(),
            normalize
        ])
    kwargs = {'num_workers': 4, 'pin_memory': True}
    if args.in_dataset == "ImageNet":
        path = os.path.join(root, 'ImageNet1000', 'val')
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(path, transform=preprocess),
            batch_size=bs, shuffle=False, **kwargs)
    elif args.in_dataset == "ImageNet-A":
        path = os.path.join(root, 'ImageNet1000', 'val_A')
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(path, transform=preprocess),
            batch_size=bs, shuffle=False, **kwargs)
    elif args.in_dataset == "ImageNet-R":
        path = os.path.join(root, 'ImageNet1000', 'val_R')
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(path, transform=preprocess),
            batch_size=bs, shuffle=False, **kwargs)
    elif args.in_dataset == "ImageNet-S":
        path = os.path.join(root, 'ImageNet1000', 'val_S')
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(path, transform=preprocess),
            batch_size=bs, shuffle=False, **kwargs)
    elif args.in_dataset == "ImageNet-V2":
        path = os.path.join(root, 'ImageNet1000', 'val_V2')
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(path, transform=preprocess),
            batch_size=bs, shuffle=False, **kwargs)
    elif args.in_dataset in ["ImageNet10", "ImageNet20", "ImageNet100"]:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(
                root, args.in_dataset, 'val'), transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "car196":
        val_loader = torch.utils.data.DataLoader(StanfordCars(root, split="test", download=True, transform=preprocess),
                                                 batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "food101":
        val_loader = torch.utils.data.DataLoader(Food101(root, split="test", download=True, transform=preprocess),
                                                 batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "pet37":
        val_loader = torch.utils.data.DataLoader(OxfordIIITPet(root, split="test", download=True, transform=preprocess),
                                                 batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "bird200":
        val_loader = torch.utils.data.DataLoader(Cub2011(root, train = False, transform=preprocess),
                    batch_size=args.batch_size, shuffle=False, **kwargs)

    return val_loader


def set_ood_loader_ImageNet(args, out_dataset, preprocess, root):
    '''
    set OOD loader for ImageNet scale datasets
    '''

    root = "/data/bpeng2/bpeng2/pythonProject1/cider-master/datasets/large_OOD_dataset"

    bs = args.batch_size

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
        # testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'Textures'),
        #                                 transform=preprocess)
    elif out_dataset == 'ImageNet10':
        testsetout = datasets.ImageFolder(os.path.join(args.root_dir, 'ImageNet10', 'train'), transform=preprocess)
    elif out_dataset == 'ImageNet20':
        testsetout = datasets.ImageFolder(os.path.join(args.root_dir, 'ImageNet20', 'val'), transform=preprocess)
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=bs,
                                            shuffle=False, num_workers=4)
    return testloaderOut

