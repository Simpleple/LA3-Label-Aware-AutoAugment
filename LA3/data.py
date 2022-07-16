import logging

import numpy as np
import os

import math
import random
import torch
import torchvision
from PIL import Image

from torch.utils.data import SubsetRandomSampler, Sampler, Subset, ConcatDataset
import torch.distributed as dist
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from theconf import Config as C

from LA3.archive import arsaug_policy, autoaug_policy, autoaug_paper_cifar10, fa_reduced_cifar10, fa_reduced_svhn, fa_resnet50_rimagenet
from LA3.augmentations import *
from LA3.common import get_logger
from LA3.imagenet import ImageNet
from LA3.networks.efficientnet_pytorch.model import EfficientNet

import pickle as pickle

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


logger = get_logger('LA3')
logger.setLevel(logging.INFO)
_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}
_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


def get_dataloaders(dataset, batch, dataroot, split=0.15, split_idx=0, multinode=False, target_lb=-1, aug_p_path='', cls_policy_path=''):
    if 'cifar' in dataset or 'svhn' in dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
    elif 'imagenet' in dataset:
        input_size = 224
        sized_size = 256

        if 'efficientnet' in C.get()['model']['type']:
            input_size = EfficientNet.get_image_size(C.get()['model']['type'])
            sized_size = input_size + 32    # TODO
            # sized_size = int(round(input_size / 224. * 256))
            # sized_size = input_size
            logger.info('size changed to %d/%d.' % (input_size, sized_size))

        transform_train = transforms.Compose([
            EfficientNetRandomCrop(input_size),
            transforms.Resize((input_size, input_size), interpolation=Image.BICUBIC),
            # transforms.RandomResizedCrop(input_size, scale=(0.1, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(
            #     brightness=0.4,
            #     contrast=0.4,
            #     saturation=0.4,
            # ),
            transforms.ToTensor(),
            # Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            EfficientNetCenterCrop(input_size),
            transforms.Resize((input_size, input_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif 'omniglot' in dataset:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        raise ValueError('dataset=%s' % dataset)

    total_aug = augs = None
    if isinstance(C.get()['aug'], list):
        logger.debug('augmentation provided.')
        transform_train.transforms.insert(0, Augmentation(C.get()['aug']))
    else:
        logger.debug('augmentation: %s' % C.get()['aug'])
        if C.get()['aug'] == 'fa_reduced_cifar10':
            transform_train.transforms.insert(0, Augmentation(fa_reduced_cifar10()))

        elif C.get()['aug'] == 'fa_reduced_imagenet':
            transform_train.transforms.insert(0, Augmentation(fa_resnet50_rimagenet()))

        elif C.get()['aug'] == 'fa_reduced_svhn':
            transform_train.transforms.insert(0, Augmentation(fa_reduced_svhn()))

        elif C.get()['aug'] == 'pda_1500':
            with open(aug_p_path+'/top_1500_aug_policy.pkl', 'rb') as handle:
                top_k_candidate_aug_policy_lst = pickle.load(handle)

            print("top 1500 candidate_aug_policy_lst: ", top_k_candidate_aug_policy_lst)
            print(len(top_k_candidate_aug_policy_lst))

            transform_train.transforms.insert(0, PDA_Aug_Policies(top_k_candidate_aug_policy_lst))

        elif C.get()['aug'] == 'pda_1000':
            with open(aug_p_path+'/top_1000_aug_policy.pkl', 'rb') as handle:
                top_k_candidate_aug_policy_lst = pickle.load(handle)

            print("top 1000 candidate_aug_policy_lst: ", top_k_candidate_aug_policy_lst)
            print(len(top_k_candidate_aug_policy_lst))

            transform_train.transforms.insert(0, PDA_Aug_Policies(top_k_candidate_aug_policy_lst))

        elif C.get()['aug'] == 'pda_500':
            with open(aug_p_path+'/top_500_aug_policy.pkl', 'rb') as handle:
                top_k_candidate_aug_policy_lst = pickle.load(handle)

            print("top 500 candidate_aug_policy_lst: ", top_k_candidate_aug_policy_lst)
            print(len(top_k_candidate_aug_policy_lst))

            transform_train.transforms.insert(0, PDA_Aug_Policies(top_k_candidate_aug_policy_lst))

        elif C.get()['aug'] == 'pda_250':
            with open(aug_p_path+'/top_250_aug_policy.pkl', 'rb') as handle:
                top_k_candidate_aug_policy_lst = pickle.load(handle)

            print("top 250 candidate_aug_policy_lst: ", top_k_candidate_aug_policy_lst)
            print(len(top_k_candidate_aug_policy_lst))

            transform_train.transforms.insert(0, PDA_Aug_Policies(top_k_candidate_aug_policy_lst))

        elif C.get()['aug'] == 'valid_ind':
            with open(aug_p_path, 'rb') as handle:
                policies = pickle.load(handle)

            print('selected policies: ', policies)
            print(len(policies))

            transform_train.transforms.insert(0, PDA_Aug_Policies_Op(policies))

        elif C.get()['aug'] == 'arsaug':
            transform_train.transforms.insert(0, Augmentation(arsaug_policy()))
        elif C.get()['aug'] == 'autoaug_cifar10':
            transform_train.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
        elif C.get()['aug'] == 'autoaug_extend':
            transform_train.transforms.insert(0, Augmentation(autoaug_policy()))
        elif C.get()['aug'] in ['default']:
            pass
        else:
            raise ValueError('not found augmentations. %s' % C.get()['aug'])

    if C.get()['cutout'] > 0:
        transform_train.transforms.append(CutoutDefault(C.get()['cutout']))

    if dataset == 'cifar10':
        total_trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'reduced_cifar10':
        total_trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=transform_train)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=4000, random_state=0)   # 4000 trainset
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        train_idx, valid_idx = next(sss)
        targets = [total_trainset.targets[idx] for idx in train_idx]
        total_trainset = Subset(total_trainset, train_idx)
        total_trainset.targets = targets

        testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'cifar10_class':
        from LA3.dataset import CIFAR10_CLASSAUG
        total_trainset = CIFAR10_CLASSAUG(root=dataroot, policy_path=cls_policy_path, train=True)
        testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'cifar10_class_prob':
        from LA3.dataset import CIFAR10_CLASSAUG_PROB
        total_trainset = CIFAR10_CLASSAUG_PROB(root=dataroot, policy_path=cls_policy_path, train=True)
        testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'cifar10_class_op':
        from LA3.dataset import CIFAR10_CLASSAUG_OP
        total_trainset = CIFAR10_CLASSAUG_OP(root=dataroot, policy_path=cls_policy_path, train=True)
        testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100':
        total_trainset = torchvision.datasets.CIFAR100(root=dataroot, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'reduced_cifar100':
        total_trainset = torchvision.datasets.CIFAR100(root=dataroot, train=True, download=True, transform=transform_train)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=4000, random_state=0)
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        train_idx, valid_idx = next(sss)
        targets = [total_trainset.targets[idx] for idx in train_idx]
        total_trainset = Subset(total_trainset, train_idx)
        total_trainset.targets = targets

        testset = torchvision.datasets.CIFAR100(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'reduced_cifar100_4000':
        from LA3.imagenet import MemoryDatasetWrapper
        total_trainset = torchvision.datasets.CIFAR100(root=dataroot, train=True, download=True, transform=transform_train)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=46000, random_state=0)   # 4000 trainset
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        train_idx, valid_idx = next(sss)
        targets = [total_trainset.targets[idx] for idx in train_idx]
        total_trainset = Subset(total_trainset, train_idx)
        total_trainset.targets = targets

        validset = torchvision.datasets.CIFAR100(root=dataroot, train=True, download=True, transform=transform_test)
        valid_targets = [validset.targets[idx] for idx in valid_idx]
        validset = Subset(validset, valid_idx)
        validset.targets = valid_targets

        # Reduce validset to 4000 imgs
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=4000, random_state=0)
        sss_val = sss_val.split(list(range(len(validset))), validset.targets)
        rmv_val_idx, test_val_idx = next(sss_val)

        # validset = torchvision.datasets.CIFAR100(root=dataroot, train=True, download=True, transform=transform_test)
        valid_targets = [validset.targets[idx] for idx in test_val_idx]
        validset = Subset(validset, test_val_idx)
        validset.targets = valid_targets

        testset = torchvision.datasets.CIFAR100(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'reduced_cifar100_class_op':
        from LA3.dataset import CIFAR100_CLASSAUG_OP
        total_trainset = CIFAR100_CLASSAUG_OP(root=dataroot, policy_path=cls_policy_path, train=True)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=46000, random_state=0)   # 4000 trainset
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        train_idx, valid_idx = next(sss)
        targets = [total_trainset.targets[idx] for idx in train_idx]
        total_trainset = Subset(total_trainset, train_idx)
        total_trainset.targets = targets
        testset = torchvision.datasets.CIFAR100(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100_class':
        from LA3.dataset import CIFAR100_CLASSAUG
        total_trainset = CIFAR100_CLASSAUG(root=dataroot, policy_path=cls_policy_path, train=True)
        testset = torchvision.datasets.CIFAR100(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100_class_op':
        from LA3.dataset import CIFAR100_CLASSAUG_OP
        total_trainset = CIFAR100_CLASSAUG_OP(root=dataroot, policy_path=cls_policy_path, train=True)
        testset = torchvision.datasets.CIFAR100(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100_class_prob':
        from LA3.dataset import CIFAR100_CLASSAUG_PROB
        total_trainset = CIFAR100_CLASSAUG_PROB(root=dataroot, policy_path=cls_policy_path, train=True)
        testset = torchvision.datasets.CIFAR100(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100_class_filter':
        from LA3.dataset import CIFAR100_CLASSAUG_FILTER
        total_trainset = CIFAR100_CLASSAUG_FILTER(root=dataroot, policy_path=cls_policy_path, train=True)
        testset = torchvision.datasets.CIFAR100(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == '!':
        trainset = torchvision.datasets.SVHN(root=dataroot, split='train', download=True, transform=transform_train)
        extraset = torchvision.datasets.SVHN(root=dataroot, split='extra', download=True, transform=transform_train)
        total_trainset = ConcatDataset([trainset, extraset])
        testset = torchvision.datasets.SVHN(root=dataroot, split='test', download=True, transform=transform_test)
    elif dataset == 'reduced_svhn':
        total_trainset = torchvision.datasets.SVHN(root=dataroot, split='train', download=True, transform=transform_train)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=73257-1000, random_state=0)  # 1000 trainset
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        train_idx, valid_idx = next(sss)
        targets = [total_trainset.targets[idx] for idx in train_idx]
        total_trainset = Subset(total_trainset, train_idx)
        total_trainset.targets = targets

        testset = torchvision.datasets.SVHN(root=dataroot, split='test', download=True, transform=transform_test)
    elif dataset == 'imagenet':
        total_trainset = ImageNet(root=dataroot, transform=transform_train)
        testset = ImageNet(root=dataroot, split='val', transform=transform_test)

        # compatibility
        total_trainset.targets = [lb for _, lb in total_trainset.samples]

        # total_trainset = MemoryDatasetWrapper(total_trainset, num_workers=24)
        # testset = MemoryDatasetWrapper(testset, num_workers=24)
    elif dataset == 'imagenet_op':
        from LA3.dataset import ImageNet_OP
        from LA3.imagenet import MemoryDatasetWrapper
        testset = ImageNet(root=dataroot, split='val', transform=transform_test)
        total_trainset = ImageNet_OP(root=dataroot, policy_path=cls_policy_path, transform=transform_train)

        # compatibility
        total_trainset.targets = [lb for _, lb in total_trainset.samples]

        # testset = MemoryDatasetWrapper(testset, num_workers=8)
        # total_trainset = MemoryDatasetWrapper(total_trainset, num_workers=8)
    elif dataset == 'reduced_imagenet':
        # randomly chosen indices
#         idx120 = sorted(random.sample(list(range(1000)), k=120))
#         idx120 = [16, 23, 52, 57, 76, 93, 95, 96, 99, 121, 122, 128, 148, 172, 181, 189, 202, 210, 232, 238, 257, 258, 259, 277, 283, 289, 295, 304, 307, 318, 322, 331, 337, 338, 345, 350, 361, 375, 376, 381, 388, 399, 401, 408, 424, 431, 432, 440, 447, 462, 464, 472, 483, 497, 506, 512, 530, 541, 553, 554, 557, 564, 570, 584, 612, 614, 619, 626, 631, 632, 650, 657, 658, 660, 674, 675, 680, 682, 691, 695, 699, 711, 734, 736, 741, 754, 757, 764, 769, 770, 780, 781, 787, 797, 799, 811, 822, 829, 830, 835, 837, 842, 843, 845, 873, 883, 897, 900, 902, 905, 913, 920, 925, 937, 938, 940, 941, 944, 949, 959]
        total_trainset = ImageNet(root=dataroot, split='train', transform=transform_train)
        testset = ImageNet(root=dataroot, split='val', transform=transform_test)

        # compatibility
        total_trainset.targets = [lb for _, lb in total_trainset.samples]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=50000, random_state=0)  # 4000 trainset
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        train_idx, valid_idx = next(sss)
        targets = [total_trainset.targets[idx] for idx in train_idx]
        total_trainset = Subset(total_trainset, train_idx)
        total_trainset.targets = targets

    elif dataset == 'omniglot':
        from LA3.dataset import Omniglot_Mem
        total_trainset = Omniglot_Mem(dataroot, split='train', download=False, transform=transform_train)
        validset = Omniglot_Mem(dataroot, split='val', download=False, transform=transform_test)
        testset = Omniglot_Mem(dataroot, split='test', download=False, transform=transform_test)
    else:
        raise ValueError('invalid dataset name=%s' % dataset)

    if total_aug is not None and augs is not None:
        total_trainset.set_preaug(augs, total_aug)
        print('set_preaug-')

    train_sampler = None
    if split > 0.0:
        sss = StratifiedShuffleSplit(n_splits=5, test_size=split, random_state=0)
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        for _ in range(split_idx + 1):
            train_idx, valid_idx = next(sss)

        if target_lb >= 0:
            train_idx = [i for i in train_idx if total_trainset.targets[i] == target_lb]
            valid_idx = [i for i in valid_idx if total_trainset.targets[i] == target_lb]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetSampler(valid_idx)

        if multinode:
            train_sampler = torch.utils.data.distributed.DistributedSampler(Subset(total_trainset, train_idx), num_replicas=dist.get_world_size(), rank=dist.get_rank())
    else:
        valid_sampler = SubsetSampler([])

        if multinode:
            train_sampler = torch.utils.data.distributed.DistributedSampler(total_trainset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
            logger.info(f'----- dataset with DistributedSampler  {dist.get_rank()}/{dist.get_world_size()}')

    trainloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=batch, shuffle=True if train_sampler is None else False, num_workers=40, pin_memory=True,
        sampler=train_sampler, drop_last=True)
    if dataset == 'omniglot':
        validloader = torch.utils.data.DataLoader(
            validset, batch_size=batch, shuffle=False, num_workers=40, pin_memory=True, drop_last=False)
    else:
        validloader = torch.utils.data.DataLoader(
            total_trainset, batch_size=batch, shuffle=False, num_workers=40, pin_memory=True,
            sampler=valid_sampler, drop_last=False)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch, shuffle=False, num_workers=40, pin_memory=True,
        drop_last=False
    )
    return train_sampler, trainloader, validloader, testloader


def get_dataloaders_search(dataset, batch, dataroot, test_dataroot=None):
    if 'cifar' in dataset or 'svhn' in dataset:
        transform_train_unnorm = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])

        transform_train_norm_op = transforms.Compose([
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
    elif 'imagenet' in dataset:
        input_size = 224
        sized_size = 256

        if 'efficientnet' in C.get()['model']['type']:
            input_size = EfficientNet.get_image_size(C.get()['model']['type'])
            sized_size = input_size + 32    # TODO
            # sized_size = int(round(input_size / 224. * 256))
            # sized_size = input_size
            logger.info('size changed to %d/%d.' % (input_size, sized_size))

        transform_train_unnorm = transforms.Compose([
            EfficientNetRandomCrop(input_size),
            transforms.Resize((input_size, input_size), interpolation=Image.BICUBIC),
            # transforms.RandomResizedCrop(input_size, scale=(0.1, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
            ),
        ])

        transform_train_norm_op = transforms.Compose([
            Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_test = transforms.Compose([
            EfficientNetCenterCrop(input_size),
            transforms.Resize((input_size, input_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError('dataset=%s' % dataset)

    if dataset == 'reduced_cifar10':
        total_trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=4000, random_state=0)
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        train_idx, valid_idx = next(sss)
        targets = [total_trainset.targets[idx] for idx in train_idx]
        total_trainset = Subset(total_trainset, train_idx)
        total_trainset.targets = targets

        validset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=transform_test)
        valid_targets = [validset.targets[idx] for idx in valid_idx]
        validset = Subset(validset, valid_idx)
        validset.targets = valid_targets

        testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'reduced_cifar100':
        total_trainset = torchvision.datasets.CIFAR100(root=dataroot, train=True, download=True)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=4000, random_state=0)
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        train_idx, valid_idx = next(sss)
        targets = [total_trainset.targets[idx] for idx in train_idx]
        total_trainset = Subset(total_trainset, train_idx)
        total_trainset.targets = targets

        validset = torchvision.datasets.CIFAR100(root=dataroot, train=True, download=True, transform=transform_test)
        valid_targets = [validset.targets[idx] for idx in valid_idx]
        validset = Subset(validset, valid_idx)
        validset.targets = valid_targets

        testset = torchvision.datasets.CIFAR100(root=dataroot, train=False, download=True, transform=transform_test)

    elif dataset == 'reduced_imagenet':
#         idx120 = sorted(random.sample(list(range(1000)), k=120))
        total_trainset = ImageNet(root=dataroot, split='train', transform=transform_test)
        testset = ImageNet(root=dataroot, split='val', transform=transform_test)

        # compatibility
        total_trainset.targets = [lb for _, lb in total_trainset.samples]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=50000, random_state=0)  # 4000 trainset
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        train_idx, valid_idx = next(sss)
        targets = [total_trainset.targets[idx] for idx in train_idx]
        total_trainset = Subset(total_trainset, train_idx)
        total_trainset.targets = targets

        validset = ImageNet(root=dataroot, split='train', transform=transform_test)
        valid_targets = [validset.targets[idx] for idx in valid_idx]
        validset = Subset(validset, valid_idx)
        validset.targets = valid_targets

    else:
        raise ValueError('invalid dataset name=%s' % dataset)

    validloader = torch.utils.data.DataLoader(
        validset, batch_size=batch, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    return total_trainset, validloader, testloader, transform_train_unnorm, transform_train_norm_op


class custDatasetGivenClsIdxOpAug(torch.utils.data.Dataset):
    def __init__(self, data_subset, transform_train_unnorm, transform_train_norm_op, given_aug_idx_lst):
        self.data_subset = data_subset
        self.transform_train_unnorm = transform_train_unnorm
        self.transform_train_norm_op = transform_train_norm_op
        self.transform_totensor = transforms.ToTensor()

        self.given_aug_idx_lst = given_aug_idx_lst
        self.aug_policy_index_dict, _, _ = get_total_augment_policy_op(False)

    def __len__(self):
        return len(self.data_subset)

    def __getitem__(self, index):
        img, target = self.data_subset[index]

        img = self.transform_train_unnorm(img)

        self.aug_op = self.aug_policy_index_dict[self.given_aug_idx_lst[target]]

        # Randomly aug img
        for i in range(3):
            op_name = self.aug_op[i]
            op_level = random.choice(range(10))

            img = apply_augment(img, op_name, float(op_level)/10.0)

        # Norm aug image
        img = self.transform_totensor(img)
        img = self.transform_train_norm_op(img)

        return img, target


class PDA_Aug_Policies(object):
    def __init__(self, policies, probs=None):
        self.policies = policies
        self.probs = probs

    def __call__(self, img):
        # Get the aug operation (op1, mag1, op2, mag2)
        selected_aug_idx = np.random.choice(range(len(self.policies)), 1, p=self.probs)[0]
        selected_aug_op = self.policies[selected_aug_idx]

        for op_i in range(2):
            op_name = selected_aug_op[2*op_i]
            op_level = selected_aug_op[2*op_i + 1]

            if op_name is not None:
                img = apply_augment(img, op_name, float(op_level)/10.0)

        return img


class PDA_Aug_Policies_Op(object):
    def __init__(self, policies, probs=None):
        self.policies = policies
        self.probs = probs

    def __call__(self, img):
        # Get the aug operation (op1, mag1, op2, mag2)
        selected_aug_idx = np.random.choice(range(len(self.policies)), 1, p=self.probs)[0]
        selected_aug_op = self.policies[selected_aug_idx]

        for op_i in range(3):
            op_name = selected_aug_op[op_i]
            op_level = random.choice(range(10))

            if op_name is not None:
                img = apply_augment(img, op_name, float(op_level)/10.0)

        return img


class PDA_Aug_Policies_Op_Pair(object):
    def __init__(self, policies, probs=None):
        self.policies = policies
        self.probs = probs

    def __call__(self, img):
        # Get the aug operation (op1, mag1, op2, mag2)
        selected_aug_idx = np.random.choice(range(len(self.policies)), 1, p=self.probs)[0]
        selected_aug_op = self.policies[selected_aug_idx]

        for op_i in range(2):
            op_name = selected_aug_op[op_i]
            op_level = random.choice(range(10))

            if op_name is not None:
                img = apply_augment(img, op_name, float(op_level)/10.0)

        return img


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
        return img


class EfficientNetRandomCrop:
    def __init__(self, imgsize, min_covered=0.1, aspect_ratio_range=(3./4, 4./3), area_range=(0.08, 1.0), max_attempts=10):
        assert 0.0 < min_covered
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]
        assert 0 < area_range[0] <= area_range[1]
        assert 1 <= max_attempts

        self.min_covered = min_covered
        self.aspect_ratio_range = aspect_ratio_range
        self.area_range = area_range
        self.max_attempts = max_attempts
        self._fallback = EfficientNetCenterCrop(imgsize)

    def __call__(self, img):
        # https://github.com/tensorflow/tensorflow/blob/9274bcebb31322370139467039034f8ff852b004/tensorflow/core/kernels/sample_distorted_bounding_box_op.cc#L111
        original_width, original_height = img.size
        min_area = self.area_range[0] * (original_width * original_height)
        max_area = self.area_range[1] * (original_width * original_height)

        for _ in range(self.max_attempts):
            aspect_ratio = random.uniform(*self.aspect_ratio_range)
            height = int(round(math.sqrt(min_area / aspect_ratio)))
            max_height = int(round(math.sqrt(max_area / aspect_ratio)))

            if max_height * aspect_ratio > original_width:
                max_height = (original_width + 0.5 - 1e-7) / aspect_ratio
                max_height = int(max_height)
                if max_height * aspect_ratio > original_width:
                    max_height -= 1

            if max_height > original_height:
                max_height = original_height

            if height >= max_height:
                height = max_height

            height = int(round(random.uniform(height, max_height)))
            width = int(round(height * aspect_ratio))
            area = width * height

            if area < min_area or area > max_area:
                continue
            if width > original_width or height > original_height:
                continue
            if area < self.min_covered * (original_width * original_height):
                continue
            if width == original_width and height == original_height:
                return self._fallback(img)      # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/preprocessing.py#L102

            x = random.randint(0, original_width - width)
            y = random.randint(0, original_height - height)
            return img.crop((x, y, x + width, y + height))

        return self._fallback(img)


class EfficientNetCenterCrop:
    def __init__(self, imgsize):
        self.imgsize = imgsize

    def __call__(self, img):
        """Crop the given PIL Image and resize it to desired size.

        Args:
            img (PIL Image): Image to be cropped. (0,0) denotes the top left corner of the image.
            output_size (sequence or int): (height, width) of the crop box. If int,
                it is used for both directions
        Returns:
            PIL Image: Cropped image.
        """
        image_width, image_height = img.size
        image_short = min(image_width, image_height)

        crop_size = float(self.imgsize) / (self.imgsize + 32) * image_short

        crop_height, crop_width = crop_size, crop_size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))


class SubsetSampler(Sampler):
    r"""Samples elements from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)


class custClsEmbedDatasetPredictionGain(torch.utils.data.Dataset):
    def __init__(self, given_dataset):
        self.given_dataset = given_dataset

    def __len__(self):
        return len(self.given_dataset)

    def __getitem__(self, index):
        data = self.given_dataset[index]
        x_data = torch.Tensor(data[:5]).long()

        y_data = torch.Tensor(data[-1:]).float()

        return x_data, y_data


