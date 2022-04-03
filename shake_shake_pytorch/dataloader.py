import numpy as np
import torch
import torchvision
from util.patchwise_aug import Patchwise_aug
from util.auto_aug import auto_aug
from util.cutout import Cutout


def get_loader(run_config,model_config, optim_config,data_config):

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor()
    ])
    if model_config['auto_augmentation']:
        auto_augment = torchvision.transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.CIFAR10)
        train_transform.transforms.append(auto_aug(auto_augment))

    if model_config['patch_permutation'] or model_config['patch_transforms']:
        train_transform.transforms.append(
            Patchwise_aug(model_config['patch_permutation'], model_config['patch_transforms'], model_config['permutation_prob']
                          , model_config['transform_prob'], model_config['patch_length']))
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    dataset_dir = 'data/'
    if data_config['dataset'] == 'CIFAR10':
        train_dataset = torchvision.datasets.CIFAR10(dataset_dir,
                                                      train=True,
                                                      transform=train_transform,
                                                      download=True)
        test_dataset = torchvision.datasets.CIFAR10(dataset_dir,
                                                     train=False,
                                                     transform=test_transform,
                                                     download=True)
    else:
        train_dataset = torchvision.datasets.CIFAR100(dataset_dir,
                                                      train=True,
                                                      transform=train_transform,
                                                      download=True)
        test_dataset = torchvision.datasets.CIFAR100(dataset_dir,
                                                     train=False,
                                                     transform=test_transform,
                                                     download=True)


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=optim_config['batch_size'],
        shuffle=True,
        num_workers=run_config['num_workers'],
        pin_memory=run_config['device'] != 'cpu',
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=optim_config['batch_size'],
        num_workers=run_config['num_workers'],
        shuffle=False,
        pin_memory=run_config['device'] != 'cpu',
        drop_last=False,
    )
    return train_loader, test_loader
