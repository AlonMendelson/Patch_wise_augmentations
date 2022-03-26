import numpy as np
import torch
import torchvision
from util.patchwise_aug import Patchwise_aug
from util.auto_aug import auto_aug
from util.cutout import Cutout


def get_loader(batch_size, num_workers, use_gpu):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    auto_augment = torchvision.transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.CIFAR10)

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        auto_aug(auto_augment),
        #Patchwise_aug(True, False, 0.2, 0, 8),
        Cutout(1,16)
        #torchvision.transforms.Normalize(mean, std),

    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize(mean, std),
    ])

    dataset_dir = 'data/'
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
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_gpu,
        drop_last=False,
    )
    return train_loader, test_loader
