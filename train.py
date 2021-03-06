
import pdb
import argparse
import numpy as np
from tqdm import tqdm
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.utils import make_grid
from torchvision import datasets, transforms
import torchvision

from util.misc import CSVLogger
from util.patchwise_aug import Patchwise_aug
from util.auto_aug import auto_aug

from model.resnet import ResNet18
from model.wide_resnet import WideResNet
import os.path
from os import path



def test(loader):
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = cnn(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    cnn.train()
    return val_acc


if __name__ == "__main__":
    model_options = ['resnet18', 'wideresnet']
    dataset_options = ['cifar10', 'cifar100']

    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument('--dataset', '-d', default='cifar10',
                    choices=dataset_options)
    parser.add_argument('--model', '-a', default='resnet18',
                    choices=model_options)
    parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 200)')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='learning rate')
    parser.add_argument('--data_augmentation', action='store_true', default=True,
                    help='augment data by flipping and cropping')
    parser.add_argument('--auto_augmentation', action='store_true', default=True,
                    help='auto augment data')
    parser.add_argument('--patch_permutation', action='store_true', default=True,
                    help='apply patchwise_permutations')
    parser.add_argument('--patch_transforms', action='store_true', default=False,
                    help='apply patchwise_transforms')
    parser.add_argument('--permutation_prob', type=float, default=0.2,
                    help='probability of permutation')
    parser.add_argument('--transform_prob', type=float, default=0.25,
                    help='probability of transforms')
    parser.add_argument('--patch_length', type=int, default=8,
                    help='length of patches')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 0)')

    args = parser.parse_args(sys.argv[1:])
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    cudnn.benchmark = True  # Should make training should go faster for large models

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    test_id = args.dataset + '_' + args.model
    if args.patch_permutation or args.patch_transform:
        test_id += '_patch-length_' + str(args.patch_length)
    if args.patch_permutation:
        test_id +=  '_patch-permutation_' + str(args.permutation_prob)

    if args.patch_transforms:
        test_id += '_patch-transforms' + str(args.transform_prob)

    print(args)

    # Image Preprocessing
    train_transform = transforms.Compose([])
    if args.data_augmentation:
        train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
    train_transform.transforms.append(transforms.ToTensor())

    if args.auto_augmentation:
        auto_augment = torchvision.transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.CIFAR10)
        train_transform.transforms.append(auto_aug(auto_augment))

    if args.patch_permutation or args.patch_transforms:
        train_transform.transforms.append(Patchwise_aug(args.patch_permutation,args.patch_transforms,args.permutation_prob
                                                    , args.transform_prob,args.patch_length))


    test_transform = transforms.Compose([
    transforms.ToTensor()])

    if args.dataset == 'cifar10':
        num_classes = 10
        train_dataset = datasets.CIFAR10(root='data/',
                                     train=True,
                                     transform=train_transform,
                                     download=True)

        test_dataset = datasets.CIFAR10(root='data/',
                                    train=False,
                                    transform=test_transform,
                                    download=True)
    elif args.dataset == 'cifar100':
        num_classes = 100
        train_dataset = datasets.CIFAR100(root='data/',
                                      train=True,
                                      transform=train_transform,
                                      download=True)

        test_dataset = datasets.CIFAR100(root='data/',
                                     train=False,
                                     transform=test_transform,
                                     download=True)


    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)

    if args.model == 'resnet18':
        cnn = ResNet18(num_classes=num_classes)
    elif args.model == 'wideresnet':
        cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                         dropRate=0.3)


    cnn = cnn.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)



    scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)

    filename = 'logs/' + test_id + '.csv'
    if not path.exists('logs'):
        os.mkdir('logs')
    csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=filename)
    max_test_acc = 0
    for epoch in range(args.epochs):

        xentropy_loss_avg = 0.
        correct = 0.
        total = 0.

        progress_bar = tqdm(train_loader)
        for i, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))

            images = images.cuda()
            labels = labels.cuda()

            cnn.zero_grad()
            pred = cnn(images)

            xentropy_loss = criterion(pred, labels)
            xentropy_loss.backward()
            cnn_optimizer.step()

            xentropy_loss_avg += xentropy_loss.item()

            # Calculate running average of accuracy
            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels.data).sum().item()
            accuracy = correct / total

            progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)


        test_acc = test(test_loader)
        if test_acc > max_test_acc:
            max_test_acc = test_acc
        tqdm.write('test_acc: %.3f' % (test_acc))

        scheduler.step()     # Use this line for PyTorch >=1.4

        row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
        csv_logger.writerow(row)
    row = {'epoch': str(args.epochs), 'train_acc': str(max_test_acc), 'test_acc': str(max_test_acc)}
    csv_logger.writerow(row)
    torch.save(cnn.state_dict(), 'checkpoints/' + test_id + '.pt')
    csv_logger.close()




