import random

import torch
import numpy as np
import torchvision
from torchvision import transforms




class Patchwise_aug(object):
    """Randomly apply augmentations to patches from an image.

    Args:
        permute (bool): apply permutations.
        transform (bool): apply transforms.
        permutation_prob (float): permutation probability.
        transform_prob (float): transform probability
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, permute, transform, permutation_prob, transform_prob,length):
        self.permute = permute
        self.transform = transform
        self.permutation_prob = permutation_prob
        self.transform_prob = transform_prob
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: augmented image
        """
        h = img.size(1)
        w = img.size(2)
        num_patches = (w//self.length)*(h//self.length)

        patches = self.patchify_image(img,h,w)
        if self.permute and not self.transform:
            patches = self.permute_patches(patches,num_patches)
        if self.transform and not self.permute:
            patches = self.transform_patches(patches,num_patches)
        if self.transform and self.permute:
            patches = self.permute_transform_patches(patches,num_patches)

        img = self.assemble_image(h,w,patches)

        return img

    def patchify_image(self,img,h,w):
        patches = []
        for i in range(h//self.length):
            for j in range(w//self.length):
                patches.append(img[:,i*self.length:(i+1)*self.length,j*self.length:(j+1)*self.length])
        return patches

    def permute_patches(self,patches,num_patches):
        permute_boolean_array = (np.random.binomial(1,self.permutation_prob,num_patches)).tolist()
        permuted_patches = [z[0] for z in zip ([i for i in range(num_patches)],permute_boolean_array) if z[1]==1]
        permuted_patches_shuffled = permuted_patches.copy()
        random.shuffle(permuted_patches_shuffled)
        new_patches_list = []
        for i in range(num_patches):
            if i in permuted_patches:
                index = permuted_patches.index(i)
                patch_to_assign = patches[permuted_patches_shuffled[index]]
            else:
                patch_to_assign = patches[i]
            new_patches_list.append(patch_to_assign)
        return new_patches_list

    def assemble_image(self,h,w,patches):
        rows = []
        pathces_per_row = w//self.length
        for i in range(h//self.length):
            rows.append(torch.cat(patches[i*pathces_per_row:(i+1)*pathces_per_row],2))

        img = torch.cat(rows,1)
        return img
    def transform_patches(self,patches,num_patches):
        auto_augment = torchvision.transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.CIFAR10)
        transform_boolean_array = (np.random.binomial(1, self.transform_prob, num_patches)).tolist()
        for i in range(num_patches):
            if transform_boolean_array[i]:
                uint8_patch = torch.mul(patches[i],255)
                uint8_patch = uint8_patch.type(torch.uint8)
                uint8_patch = auto_augment(uint8_patch)
                uint8_patch = uint8_patch.type(torch.float)
                patches[i] = torch.div(uint8_patch,255)
        return patches

    def permute_transform_patches(self,patches,num_patches):
        auto_augment = torchvision.transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.CIFAR10)
        permute_boolean_array = (np.random.binomial(1,self.permutation_prob,num_patches)).tolist()
        permuted_patches = [z[0] for z in zip ([i for i in range(num_patches)],permute_boolean_array) if z[1]==1]
        permuted_patches_shuffled = permuted_patches.copy()
        random.shuffle(permuted_patches_shuffled)
        new_patches_list = []
        for i in range(num_patches):
            if i in permuted_patches:
                index = permuted_patches.index(i)
                patch_to_assign = patches[permuted_patches_shuffled[index]]
                if np.random.rand() >= (1 - self.transform_prob):
                    uint8_patch = torch.mul(patch_to_assign, 255)
                    uint8_patch = uint8_patch.type(torch.uint8)
                    uint8_patch = auto_augment(uint8_patch)
                    uint8_patch = uint8_patch.type(torch.float)
                    patch_to_assign = torch.div(uint8_patch, 255)
            else:
                patch_to_assign = patches[i]
            new_patches_list.append(patch_to_assign)
        return new_patches_list




