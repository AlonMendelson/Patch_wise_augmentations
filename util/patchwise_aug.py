import random

import torch
import numpy as np



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

        permuted_patches = self.permute_patches(patches,num_patches)

        img = self.assemble_image(h,w,permuted_patches)

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



