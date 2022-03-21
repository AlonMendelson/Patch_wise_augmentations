import random

import torch
import numpy as np
import torchvision
from torchvision import transforms




class auto_aug(object):
    """Auto augment an image

    Args:
        augment (auto-augment): apply auto tugmentation.
    """
    def __init__(self, auto_augment):
        self.auto_augment = auto_augment

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: augmented image
        """

        uint8_img = torch.mul(img, 255)
        uint8_img = uint8_img.type(torch.uint8)
        uint8_img = self.auto_augment(uint8_img)
        img = uint8_img.type(torch.float)
        img = torch.div(img, 255)
        #img = self.auto_augment(img)

        return img






