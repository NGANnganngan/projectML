from torchvision import transforms
import math
import os
import argparse
import cv2
# import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torchvision import transforms
#import videotransforms

import numpy as np

import torch.nn.functional as F
#from pytorch_i3d import InceptionI3d

# from nslt_dataset_all import NSLT as Dataset
#from datasets.nslt_dataset_all import NSLT as Dataset

import numbers
import random

class CenterCrop(object):
    """Crops the given seq Images at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        t, h, w, c = imgs.shape
        th, tw = self.size
        i = int(np.round((h - th) / 2.))
        j = int(np.round((w - tw) / 2.))

        return imgs[:, i:i+th, j:j+tw, :]


    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)




def load_rgb_frames_from_video(video_path, start, num):

    vidcap = cv2.VideoCapture(video_path)
    # vidcap = cv2.VideoCapture('/home/dxli/Desktop/dm_256.mp4')

    frames = []

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for offset in range(num):
        success, img = vidcap.read()

        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.) * 2 - 1

        frames.append(img)

    return np.asarray(frames, dtype=np.float32)

def video_to_tensor(pic):

    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))

def data_reader(video_path) :
    num_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
    start_f = 0
    start_e = num_frames
    imgs = load_rgb_frames_from_video(video_path, start_f, start_e)
    test_transforms = transforms.Compose([CenterCrop(224)])
    imgs = test_transforms(imgs)
    ret_img = video_to_tensor(imgs)
    return ret_img
