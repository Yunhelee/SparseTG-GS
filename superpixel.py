import argparse
from skimage import io
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
from torchvision.utils import save_image
from CifarNet import *
import numpy as np
import torch


# np.set_printoptions(threshold=np.sys.maxsize)

# for idx, (images, labels) in enumerate(test_loader):
#     if idx == 1:
#         save_image(images, 'C:/Users/HLM/Desktop/2.png')

def superpixel(image,n_segments):
    tempImg = image.cpu().permute(0, 2, 3, 1).data
    segments = slic(tempImg, n_segments=n_segments)
    segments = torch.from_numpy(segments)
    segments = torch.unsqueeze(segments, dim=0)
    segments = segments.repeat(3, 1, 1)
    segments = torch.unsqueeze(segments, dim=0)
    return segments
    # fig = plt.figure("Superpixels -- %d segments" % (80))
    # plt.subplot(131)
    # plt.title('image')
    # plt.imshow(image)
    # plt.subplot(132)
    # plt.title('segments')
    # plt.imshow(segments)
    # print(segments)
    # plt.subplot(133)
    # plt.title('image and segments')
    # plt.imshow(mark_boundaries(image, segments))
    # plt.show()
