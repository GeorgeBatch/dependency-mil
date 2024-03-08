"""
Functions for filtering out background pixels in the WSI and creating tissue masks.
"""

import warnings

import numpy as np
from skimage.color import rgb2hsv
from skimage.util import img_as_ubyte

warnings.simplefilter('ignore')


def thres_saturation(img, t=15):
    # saturation threshold: typical t = 15
    img = rgb2hsv(img)
    h, w, c = img.shape
    sat_img = img[:, :, 1]
    sat_img = img_as_ubyte(sat_img)
    ave_sat = np.sum(sat_img) / (h * w)
    return ave_sat >= t
