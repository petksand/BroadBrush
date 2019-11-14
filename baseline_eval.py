from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os
from skimage import io


def unpickle(path):
    file = open(path, "rb")
    file.close()
