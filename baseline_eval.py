from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os
from skimage import io
from baseline import RGB2HEX, load_img, get_colours
import colorsys


def unpickle(path):
    hex = []
    rgb = open(path, "rb")
    colours = pickle.load(rgb)
    rgb.close()

    return colours


if __name__ == "__main__":
    colours = unpickle("averages/Cubism_fernand_leger.pkl")
    colours = np.uint8(np.array(colours))
    # convert to lab
    colours = rgb2lab(colours)

    rgb_img = load_img("artist_dataset/Cubism/fernand_leger/fernand_leger_6.jpg")
    # convert to lab
    hsv_img = rgb2lab([rgb_img])

    diff = []
    for i, colour in enumerate(colours):
        print(i)
        if i > 50:
            break
        for col in colour:
            comp = deltaE_cie76(col, hsv_img)
            # print(comp)
            diff.extend(comp)
    print(min(diff))