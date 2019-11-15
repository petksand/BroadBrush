from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
import csv
import random
import pickle
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os
from skimage import io
from baseline import RGB2HEX, load_img, get_colours, save_list, get_files
import colorsys

_ERAS = ["Baroque", "Cubism", "Fauvism", "Impressionism", "Post"]
# _WEIGHTS = [925.0/6700.0, 542.0/6700.0, 280.0/6700.0, 3852.0/6700.0, 1101.0/6700.0]
_WEIGHTS = [0.18, 0.22, 0.19, 0.15, 0.18]
# impressionism was 0.15

def unpickle(path):
    hex = []
    rgb = open(path, "rb")
    colours = pickle.load(rgb)
    rgb.close()

    return colours

def get_pictures():
    """
    gets filepaths of all pictures from filepaths.csv
    """
    filepaths = []
    with open("filepaths.csv") as f:
        csv_reader = csv.reader(f, delimiter="\n")
        for row in csv_reader:
            filepaths.extend(row)
    return filepaths
    
def get_averages():
    """
    Unpacks all pickle files in averages folder
    """
    era_averages = {
        "Baroque": [],
        "Cubism": [],
        "Fauvism": [],
        "Impressionism": [],
        "Post": [],
    }
    avgs = get_files("averages/")
    for avg in avgs:
        # get era from filepath
        era = avg.split("_")[0]
        # unpickle the file
        lst = unpickle("averages/{}".format(avg))
        # convert to lab
        lst = np.uint8(np.asarray(lst))
        lst = rgb2lab(lst)
        # add to dictionary
        era_averages[era].append(lst)

    return era_averages


def comp_era(era, era_averages, img):
    era_min = None
    min_sum = []
    artists = era_averages[era]
    for artist in artists:
        diff = []
        artist_min = None
        artist_min_sum = 8000
        tot = 0
        for i, picture in enumerate(artist):
            if i > 100:
                break
            comp = deltaE_cie76(picture, img)
            diff.extend(comp)
            # for col in colour:
            #     comp = deltaE_cie76(col, img)
            #     diff.extend(comp)
        for array in diff:
            tot = np.sum(array[0])
            if tot < artist_min_sum:
                artist_min_sum = tot
        min_sum.extend([artist_min_sum])
            # if tot < artistmin_sum:
            #     minimum = array
            #     min_sum = tot
    min_sum = min(min_sum)
    return min_sum
        

if __name__ == "__main__":
    # filepaths of all picture files
    pictures = get_pictures()
    # shuffle them
    random.shuffle(pictures)
    averages = get_averages()
    accuracy = 0
    for i, pic in enumerate(pictures):
        info = pic.split("_")
        era = info[1].split("/")[1]
        # if era == "Impressionism":
        #     continue
        if i > 100:
            break
        print("{}/{}".format(i, len(pictures)))
        artist = info[2]
        rgb = load_img(pic)
        # get top 8 colours
        top_8 = [get_colours(rgb, 8, False)]
        top_8 = np.uint8(np.asarray(top_8))
        # convert to lab
        lab = rgb2lab([top_8])
        comparisons = [comp_era(time_p, averages, top_8) for time_p in _ERAS]
        # comparisons = [comp*weight for comp,weight in zip(comparisons, _WEIGHTS)]
        print(comparisons)
        predicted_era = _ERAS[comparisons.index(min(comparisons))]
        print("Predicted: {} Expected: {}".format(predicted_era, era))
        if predicted_era == era:
            accuracy += 1
    print("Accuracy {}/100".format(accuracy))
    




    # colours = unpickle("averages/Cubism_fernand_leger.pkl")
    # colours = np.uint8(np.asarray(colours))
    # # convert to lab
    # colours = rgb2lab(colours)

    # rgb_img = load_img("artist_dataset/Cubism/fernand_leger/fernand_leger_6.jpg")
    # rgb_img = load_img("artist_dataset/Fauvism/mary_fedden/mary_fedden_6.jpg")
    # # get top 8 colours
    # top_8 = [get_colours(rgb_img, 8, False)]
    # top_8 = np.uint8(np.asarray(top_8))
    # # convert to lab
    # hsv_img = rgb2lab([top_8])

    # diff = []
    # for i, colour in enumerate(colours):
    #     print(i)
    #     if i > 50:
    #         break
    #     for col in colour:
    #         comp = deltaE_cie76(col, hsv_img)
    #         # print(comp)
    #         diff.extend(comp)
    # min = None
    # min_sum = 8000
    # for array in diff:
    #     tot = np.sum(array[0])
    #     if tot < min_sum:
    #         min = array
    # print(np.sum(min))


