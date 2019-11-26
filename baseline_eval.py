import csv
import os
import pickle
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import deltaE_cie76, rgb2lab

from baseline import RGB2HEX, get_colours, get_files, load_img, save_list

_ERAS = ["Baroque", "Cubism", "Fauvism", "Impressionism", "Post"]
# _WEIGHTS = [925.0/6700.0, 542.0/6700.0, 280.0/6700.0, 3852.0/6700.0, 1101.0/6700.0]

def unpickle(path):
    """
    Unpickles a file
    """
    rgb = open(path, "rb")
    colours = pickle.load(rgb)
    rgb.close()

    return colours

def get_pictures():
    """
    Gets filepaths of all pictures from filepaths.csv
    """
    filepaths = []
    # open + read file
    with open("filepaths.csv") as f:
        csv_reader = csv.reader(f, delimiter="\n")
        for row in csv_reader:
            filepaths.extend(row)

    return filepaths
    
def get_averages():
    """
    Unpacks all pickle files in averages folder, sorts them into era 
    """
    # dictionary to store all top colours
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
    """
    Compares img with all pictures in the given era. Returns smallest distance

    - Gets era, goes through each artist, goes through each of the artists artworks
    - Collects smallest value (closes similarity) per artist in artist_min_sum var
    - Collects min from all artists in min_sum
    - Returns the min
    """
    # variable to hold minimum values for all artists in era
    min_sum = []
    # get artists from given era
    artists = era_averages[era]
    for artist in artists:
        diff = []
        # tracks similarity between images
        artist_min_sum = 8000
        # go through each artwork
        for i, picture in enumerate(artist):
            if i > 100:
                # don't go through more than 100 per artist (for balancing)
                break
            # compare the given image and the artist's artwork
            comp = deltaE_cie76(picture, img)
            diff.extend(comp)
        tot = 0
        for array in diff:
            # sum all similarity vectors per artist + normalize
            tot += np.sum(array)/max(array)
        # get average similarity to the artist
        tot = tot/len(diff)
        min_sum.extend([tot])
    # average all values
    min_sum = sum(min_sum)/len(min_sum)
    return min_sum
        

if __name__ == "__main__":
    # filepaths of all picture files
    pictures = get_pictures()
    # shuffle them
    random.shuffle(pictures)
    # get top colours from all pictures
    averages = get_averages()
    accuracy = 0
    for i, pic in enumerate(pictures):
        # get info from pic filepath string
        info = pic.split("_")
        era = info[1].split("/")[1]
        if i > 100:
            # only going through 100 iterations (not 6700)
            break
        print("{}/{}".format(i, len(pictures)))
        rgb = load_img(pic)
        # get top 8 colours
        top_8 = [get_colours(rgb, 8, False)]
        top_8 = np.uint8(np.asarray(top_8))
        # convert to lab
        lab = rgb2lab([top_8])
        comparisons = [comp_era(time_p, averages, top_8) for time_p in _ERAS]
        print(comparisons)
        predicted_era = _ERAS[comparisons.index(min(comparisons))]
        print("Predicted: {} Expected: {}".format(predicted_era, era))
        if predicted_era == era:
            accuracy += 1
    print("Accuracy {}/100".format(accuracy))
