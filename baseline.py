import os
import pickle
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.color import deltaE_cie76, rgb2lab
from sklearn.cluster import KMeans


def load_img(filepath):
    """
    Loads image
    """
    img = cv2.imread(filepath)
    # convert to proper RGB scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def get_colours(img, num_colours, show_chart=False):
    """
    Get top n = num_colours colours from a given img
    """
    # resize to make processing time shorter (and it doesn't affect colours)
    mod_img = cv2.resize(img, (400, 400), interpolation = cv2.INTER_AREA)
    # reshape
    mod_img = mod_img.reshape(mod_img.shape[0]*mod_img.shape[1],3)
    clf = KMeans(n_clusters = num_colours)
    labels = clf.fit_predict(mod_img)

    counts = Counter(labels)

    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    rgb_colors = [list(ordered_colors[i]) for i in counts.keys()]

    if (show_chart):
        plt.figure(figsize = (8, 6))
        plt.pie(counts.values(), labels = rgb_colors, colors = hex_colors)
        plt.show()

    return rgb_colors

def get_files(path):
    """
    Gets all files from a given path
    """
    files = os.listdir(path)
    if ".DS_Store" in files:
        files.remove(".DS_Store")
    
    return files

def get_avg_colour(avg, n):
    """
    Calculates average colour
    """
    # get colours out
    red = avg[:,:,0]
    green = avg[:,:,1]
    blue = avg[:,:,2]
    # average them
    red = sum(red)/n
    green = sum(green)/n
    blue = sum(blue)/n
    # create final array
    final = np.array([[[r,g,b] for r,g,b in zip(red,green,blue)]])
    final = final.astype(np.uint8)

    return final

def save_list(path, lst):
    """
    Saves list as a pickle file
    """
    output = open("{}.pkl".format(path), "wb")
    pickle.dump(avg, output)
    output.close()

    return True

if __name__ == "__main__":
    # get all eras in the dataset
    path = "artist_dataset/"
    eras = get_files(path)
    for era in eras:
        artists = get_files("{}{}/".format(path,era))
        # iterate through all artists in era
        for artist in artists:
            # get all artworks from the artist
            pics = get_files("{}{}/{}/".format(path,era,artist))
            num_pics = len(pics)
            avg = []
            for i, pic in enumerate(pics):
                pic_path = "{}{}/{}/{}".format(path, era, artist,pic)
                # print progress
                print("Era: {} Artist: {} {}/{}".format(era, artist, i, num_pics))
                # load image + get top 8 colours
                img = load_img(pic_path)
                avg.append(get_colours(img,8,False))
            avg = np.array(avg)
            # ger overall average colours
            final = get_avg_colour(avg, num_pics)
            # save as pickle files
            save_list("averages/{}_{}".format(era,artist), avg)
            save_list("final/{}_{}".format(era,artist), final)
            # display
            # plt.imshow(final)
            # save figure
            plt.savefig('final/colours_{}_{}.png'.format(era,artist), bbox_inches='tight')
