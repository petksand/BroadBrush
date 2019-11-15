from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os
from skimage import io


# load test image
def load_img(filepath):
    if not filepath:
        raise Exception
    img = cv2.imread(filepath)
    # convert to proper RGB scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_colours(img, num_colours, show_chart=False):
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
    # hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [list(ordered_colors[i]) for i in counts.keys()]

    if (show_chart):
        plt.figure(figsize = (8, 6))
        plt.pie(counts.values(), labels = rgb_colors, colors = hex_colors)
        plt.show()

    return rgb_colors
    # return [rgb_colors, counts.values()]

def get_files(path):
    files = os.listdir(path)
    if ".DS_Store" in files:
        files.remove(".DS_Store")
    return files

def get_avg_colour(avg, n):
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
    output = open("{}.pkl".format(path), "wb")
    pickle.dump(avg, output)
    output.close()

    return True

if __name__ == "__main__":
    path = "artist_dataset/"
    eras = get_files(path)
    for era in eras:
        if era != "Post_Impressionism":
            continue
        artists = get_files("{}{}/".format(path,era))
        for artist in artists:
            if artist == "emily_carr" or artist == "paul_cezanne":
                continue
            pics = get_files("{}{}/{}/".format(path,era,artist))
            num_pics = len(pics)
            avg = []
            for i, pic in enumerate(pics):
                pic_path = "{}{}/{}/{}".format(path, era, artist,pic)
                # print progress
                print("Era: {} Artist: {} {}/{}".format(era, artist, i, num_pics))
                print(pic_path)
                img = load_img(pic_path)
                avg.append(get_colours(img,8,False))
            avg = np.array(avg)
            final = get_avg_colour(avg, num_pics)
            save_list("averages/{}_{}".format(era,artist), avg)
            save_list("final/{}_{}".format(era,artist), final)
            plt.imshow(final)
            plt.savefig('final/colours_{}_{}.png'.format(era,artist), bbox_inches='tight')

    # plt.show()
    # print(final)
