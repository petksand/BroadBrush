import os
import numpy as np
import matplotlib.pyplot as plt
import random
import shutil


path = "/Users/stephenbrade/ECE324/BroadBrush/dataset"
target = "/Users/stephenbrade/ECE324/BroadBrush/dataset_crop_random"
os.mkdir("/Users/stephenbrade/ECE324/BroadBrush/dataset_crop_random")
os.mkdir("/Users/stephenbrade/ECE324/BroadBrush/dataset_crop_random/Fauvism")
os.mkdir("/Users/stephenbrade/ECE324/BroadBrush/dataset_crop_random/Cubism")
os.mkdir("/Users/stephenbrade/ECE324/BroadBrush/dataset_crop_random/Impressionism")
os.mkdir("/Users/stephenbrade/ECE324/BroadBrush/dataset_crop_random/Renaissance")
os.mkdir("/Users/stephenbrade/ECE324/BroadBrush/dataset_crop_random/Baroque")

dirs = os.listdir(path)
min_m = np.inf
min_n = np.inf
m_path = ""
n_path = ""
for dirpath, dirnames, filenames in os.walk(path):
   #print(dirpath)
    for dirpath2, dirnames2, filenames2 in os.walk(dirpath):
        for file in filenames:
            if file != ".DS_Store":
                target_path = ""
                if "Fauvism" in dirpath:
                    target_path = target + "/Fauvism"
                elif "Cubism" in dirpath:
                    target_path = target + "/Cubism"
                elif "Impressionism" in dirpath:
                    target_path = target + "/Impressionism"
                elif "Renaissance" in dirpath:
                    target_path = target + "/Renaissance"
                elif "Baroque" in dirpath:
                    target_path = target + "/Baroque"
                if target_path != "":
                    img = plt.imread(dirpath + "/" + file)
                    m, n, o = np.shape(img)
                    span_m = m - 178
                    span_n = n - 178
                    rand_m = random.randint(0, span_m)
                    rand_n = random.randint(0, span_n)
                    img_crop = img[rand_m:177+rand_m, rand_n:177+rand_n, :]
                    plt.imsave(target_path+"/"+file, img_crop)
                print(target_path)
#                 # if m < min_m:
#                 #     min_m = m
#                 #     m_path = dirpath + "/" + file
#                 #     print(m_path)
#                 # if n < min_n:
#                 #     min_n = n
#                 #     n_path = dirpath + "/" + file
#                 #     print(n_path)

# # img = plt.imread(m_path)
# # plt.imshow(img)
# # plt.show()
# #
# # img = plt.imread(n_path)
# # plt.imshow(img)
# # plt.show()
#
#
# print(min_m, min_n)
                # plt.imshow(img)
                # plt.show()

                # if "Fauvism" in dirpath:
                #     target_path = target + "/Fauvism"
                # elif "Cubism" in dirpath:
                #     target_path = target + "/Cubism"
                # elif "Impressionism" in dirpath:
                #     target_path = target + "/Impressionism"
                # elif "Renaissance" in dirpath:
                #     target_path = target + "/Renaissance"
                # elif "Baroque" in dirpath:
                #     target_path = target + "/Baroque"

    # for item in dir:
    #     print(item)
    # if os.path.isfile(path+item):
    #     # m, n, o = np.shape(img)
    #     img = plt.imread("pink_lake.png")
    #     # img_crop = img[0:int(m / 2), 0:n, :]
    #     # plt.imsave("img_crop.png", img_crop)


# plt.imshow(img)
# plt.show()
