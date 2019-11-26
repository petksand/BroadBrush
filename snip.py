import os
import numpy as np
import matplotlib.pyplot as plt
import random
import shutil


path = "/Users/stephenbrade/ECE324/BroadBrush/dataset"
target = "/Users/stephenbrade/ECE324/BroadBrush/dataset_era_535"
os.mkdir("/Users/stephenbrade/ECE324/BroadBrush/dataset_era_535")
os.mkdir("/Users/stephenbrade/ECE324/BroadBrush/dataset_era_535/Fauvism")
os.mkdir("/Users/stephenbrade/ECE324/BroadBrush/dataset_era_535/Cubism")
os.mkdir("/Users/stephenbrade/ECE324/BroadBrush/dataset_era_535/Pop_Art")
os.mkdir("/Users/stephenbrade/ECE324/BroadBrush/dataset_era_535/Post_Impressionism")
os.mkdir("/Users/stephenbrade/ECE324/BroadBrush/dataset_era_535/Impressionism")
os.mkdir("/Users/stephenbrade/ECE324/BroadBrush/dataset_era_535/Renaissance")
os.mkdir("/Users/stephenbrade/ECE324/BroadBrush/dataset_era_535/Baroque")

dirs = os.listdir(path)
min_m = np.inf
min_n = np.inf
m_path = ""
n_path = ""
target_path = ""
era_sizes = []
a = False
former_current = ""
current = ""
f_count = 0
c_count = 0
i_count = 0
r_count = 0
b_count = 0
p_count = 0
pi_count = 0
for dirpath, dirnames, filenames in os.walk(path):
    for dirpath2, dirnames2, filenames2 in os.walk(dirpath):
        for file in filenames:
            if file != ".DS_Store":
                img = plt.imread(dirpath + "/" + file)
                m, n, o = np.shape(img)
                if "Fauvism" in dirpath:
                    if f_count < 535 and m > 300 and n > 300:
                        target_path = target + "/Fauvism"
                        f_count += 1
                    else:
                        target_path = ""
                elif "Cubism" in dirpath:
                    if c_count < 535 and m > 300 and n > 300:
                        target_path = target + "/Cubism"
                        c_count += 1
                    else:
                        target_path = ""
                elif "Impressionism" in dirpath and "Post" not in dirpath:
                    if i_count < 535 and m > 300 and n > 300:
                        target_path = target + "/Impressionism"
                        i_count += 1
                    else:
                        target_path = ""
                elif "Renaissance" in dirpath:
                    if r_count < 535 and m > 300 and n > 300:
                        target_path = target + "/Renaissance"
                        r_count += 1
                    else:
                        target_path = ""
                elif "Baroque" in dirpath:
                    if b_count < 535 and m > 300 and n > 300:
                        target_path = target + "/Baroque"
                        b_count += 1
                    else:
                        target_path = ""
                elif "Pop_Art" in dirpath:
                    if p_count < 535 and m > 300 and n > 300:
                        target_path = target + "/Pop_Art"
                        p_count += 1
                    else:
                        target_path = ""
                elif "Post_Impressionism" in dirpath:
                    if pi_count < 535 and m > 300 and n > 300:
                        target_path = target + "/Post_Impressionism"
                        pi_count += 1
                    else:
                        target_path = ""
                if target_path != "":
                    print(file)
                    span_m = m - 300
                    span_n = n - 300
                    rand_m = random.randint(0, span_m)
                    rand_n = random.randint(0, span_n)
                    img_crop = img[rand_m:299+rand_m, rand_n:299+rand_n, :]
                    plt.imsave(target_path+"/"+file, img_crop)
for dirpath, dirnames, filenames in os.walk(target):
    era_sizes += [len(os.listdir(dirpath))]
era_sizes = era_sizes[1:]
print(era_sizes)

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
