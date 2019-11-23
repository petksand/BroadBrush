import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss
import torch
import random
import pickle
import torch.nn.functional as F
from art_model import ArtNet
import torchvision
import torchvision.transforms as transforms
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from torchsummary import summary
from shutil import copyfile, rmtree

def get_files(path):
    """
    Gets all files from a given path
    """
    files = os.listdir(path)
    if ".DS_Store" in files:
        files.remove(".DS_Store")
    
    return files

###############################################################################################################
##################################### BUILDING DATA SET #######################################################
###############################################################################################################

# ## DISTINGUISH DATASET ##
_ERAS_ = {"Baroque": [], "Cubism": [], "Impressionism": [], "Pop_Art": [], "Realism": [], "Renaissance": []}
# _ARTISTS_ = {}
# _ARTWORKS_ = {}


# # check which artists from each era contain enough photos (50)
# eras = get_files("dataset")
# for era in eras:
#     artists = get_files("dataset/{}".format(era))
#     for artist in artists:
#         paintings = get_files("dataset/{}/{}".format(era, artist))
#         if len(paintings) >= 50:
#             _ERAS_[era].append(artist)
#             _ARTISTS_[artist] = len(paintings)

# # even out so we're evaluating a like number of artists from each era
# min = 100
# for key in _ERAS_:
#     if len(_ERAS_[key]) < min:
#         min = len(_ERAS_[key])
# for key in _ERAS_:
#     _ERAS_[key] = _ERAS_[key][:min]


# ## CREATE DATASET ##
# path = 'artist_dataset/'
# target_path = "artist_dataset_cropped/"
# # need to collect random 300 x 300 snippet from each picture (min number of pics is 50)
# # will collect ~ 300 samples from each artist (loop 6 times)
# for era in _ERAS_:
#     if not os.path.exists(target_path+era):
#         os.makedirs(target_path+era)
#     for artist in _ERAS_[era]:
#         if not os.path.exists(target_path+era+"/"+artist):
#             os.makedirs(target_path+era+"/"+artist)
#         new_path = path + era + "/" + artist
#         artworks = get_files(new_path)
#         # loop to get 6 from each
#         for k in range(6):
#             random.shuffle(artworks)
#             for i, artwork in enumerate(artworks):
#                 # if i > 50:
#                 #     # don't collect more than 50 samples from each artist
#                 #     break
#                 # keep track of artworks
#                 if _ARTWORKS_.get(artist) == None:
#                     _ARTWORKS_[artist] = [artwork]
#                 else:
#                     _ARTWORKS_[artist].append(artwork)
#                 img = plt.imread(new_path + "/" + artwork)
#                 x, y, z = np.shape(img)
#                 if x >= 300 and y >= 300:
#                     x_coord = random.randint(0,x-300)
#                     y_coord = random.randint(0,y-300)
#                     crop = img[x_coord:x_coord+300, y_coord:y_coord+300, :]
#                     save_path = target_path + era + '/' + artist + "/" + str(x_coord) + "_cropped_" + artwork
#                     print(save_path)
#                     plt.imsave(save_path, crop)

# # dump _ARTWORKS_ into pickle for next step
# output = open("ARTWORKS.pkl".format(path), "wb")
# pickle.dump(_ARTWORKS_, output)
# output.close()

###############################################################################################################
###############################################################################################################
            
# SPLITTING DATASET
# we'll split the minimum 300 into 75% training, 15% validation, 10% testing
# this means 225 training, 45 validation, 30 testing
# each artwork has ~6 duplicates therefore, for 225 training, pick 37 art pieces
# each artwork has ~6 duplicates therefore, for 45 validation, pick 7 art pieces
# each artwork has ~6 duplicates therefore, for 30 testing, pick 5 art pieces

# # open pickle
# file = open("ARTWORKS.pkl", "rb")
# _ARTWORKS_ = pickle.load(file)
# file.close()

# # iterate through cropped images
# path = "artist_dataset_cropped/"
# train_path = "artist_dataset_train/"
# valid_path = "artist_dataset_valid/"
# test_path = "artist_dataset_test/"
# for era in _ERAS_:
#     artists = get_files(path+era)
#     for artist in artists:
#         # make directories for each artist
#         # if not os.path.exists(path+era+"/"+artist+"/"+"training"):
#         #     os.makedirs(path+era+"/"+artist+"/"+"training")
#         # if not os.path.exists(path+era+"/"+artist+"/"+"validation"):
#         #     os.makedirs(path+era+"/"+artist+"/"+"validation")
#         # if not os.path.exists(path+era+"/"+artist+"/"+"testing"):
#         #     os.makedirs(path+era+"/"+artist+"/"+"testing")
#         src_path = path + era + "/" + artist + "/"

#         if not os.path.exists(train_path+artist):
#             os.makedirs(train_path+artist)
#         if not os.path.exists(valid_path+artist):
#             os.makedirs(valid_path+artist)
#         if not os.path.exists(test_path+artist):
#             os.makedirs(test_path+artist)

#         train_num = 0
#         valid_num = 0
#         test_num = 0
#         # remove duplicates with set()
#         for work in set(_ARTWORKS_[artist]):
#             in_files = get_files(src_path)
#             matches = [file for file in in_files if work in file]

#             # copy into desired folder
#             if train_num <= 255:
#                 # [copyfile(src_path+file, src_path+"training/"+ file) for file in matches]
#                 [copyfile(src_path+file, train_path+ artist + "/"+ file) for file in matches]
#                 train_num += len(matches)
#             elif valid_num <= 45:
#                 # [copyfile(src_path+file, src_path+"validation/"+ file) for file in matches]
#                 [copyfile(src_path+file, valid_path+ artist + "/"+ file) for file in matches]
#                 valid_num += len(matches)
#             elif test_num <= 30:
#                 # [copyfile(src_path+file, src_path+"testing/"+ file) for file in matches]
#                 [copyfile(src_path+file, test_path+ artist + "/"+ file) for file in matches]
#                 test_num += len(matches)

# def teardown():
#     # iterate through cropped images
#     path = "artist_dataset_cropped/"
#     for era in _ERAS_:
#         artists = get_files(path+era)
#         for artist in artists:
#             # make directories for each artist
#             if os.path.exists(path+era+"/"+artist+"/"+"training"):
#                 rmtree(path+era+"/"+artist+"/"+"training")
#             if os.path.exists(path+era+"/"+artist+"/"+"validation"):
#                 rmtree(path+era+"/"+artist+"/"+"validation")
#             if os.path.exists(path+era+"/"+artist+"/"+"testing"):
#                 rmtree(path+era+"/"+artist+"/"+"testing")
# teardown()

###############################################################################################################
##################################### LOADING DATA SET #######################################################
###############################################################################################################

path = "artist_dataset_cropped/"

# create composite transformation for all photos
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

def one_h(x):
    lst = [0]*10
    lst[x] = 1
    return torch.Tensor(lst)


## CONSTANTS ##
batch_size = 4
# MSE loss
# stochastic grad optimizer
lr = 0.1
epochs = 100
# seed torch
seed = 22
eval_every = 1
torch.manual_seed(seed)

def show_image(img):
    """
    Displays image from dataset
    """
    # unnormalize?
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# create composite transformation for all photos
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

def one_h(x):
    lst = [0]*10
    lst[x] = 1
    return torch.Tensor(lst)

# creating a training + validation set from all photos
trainset = torchvision.datasets.ImageFolder(root='artist_dataset_train', transform=transform)
validset = torchvision.datasets.ImageFolder(root='artist_dataset_valid', transform=transform)
testset = torchvision.datasets.ImageFolder(root='artist_dataset_test', transform=transform)
trainset, overfitset = train_test_split(trainset, test_size=0.001, random_state=seed)
# load data
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=2)
overfitloader = torch.utils.data.DataLoader(overfitset, batch_size=batch_size, shuffle=True, num_workers=2)


## TRAINING LOOP ##

def evaluate_ce(loader, net):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i,data in enumerate(loader,0):
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct/total

def evaluate(model, valloader):
    total = 0
    corr = 0

    for i, vbatch in enumerate(valloader):
        feats, label = vbatch
        feats = feats.float()
        # run neural net
        predictions = model(feats)
        _, predicted = torch.max(predictions.data, 1)
        total += label.size(0)
        corr += (predicted == label.max(axis=1)[1]).sum().item()

    return float(corr)/total

def load_model(lr, data):

    model = ArtNet(2, 32, 15, True, 24)
    loss_fnc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    return model, loss_fnc, optimizer

# for graphing
train_vals = []
valid_vals = []
loss_graph = []

model, loss_func, optimizer = load_model(lr, trainloader)
t=0
start = time.time()
for epoch in range(epochs):
    accum_loss = 0
    corr = 0
    total = 0
    valid_acc = 0

    for i, batch in enumerate(overfitloader,0):
        feats, label = batch
        # zero gradients
        optimizer.zero_grad()
        # run the neural net on the batch
        predictions = model(feats)
        # compute loss function
        batch_loss = loss_func(input=predictions,target=label)
        accum_loss += batch_loss.item()
        # back prop
        batch_loss.backward()
        # update params
        optimizer.step()
        # find correct predictions
        if (t+1) % eval_every == 0:
            _, predicted = torch.max(predictions.data, 1)
            total += label.size(0)
            # corr += (predicted == label.max(axis=1)[1]).sum().item()
            corr += (predicted == label).sum().item()
            # # evaluate model
            # valid_acc = evaluate(model, validloader)
            print("Epoch: {}, Step {} ".format(epoch+1, t+1))
        t += 1
    train_vals.append(float(corr)/total)
    print(float(corr)/total)
    # valid_vals.append(valid_acc)
    # loss_graph.append(accum_loss)

# save model
torch.save(model, 'model_baseline.pt')

# caluclate and print duration
dur = time.time()-start
print("It took {} seconds to execute".format(dur))
# print post-training accuracy
print("Train acc:{}".format(float(corr)/len(trainloader.dataset)))

## GRAPHING ##
# training accuracy + validation graph
x = list(range(len(train_vals)))
# smooth values (mine is hella janky)
smooth_train = ss.savgol_filter(np.ravel(train_vals),3,1)
plt.plot(x, smooth_train, "b")
# plt.plot(x, valid_vals, "g")
plt.title("Training and Validation Accuracy")
plt.legend(['Training Accuracy', "Validation Accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.show()
