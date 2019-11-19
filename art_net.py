import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss
import torch
import random
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.preprocessing import OneHotEncoder
from torchsummary import summary

## DISTINGUISH DATASET ##
# check which artists from each era contain enough photos (50)
_ERAS_ = {"Baroque": [], "Cubism": [], "Impressionism": [], "Pop_Art": [], "Realism": [], "Renaissance": []}
_ARTISTS_ = {}

def get_files(path):
    """
    Gets all files from a given path
    """
    files = os.listdir(path)
    if ".DS_Store" in files:
        files.remove(".DS_Store")
    
    return files

eras = get_files("dataset")
for era in eras:
    artists = get_files("dataset/{}".format(era))
    for artist in artists:
        paintings = get_files("dataset/{}/{}".format(era, artist))
        if len(paintings) >= 50:
            _ERAS_[era].append(artist)
            _ARTISTS_[artist] = len(paintings)

# even out so we're evaluating a like number of artists from each era
min = 100
for key in _ERAS_:
    if len(_ERAS_[key]) < min:
        min = len(_ERAS_[key])
for key in _ERAS_:
    _ERAS_[key] = _ERAS_[key][:min]


## CREATE DATASET ##
path = 'artist_dataset/'
target_path = "artist_dataset_cropped/"
# need to collect random 300 x 300 snippet from each picture (min number of pics is 50)
# will collect ~ 300 samples from each artist (loop 6 times)
for era in _ERAS_:
    if not os.path.exists(target_path+era):
        os.makedirs(target_path+era)
    for artist in _ERAS_[era]:
        if not os.path.exists(target_path+era+"/"+artist):
            os.makedirs(target_path+era+"/"+artist)
        new_path = path + era + "/" + artist
        artworks = get_files(new_path)
        # loop to get 300 from each
        for k in range(6):
            random.shuffle(artworks)
            for i, artwork in enumerate(artworks):
                print(i)
                if i > 50:
                    # don't collect more than 50 samples from each artist
                    break
                img = plt.imread(new_path + "/" + artwork)
                x, y, z = np.shape(img)
                if x >= 300 and y >= 300:
                    x_coord = random.randint(0,x-300)
                    y_coord = random.randint(0,y-300)
                    crop = img[x_coord:x_coord+300, y_coord:y_coord+300, :]
                    save_path = target_path + era + '/' + artist + "/" + str(x_coord) + "_cropped_" + artwork
                    print(save_path)
                    plt.imsave(save_path, crop)
            
    




# ## CONSTANTS ##
# batch_size = 4
# # MSE loss
# # stochastic grad optimizer
# lr = 0.1
# epochs = 200
# # seed torch
# seed = 22
# eval_every = 150
# torch.manual_seed(seed)

# def show_image(img):
#     """
#     Displays image from dataset
#     """
#     # unnormalize?
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()

# # create composite transformation for all photos
# transform = transforms.Compose(
#     [
#         transforms.ToTensor(),
#     ]
# )

# def one_h(x):
#     lst = [0]*10
#     lst[x] = 1
#     return torch.Tensor(lst)

# # creating a training + validation set from all photos
# trainset = torchvision.datasets.ImageFolder(root='./asl_images_train', transform=transform, target_transform=one_h)
# validset = torchvision.datasets.ImageFolder(root='./asl_images_valid', transform=transform, target_transform=one_h)
# # load data
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
# validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=2)


# ## TRAINING LOOP ##
# def evaluate(model, valloader):
#     total = 0
#     corr = 0

#     for i, vbatch in enumerate(valloader):
#         feats, label = vbatch
#         feats = feats.float()
#         # run neural net
#         predictions = model(feats)
#         _, predicted = torch.max(predictions.data, 1)
#         total += label.size(0)
#         corr += (predicted == label.max(axis=1)[1]).sum().item()

#     return float(corr)/total

# def load_model(lr, data):

#     model = MultiLayerPerceptron()
#     loss_fnc = torch.nn.MSELoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=lr)

#     return model, loss_fnc, optimizer

# # for graphing
# train_vals = []
# valid_vals = []
# loss_graph = []

# model, loss_func, optimizer = load_model(lr, trainloader)
# print(summary(model, (3, 56, 56)))
# t=0
# start = time.time()
# for epoch in range(epochs):
#     accum_loss = 0
#     corr = 0
#     total = 0
#     valid_acc = 0

#     for i, batch in enumerate(trainloader,0):
#         feats, label = batch
#         # zero gradients
#         optimizer.zero_grad()
#         # run the neural net on the batch
#         predictions = model(feats)
#         # compute loss function
#         batch_loss = loss_func(input=predictions,target=label)
#         accum_loss += batch_loss.item()
#         # back prop
#         batch_loss.backward()
#         # update params
#         optimizer.step()
#         # find correct predictions
#         if (t+1) % eval_every == 0:
#             _, predicted = torch.max(predictions.data, 1)
#             total += label.size(0)
#             corr += (predicted == label.max(axis=1)[1]).sum().item()
#             # evaluate model
#             valid_acc = evaluate(model, validloader)
#             # valid_vals.append(valid_acc)
#             # train_vals.append(float(corr)/batch_size)
#             # print("Epoch: {}, Step {} | Loss: {}| Valid acc: {}".format(epoch+1, t+1, accum_loss / args.eval_every, valid_acc))
#             print("Epoch: {}, Step {} ".format(epoch+1, t+1))
#             # accum_loss = 0
#         t += 1
#     train_vals.append(float(corr)/total)
#     valid_vals.append(valid_acc)
#     # train_vals.append(float(corr)/total)
#     loss_graph.append(accum_loss)

# # save model
# torch.save(model, 'model_baseline.pt')

# # caluclate and print duration
# dur = time.time()-start
# print("It took {} seconds to execute".format(dur))
# # print post-training accuracy
# print("Train acc:{}".format(float(corr)/len(trainloader.dataset)))

# ## GRAPHING ##
# # training accuracy + validation graph
# x = list(range(len(train_vals)))
# # smooth values (mine is hella janky)
# smooth_train = ss.savgol_filter(np.ravel(train_vals),3,1)
# plt.plot(x, smooth_train, "b")
# plt.plot(x, valid_vals, "g")
# plt.title("Training and Validation Accuracy")
# plt.legend(['Training Accuracy', "Validation Accuracy"])
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")

# plt.show()