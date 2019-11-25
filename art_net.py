import os
<<<<<<< HEAD
import pickle
import random
import time
from shutil import copyfile, rmtree
=======
import time
>>>>>>> 020ee17b399f810c775b8fac0292689ed79a9a76

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss
import torch
<<<<<<< HEAD
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torchsummary import summary

from art_model import ArtNet
=======
import random
import pickle
import torch.nn.functional as F
# from art_model import ArtNet
import torchvision
import torchvision.transforms as transforms
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from torchsummary import summary
from shutil import copyfile, rmtree
>>>>>>> 020ee17b399f810c775b8fac0292689ed79a9a76

# gor Google CoLab
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print('Using CUDA')

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
batch_size = 32
lr = 0.001
epochs = 25
# seed torch
seed = 22
eval_every = 150
torch.manual_seed(seed)


# create composite transformation for all photos
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# creating a training + validation set from all photos
trainset = torchvision.datasets.ImageFolder(root='artist_dataset_train', transform=transform)
validset = torchvision.datasets.ImageFolder(root='artist_dataset_valid', transform=transform)
testset = torchvision.datasets.ImageFolder(root='artist_dataset_test', transform=transform)
trainset, overfitset = train_test_split(trainset, test_size=0.001, random_state=seed)
# load data
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=0)
overfitloader = torch.utils.data.DataLoader(overfitset, batch_size=batch_size, shuffle=True, num_workers=0)


## TRAINING LOOP ##

def evaluate(net, loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct/len(loader.dataset)

def load_model(lr):

    model = ArtNet(2, 32, 50, True, 24)
    loss_fnc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    return model, loss_fnc, optimizer

# for graphing
train_vals = []
valid_vals = []
loss_graph = []

model, loss_func, optimizer = load_model(lr)
t=0
start = time.time()
for epoch in range(epochs):
    accum_loss = 0
    corr = 0
    total = 0
    valid_acc = 0

    for i, batch in enumerate(trainloader,0):
        feats, label = batch
        if torch.cuda.is_available():
                feats, label = feats.to(device), label.to(device)
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
            corr += (predicted == label).sum().item()
            # evaluate model
            valid_acc = evaluate(model, validloader)
            # print("Epoch: {}, Step {} ".format(epoch+1, t+1))
        t += 1
    train_vals.append(float(corr)/total)
    valid_vals.append(valid_acc)
    print("Train accuracy: ", float(corr)/total)
    print("Valid accuracy: ",valid_acc)
    print("EPOCH {}".format(epoch))
    # loss_graph.append(accum_loss)

# save model
torch.save(model, 'model_baseline.pt')

# caluclate and print duration
dur = time.time()-start
print("It took {} seconds to execute".format(dur))
# print post-training accuracy + validation
print("Train acc:{}".format(max(train_vals)))
print("Valid acc:{}".format(max(valid_vals)))

## GRAPHING ##
# training accuracy + validation graph
x = list(range(len(train_vals)))
# smooth values (mine is hella janky)
smooth_train = ss.savgol_filter(np.ravel(train_vals),3,1)
plt.plot(x, smooth_train, "b")
plt.plot(x, valid_vals, "g")
plt.title("Training and Validation Accuracy")
plt.legend(['Training Accuracy', "Validation Accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("train_valid_acc.png", dpi=300)
plt.show()