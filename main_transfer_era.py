from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchsummary import summary
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.model_selection import train_test_split

########################################################################################################################

# Calculates accuracy
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
# Calculates loss

def eval_loss(loader, net):
    net.eval()
    total = 0
    total_loss = 0
    net.eval()
    for i, data in enumerate(loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(input=outputs, target=labels).mean()
        total_loss += loss
        total += 1
    return total_loss/total

# Generates figures
def make_figs(epochs, net, total_loss, total_valid_loss, total_train_acc, total_valid_acc, save, name):
    summary(net, input_size=(3, 299, 299))
    print(max(total_valid_acc))
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(range(0, epochs), total_loss, label='Training Data')
    ax.plot(range(0, epochs), total_valid_loss, label='Validation Data')
    plt.title('Transfer Learning with : Era Prediction Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    ax.legend()
    if save:
        plt.savefig(name + "_loss.png")
    plt.show()

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(range(0, epochs), total_train_acc,
            label='Training Data')
    ax.plot(range(0, epochs), total_valid_acc,
            label='Validation Data')
    plt.ylim(0, 1)
    plt.title('Transfer Learning: Era Prediction Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    ax.legend()
    if save:
        plt.savefig(name + "_acc.png")
    plt.show()


def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, epochs, device):
    total_train_acc = []
    total_valid_acc = []
    total_loss = []
    total_valid_loss = []
    true = []
    predict = []
    start = time.time()
    max_valid_acc = 0
    for e in range(epochs):  # loop over the dataset multiple times
        accum_loss = 0.0
        tot_corr = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.to(device), labels.to(device)
            # print(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            # print(len(outputs), len(labels))
            loss = criterion(input=outputs, target=labels).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # print statistics
            accum_loss += loss
        valid_loss = eval_loss(valid_loader, model)
        total_valid_loss.append(valid_loss)
        valid_acc = evaluate_ce(valid_loader, model)
        if valid_acc > max_valid_acc:
            torch.save(model.state_dict(), 'model_era.pt')
            max_valid_acc = valid_acc
        total_valid_acc.append(valid_acc)
        total_loss.append(loss)
        train_acc = evaluate_ce(train_loader, model)
        total_train_acc.append(train_acc)
        print('Epoch {} | Training Loss: {:.5f} | Training Acc: {:.1f}% |'.format(
        e + 1, accum_loss / (i + 1), train_acc * 100, valid_loss, valid_acc * 100))
    make_figs(epochs, model, total_loss, total_valid_loss, total_train_acc, total_valid_acc, True, "transfer_era")
    return model


########################################################################################################################
# This section below loads the different data into the program
train_root = '/Users/stephenbrade/ECE324/BroadBrush/era_dataset_train'
valid_root = '/Users/stephenbrade/ECE324/BroadBrush/era_dataset_valid'
# test_root = '/Users/stephenbrade/ECE324/BroadBrush/era_dataset_test'
bs = 32
epochs = 30
seed = 15
lr = 0.01

torch.manual_seed(seed)
train = datasets.ImageFolder(train_root, transform=transforms.ToTensor())
validate = datasets.ImageFolder(train_root, transform=transforms.ToTensor())
# test = datasets.ImageFolder(train_root, transform=transforms.ToTensor())
# train, overfit = train_test_split(train, test_size=0.001, random_state=seed)


train_loader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True)
valid_loader = torch.utils.data.DataLoader(validate, batch_size=bs, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test, batch_size=bs, shuffle=True)
# overfit_loader = torch.utils.data.DataLoader(overfit, batch_size=bs, shuffle=True)

# INnstantiates GPU device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Loads pretrained model
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 6)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(), lr=lr)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
# Trains model and calls all above functions
trained_model = train_model(model_ft, train_loader, valid_loader, criterion, optimizer_ft, exp_lr_scheduler, epochs, device)

