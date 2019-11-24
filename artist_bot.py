import torch

import torchtext
import torch.nn as nn

import torchvision.transforms as transforms
from torchtext import data
from art_model import ArtNet
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import spacy
from torchvision import datasets, models, transforms

import os
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def get_model(filepath, art=True):
    # model = torch.load(filepath, map_location=torch.device('cpu'))
    # model = load_state_dict(torch.load(filepath))
    # model = model.load_state_dict(model)

    model_ft = models.resnet101(pretrained='imagenet')
    if art:
        model_ft.fc = nn.Linear(model_ft.fc.in_features, 24)
    else:
        model_ft.fc = nn.Linear(model_ft.fc.in_features, 6)

    # model_ft.cuda()
    model_ft = torch.nn.DataParallel(model_ft)

    model_ft.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
    # model_ft.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
    model_ft.eval()
    return model_ft

# create composite transformation for all photos
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# ArtNet = get_model("artist_model.pt")
ArtNet = get_model("artist_resnet.pt")
EraNet = get_model("era_resnet.pt", art=False)
models = [ArtNet]

artist_testset = torchvision.datasets.ImageFolder(root='artist_dataset_test', transform=transform)
print(artist_testset.classes)
era_testset = torchvision.datasets.ImageFolder(root='era_dataset_test', transform=transform)
print(era_testset.classes)


corr = 0
tot = 0
for obj in era_testset:
    img, label = obj
    
    #resize img for input into model
    img = img.view(1,3,300,300)
    # art_preds = ArtNet(img)
    art_preds = EraNet(img)
    art_preds = art_preds.tolist()[0]
    art_preds = [sigmoid(pred) for pred in art_preds]
    art_pred = art_preds.index(max(art_preds))
    # print("LABEL {} vs PRED {}".format(label, art_pred))
    tot += 1
    if int(label) == int(art_pred):
        corr += 1
    print("Runninc acc: {}".format(corr/tot))

print("Got {} out of {} correct".format(corr, tot))