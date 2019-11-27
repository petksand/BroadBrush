import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchtext
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from torchtext import data
from torchvision import datasets, models

from art_model import ArtNet

_ERAS_ = {
    0: [3, 6, 7, 22],       # Baroque
    1: [10, 15, 17, 18],    # Cubism
    2: [8, 9, 14, 19],      # Impressionism
    3: [2, 13, 21, 23],     # Pop Art
    4: [1, 4, 5, 12],       # Realism
    5: [0, 11, 16, 20],     # Renaissance
}

_ARTISTS_ = [5, 4, 3, 0, 4, 4, 0, 0, 2, 2, 1, 5, 4, 3, 2, 1, 5, 1, 1, 2, 5, 3, 0, 3]

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def get_model(filepath, art=True, transfer=True):
    if not transfer:
        model = torch.load(filepath, map_location=torch.device('cpu'))
        model.eval()
        return model
    else:
        model_ft = models.resnet101(pretrained='imagenet')
        if art:
            model_ft.fc = nn.Linear(model_ft.fc.in_features, 24)
        else:
            model_ft.fc = nn.Linear(model_ft.fc.in_features, 6)
        model_ft = torch.nn.DataParallel(model_ft)
        model_ft.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
        model_ft.eval()
        return model_ft


# create composite transformation for all photos
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ArtNet = get_model("artist_model.pt", transfer=False)
ArtNet = get_model("artist_resnet.pt")
EraNet = get_model("era_resnet.pt", art=False)

artist_testset = torchvision.datasets.ImageFolder(root='artist_dataset_test', transform=transform)
# print(artist_testset.classes)
# era_testset = torchvision.datasets.ImageFolder(root='era_dataset_test', transform=transform)

corr_era = 0
corr_art = 0
corr_era_to_art = 0
tot = 0

true_era = []
true_artist = []

pred_era = []
pred_artist = []
pred_artist_from_era = []

for obj in artist_testset:
    img, artist = obj
    era = _ARTISTS_[artist]
    #resize img for input into model
    img = img.view(1,3,300,300)
    
    true_artist.append(artist)
    true_era.append(era)

    tot += 1
    # input into era
    era_preds = EraNet(img)
    era_preds = era_preds.tolist()[0]
    era_preds = [sigmoid(pred) for pred in era_preds]
    era_pred = era_preds.index(max(era_preds))
    # check era accuracy
    if int(era_pred) == int(era):
        corr_era += 1

    # input into art
    art_preds = ArtNet(img)
    art_preds = art_preds.tolist()[0]
    art_preds = [sigmoid(pred) for pred in art_preds]

    # evaluate without era
    art_pred = art_preds.index(max(art_preds))
    if float(artist) == float(art_pred):
        corr_art += 1

    # take out indexes only for appropriate era
    art_preds_cut = [art_preds[k] for k in _ERAS_[era]]
    art_pred_from_era = art_preds_cut.index(max(art_preds_cut))
    art_pred_from_era = _ERAS_[era][art_pred_from_era]
    if float(artist) == float(art_pred_from_era):
        corr_era_to_art += 1

    pred_era.append(era_pred)
    pred_artist.append(art_pred)
    pred_artist_from_era.append(art_pred_from_era)
    
    print("Runninc acc: {}".format(corr_era_to_art/tot))

print("Got {} out of {} eras correct for an accuracy of {}".format(corr_era, tot, float(corr_era)/tot))
print("Got {} out of {} artists correct for an accuracy of {} when fed from era".format(corr_era_to_art, tot, float(corr_era_to_art)/tot))
print("Got {} out of {} artists correct for an accuracy of {} when not fed from era".format(corr_art, tot, float(corr_art)/tot))

# save conf arrays
np.savetxt("true_era.txt", np.asarray(true_era))
np.savetxt("true_artist.txt", np.asarray(true_artist))

np.savetxt("pred_era.txt", np.asarray(pred_era))
np.savetxt("pred_artist.txt", np.asarray(pred_artist))
np.savetxt("pred_artist_from_era.txt", np.asarray(pred_artist_from_era))
