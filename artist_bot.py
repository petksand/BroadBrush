import math
import os

import matplotlib.pyplot as plt
import numpy as np
import spacy
import torch
import torch.nn as nn
import torchtext
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from torchtext import data
from torchvision import datasets, models
from torchvision import transforms as transforms

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

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

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

# ArtNet = get_model("artist_model.pt", transfer=False)
ArtNet = get_model("artist_resnet.pt")
EraNet = get_model("era_resnet.pt", art=False)

artist_testset = torchvision.datasets.ImageFolder(root='artist_dataset_test', transform=transform)
# print(artist_testset.classes)
era_testset = torchvision.datasets.ImageFolder(root='era_dataset_test', transform=transform)

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
    update_conf(era, era_pred, conf=0)

    # input into art
    art_preds = ArtNet(img)
    art_preds = art_preds.tolist()[0]
    art_preds = [sigmoid(pred) for pred in art_preds]

    # evaluate without era
    art_pred = art_preds.index(max(art_preds))
    if int(artist) == int(art_pred):
        corr_art += 1
    update_conf(artist, art_pred, conf=1)

    # take out indexes only for appropriate era
    art_preds_cut = [art_preds[k] for k in _ERAS_[era]]
    art_pred_from_era = art_preds_cut.index(max(art_preds_cut))
    if int(artist) == int(art_pred_from_era):
        corr_era_to_art += 1
    update_conf(artist, art_pred_from_era, conf=2)

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

# create confusion matrix
# plot_confusion_matrix(true_era, pred_era, classes=["Baroque", "Cubism", "Impressionism", "Pop Art", "Realism", "Renaissance"], title="Era Confusion Matrix")
# plot_confusion_matrix(true_artist, pred_artist, classes=artist_testset.classes, title="Artist Confusion Matrix")
# plot_confusion_matrix(true_artist, pred_artist_from_era, classes=artist_testset.classes, title="Artist from Era Confusion Matrix")
# plt.show()
