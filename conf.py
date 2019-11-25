import csv

import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def open_file(filename):
    x = []
    with open(filename) as file:
        csv_reader = csv.reader(file, delimiter="\n")
        for row in csv_reader:
            entry = row[0]
            x.append(int(float(entry)))
    return np.array(x)

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
    ax.set_ylim(cm.shape[0]-0.5, -0.5)

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

# create composite transformation for all photos
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

artist_testset = torchvision.datasets.ImageFolder(root='artist_dataset_test', transform=transform)

# open all files into arrays
true_era = open_file("true_era.txt")
true_artist = open_file("true_artist.txt")

pred_era = open_file("pred_era.txt")
pred_artist = open_file("pred_artist.txt")
pred_artist_from_era = open_file("pred_artist_from_era.txt")

print(pred_artist_from_era)

# create confusion matrix
plot_confusion_matrix(true_era, pred_era, classes=np.array(["Baroque", "Cubism", "Impressionism", "Pop Art", "Realism", "Renaissance"]), title="Era Confusion Matrix")
plot_confusion_matrix(true_artist, pred_artist, classes=np.array(artist_testset.classes), title="Artist Confusion Matrix")
plot_confusion_matrix(true_artist, pred_artist_from_era, classes=np.array(artist_testset.classes), title="Artist from Era Confusion Matrix")
plt.show()
