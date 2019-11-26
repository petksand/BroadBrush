import argparse
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
import torch
from torchvision import datasets
from model import Net
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision
from scipy.signal import savgol_filter
from torchsummary import summary
from sklearn.metrics import confusion_matrix



def one_hot(x):
    encoded = torch.zeros(10)
    encoded[x] = 1
    return encoded

def decode(one_hot, size):
    print(one_hot)
    one_hot = one_hot.tolist()
    corr = []
    for i in range(size):
        corr.append(one_hot[i].index(max(one_hot[i])))
    return corr
def evaluate(loader, net):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # print(predicted.shape, labels.shape)
            correct += (predicted == labels.max(axis=1)[1]).sum().item()
    #print(correct/len(loader.dataset))
    net.train()
    return correct/len(loader.dataset)


def evaluate_ce(loader, net):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i,data in enumerate(loader,0):
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    net.train()
    return correct/total

#
def eval_loss(loader, net):
    net.eval()
    total = 0
    total_loss = 0
    for i, data in enumerate(loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # print(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        # print(len(outputs), len(labels))
        loss = criterion(input=outputs, target=labels).mean()
        total_loss += loss
        total += 1
    net.train()
    return total_loss/total

def imshow(img):    # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images





parser = argparse.ArgumentParser()
parser.add_argument('--start', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--eval_every', type=int, default=50)
parser.add_argument('--act', type=str, default='relu')
parser.add_argument('--seed', type=int, default=12)
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--hidden_size', type=int, default=32)
parser.add_argument('--num_kernel', type=int, default=30)
parser.add_argument('--batch_norm', type=bool, default=True)
parser.add_argument('--loss_func', type=str, default='mse')




args = parser.parse_args()
loss_func = args.loss_func
layers = args.layers
hidden_size = args.hidden_size
num_kernel = args.num_kernel
batch_norm = args.batch_norm
act = args.act
epochs = args.epochs
batch_size = args.batch_size
lr = args.lr
torch.manual_seed(args.seed)

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K']

root = '/Users/stephenbrade/ECE324/BroadBrush/dataset_crop_random'

torch.manual_seed(args.seed)
data = datasets.ImageFolder(root, transform=transforms.ToTensor())
data_loader = torch.utils.data.DataLoader(data, batch_size=1370, shuffle=True)
mean = 0.0
std = 0.0
max_valid_acc = 0
for files, _ in data_loader:
    samples = files.size(0)
    files = files.view(samples, files.size(1), -1)
    mean += files.mean(2).sum(0)
    std += files.std(2).sum(0)
mean /= len(data_loader.dataset)
std /= len(data_loader.dataset)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
data = datasets.ImageFolder(root, transform=transform)
train, validate = train_test_split(data, test_size=0.2, random_state=1)
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(validate, batch_size=batch_size, shuffle=True)
net = Net(layers=layers, hidden_size=hidden_size, batch_norm=batch_norm, num_kernel=num_kernel)
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
t = 0
total_train_acc = []
total_valid_acc = []
total_loss = []
total_valid_loss = []
true = []
predict = []
start = time.time()
time.clock()
for e in range(epochs):  # loop over the dataset multiple times
    accum_loss = 0.0
    tot_corr = 0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # print(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        # print(len(outputs), len(labels))
        loss = criterion(input=outputs, target=labels).mean()
        loss.backward()
        optimizer.step()
        # print statistics
        accum_loss += loss

        # if i % args.eval_every == 0 or i == 0:
        #     print(" Loss: ", accum_loss/args.eval_every)
        #     train_acc = evaluate(train_loader, net)
        #     valid_acc = evaluate(valid_loader, net)
        #     print("Train Acc:", train_acc, "Valid Acc:", valid_acc)
        #     total_train_acc.append(train_acc)
        #     # total_valid_acc.append(valid_acc)
        #     accum_loss = 0
        # t += 1
    # labels_decoded = decode(labels, labels.shape[0])
    labels = torch.FloatTensor(labels.float())
    true.extend((labels.numpy()))

    _, predicted = torch.max(outputs.data, 1)
    predict.extend(predicted.numpy())

    valid_loss = eval_loss(valid_loader, net)
    total_valid_loss.append(valid_loss)
    valid_acc = evaluate_ce(valid_loader, net)
    if valid_acc > max_valid_acc:
        torch.save(net.state_dict(), 'MyBestSmall.pt')
        max_valid_acc = valid_acc
    total_valid_acc.append(valid_acc)
    total_loss.append(loss)
    train_acc = evaluate_ce(train_loader, net)
    total_train_acc.append(train_acc)
    print('Epoch {} | Training Loss: {:.5f} | Training Acc: {:.1f}% | Valid Loss: {:.5f} | Valid Acc: {:.1f}% |'.format(
        e + 1, accum_loss / (i + 1), train_acc * 100, valid_loss, valid_acc * 100))

print('Finished Training')
summary(net, input_size=(3, 178, 178))
print(time.time() - start)
print(max(total_valid_acc))
total_loss = savgol_filter(total_loss, 5, 3)
total_valid_loss = savgol_filter(total_valid_loss, 5, 3)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(range(0, epochs), total_loss, label='Training Data')
ax.plot(range(0, epochs), total_valid_loss, label='Validation Data')
plt.title('Best Result with Cross Entropy: Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
ax.legend()
plt.savefig("loss_my_data1.png")
plt.show()

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(range(0, epochs), total_train_acc,
        label='Training Data')
ax.plot(range(0, epochs), total_valid_acc,
        label='Validation Data')
ax.plot(range(0, len(total_train_acc) * args.eval_every, args.eval_every), total_train_acc,
        label='Training Data')
ax.plot(range(0, len(total_valid_acc) * args.eval_every, args.eval_every), total_valid_acc,
        label='Validation Data ')
plt.ylim(0, 1)
plt.title('Best Result with Cross Entropy: Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
ax.legend()
plt.savefig("acc_my_data1.png")
print(confusion_matrix(true, predict))
plt.show()