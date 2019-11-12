# The best possible models are commented out in the this file. Before running a model uncomment it and comment all other
# models. Above them, find the command line argument used to run them them in conjunction with main.py. The modular code
# from 4.3 can also be found here. Use the self explanatory command line arguments to change the model that you desire
# from the command line. Note: the best model will be left uncommented by default however, if you wish to make this code
# modular, uncomment the code labelled modular.
#
import torch.nn as nn
import torch.nn.functional as F
import torch


import torch.nn as nn
import torch.nn.functional as F
import math

class Net(nn.Module):   #Best Model
    #python3 main.py --epochs 100  --lr 0.1 --batch_size 32 --layers 2 --hidden_size 64 --batch_norm False --num_kernel 7 --loss_func ce
    # def __init__(self, layers, hidden_size, num_kernel, batch_norm):
    #     super(Net, self).__init__()
    #     self.layers = layers
    #     self.hidden_size = hidden_size
    #     self.num_kernel = num_kernel
    #     self.batch_norm = batch_norm
    #     self.pool = nn.MaxPool2d(2, 2)
    #
    #     self.conv1 = nn.Sequential(nn.Conv2d(3, num_kernel, 7), nn.BatchNorm2d(num_kernel))
    #     self.conv2 = nn.Sequential(nn.Conv2d(num_kernel, num_kernel, 6), nn.ReLU(), nn.BatchNorm2d(num_kernel))
    #     self.conv3 = nn.Sequential(nn.Conv2d(num_kernel, num_kernel, 6), nn.ReLU(), nn.BatchNorm2d(num_kernel))
    #     self.conv4 = nn.Sequential(nn.Conv2d(num_kernel, num_kernel, 4), nn.ReLU(), nn.BatchNorm2d(num_kernel))
    #     self.conv5 = nn.Sequential(nn.Conv2d(num_kernel, num_kernel, 3), nn.ReLU(), nn.BatchNorm2d(num_kernel))
    #
    #     self.dropout = nn.Dropout(0.2)
    #
    #     self.flat_size = num_kernel * 7 * 7
    #     self.fc1 = nn.Sequential(nn.Linear(self.flat_size, hidden_size), nn.ReLU(), nn.BatchNorm1d(hidden_size))
    #     # self.fc2 = nn.Linear(100, hidden_size)
    #     self.fc3 = nn.Linear(hidden_size, 10)
    #
    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.dropout(x)
    #     x = self.conv2(x)
    #     x = self.dropout(x)
    #     x = self.pool(self.conv3(x))
    #     x = self.dropout(x)
    #     x = self.conv4(x)
    #     x = self.dropout(x)
    #     x = self.pool(self.conv5(x))
    #     x = self.dropout(x)
    #     x = x.view(-1, self.flat_size)
    #     x = self.fc1(x)
    #     x = self.dropout(x)
    #     # x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x

    def __init__(self, layers, hidden_size, num_kernel, batch_norm):
        super(Net, self).__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.num_kernel = num_kernel
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(3, num_kernel, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(num_kernel)
        self.bn2 = nn.BatchNorm2d(num_kernel)
        self.conv2 = nn.Conv2d(num_kernel, num_kernel, 3)
        self.fc1 = nn.Linear(1693440, 100) # num_kernel * 45 * 45
        self.fc2 = nn.Linear(100, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)
        self.fc1_bn = nn.BatchNorm1d(100)
        self.fc2_bn = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 1693440) # 45 * 45 * self.num_kernel
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 1693440)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

    # # modular code from 4.3
    # # main.py --epochs 100  --lr 0.1 --batch_size 32 --layers 1 --hidden_size 64 --batch_norm False --num_kernel 7 --loss_func ce
    # # change things accordingly to test
    # def __init__(self, layers, hidden_size, num_kernel, batch_norm):
    #     super(Net, self).__init__()
    #     self.layers = layers
    #     self.hidden_size = hidden_size
    #     self.num_kernel = num_kernel
    #     self.batch_norm = batch_norm
    #     if layers == 4 and batch_norm == False:
    #         self.conv1 = nn.Conv2d(3, num_kernel, 3)
    #         self.pool = nn.MaxPool2d(2, 2)
    #         self.conv2 = nn.Conv2d(num_kernel, num_kernel, 3)
    #         self.conv3 = nn.Conv2d(num_kernel, num_kernel, 3)
    #         self.conv4 = nn.Conv2d(num_kernel, num_kernel, 3)
    #         self.fc1 = nn.Linear(num_kernel*1, 100)
    #         self.fc2 = nn.Linear(100, hidden_size)
    #         self.fc3 = nn.Linear(hidden_size, 10)
    #     elif layers == 2 and batch_norm == False:
    #         self.conv1 = nn.Conv2d(3, num_kernel, 3)
    #         self.pool = nn.MaxPool2d(2, 2)
    #         self.conv2 = nn.Conv2d(num_kernel, num_kernel, 3)
    #         self.fc1 = nn.Linear(num_kernel * 12 * 12, 100)
    #         self.fc2 = nn.Linear(100, hidden_size)
    #         self.fc3 = nn.Linear(hidden_size, 10)
    #     elif layers == 1 and batch_norm == False:
    #         self.conv1 = nn.Conv2d(3, num_kernel, 3)
    #         self.pool = nn.MaxPool2d(2, 2)
    #         self.fc1 = nn.Linear(27 * 27 * num_kernel, 100)
    #         self.fc2 = nn.Linear(100, hidden_size)
    #         self.fc3 = nn.Linear(hidden_size, 10)
    #         self.fc1_bn = nn.BatchNorm1d(100)
    #         self.fc2_bn = nn.BatchNorm1d(hidden_size)
    #     elif layers == 4 and batch_norm:
    #         self.conv1 = nn.Conv2d(3, num_kernel, 3)
    #         self.pool = nn.MaxPool2d(2, 2)
    #         self.bn1 = nn.BatchNorm2d(num_kernel)
    #         self.bn2 = nn.BatchNorm2d(num_kernel)
    #         self.conv2 = nn.Conv2d(num_kernel, num_kernel, 3)
    #         self.conv3 = nn.Conv2d(num_kernel, num_kernel, 3)
    #         self.conv4 = nn.Conv2d(num_kernel, num_kernel, 3)
    #         self.fc1 = nn.Linear(num_kernel * 1, 100)
    #         self.fc2 = nn.Linear(100, hidden_size)
    #         self.fc3 = nn.Linear(hidden_size, 10)
    #         self.fc1_bn = nn.BatchNorm1d(100)
    #         self.fc2_bn = nn.BatchNorm1d(hidden_size)
    #     elif layers == 2 and batch_norm:
    #         self.conv1 = nn.Conv2d(3, num_kernel, 3)
    #         self.pool = nn.MaxPool2d(2, 2)
    #         self.bn1 = nn.BatchNorm2d(num_kernel)
    #         self.bn2 = nn.BatchNorm2d(num_kernel)
    #         self.conv2 = nn.Conv2d(num_kernel, num_kernel, 3)
    #         self.fc1 = nn.Linear(num_kernel * 12 * 12, 100)
    #         self.fc2 = nn.Linear(100, hidden_size)
    #         self.fc3 = nn.Linear(hidden_size, 10)
    #         self.fc1_bn = nn.BatchNorm1d(100)
    #         self.fc2_bn = nn.BatchNorm1d(hidden_size)
    #     elif layers == 1 and batch_norm:
    #         self.conv1 = nn.Conv2d(3, num_kernel, 3)
    #         self.bn1 = nn.BatchNorm2d(num_kernel)
    #         self.bn2 = nn.BatchNorm2d(num_kernel)
    #         self.pool = nn.MaxPool2d(2, 2)
    #         self.fc1 = nn.Linear(27 * 27 * num_kernel, 100)
    #         self.fc2 = nn.Linear(100, hidden_size)
    #         self.fc3 = nn.Linear(hidden_size, 10)
    #         self.fc1_bn = nn.BatchNorm1d(100)
    #         self.fc2_bn = nn.BatchNorm1d(hidden_size)
    #
    # def forward(self, x):
    #     if self.layers == 4 and self.batch_norm == False:
    #         x = self.pool(F.relu(self.conv1(x)))
    #         x = self.pool(F.relu(self.conv2(x)))
    #         x = self.pool(F.relu(self.conv3(x)))
    #         x = self.pool(F.relu(self.conv4(x)))
    #         x = x.view(-1, 1*self.num_kernel)
    #         x = F.relu(self.fc1(x))
    #         x = F.relu(self.fc2(x))
    #         x = self.fc3(x)
    #         return x
    #     elif self.layers == 2 and self.batch_norm == False:
    #         x = self.pool(F.relu(self.conv1(x)))
    #         x = self.pool(F.relu(self.conv2(x)))
    #         x = x.view(-1, 12 * 12 * self.num_kernel)
    #         x = F.relu(self.fc1(x))
    #         x = F.relu(self.fc2(x))
    #         x = self.fc3(x)
    #         return x
    #     elif self.layers == 1 and self.batch_norm == False:
    #         x = self.pool(F.relu(self.conv1(x)))
    #         x = x.view(-1, 27 * 27 * self.num_kernel)
    #         x = F.relu(self.fc1(x))
    #         x = F.relu(self.fc2(x))
    #         x = self.fc3(x)
    #         return x
    #     elif self.layers == 4 and self.batch_norm:
    #         x = self.pool(F.relu(self.bn1(self.conv1(x))))
    #         x = self.pool(F.relu(self.bn2(self.conv2(x))))
    #         x = self.pool(F.relu(self.bn2(self.conv3(x))))
    #         x = self.pool(F.relu(self.bn2(self.conv4(x))))
    #         x = x.view(-1, 1 * self.num_kernel)
    #         x = F.relu(self.fc1_bn(self.fc1(x)))
    #         x = F.relu(self.fc2_bn(self.fc2(x)))
    #         x = self.fc3(x)
    #         return x
    #     elif self.layers == 2 and self.batch_norm:
    #         x = self.pool(F.relu(self.bn1(self.conv1(x))))
    #         x = self.pool(F.relu(self.bn2(self.conv2(x))))
    #         x = x.view(-1, 12*12*self.num_kernel)
    #         x = F.relu(self.fc1_bn(self.fc1(x)))
    #         x = F.relu(self.fc2_bn(self.fc2(x)))
    #         x = self.fc3(x)
    #         return x
    #     elif self.layers == 1 and self.batch_norm:
    #         x = self.pool(F.relu(self.bn1(self.conv1(x))))
    #         x = x.view(-1, 27*27*self.num_kernel)
    #         x = F.relu(self.fc1_bn(self.fc1(x)))
    #         x = F.relu(self.fc2_bn(self.fc2(x)))
    #         x = self.fc3(x)
    #         return x



