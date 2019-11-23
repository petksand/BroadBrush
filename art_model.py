import torch.nn as nn
import torch
import torch.nn.functional as F


class ArtNet(nn.Module):

    def __init__(self, layers, hidden_size, num_kernel, batch_norm, output_size):
        """
        :param output_size: (int) dependent on era, this may change?

        """
        super(ArtNet, self).__init__()

        self.layers = layers
        self.hidden_size = hidden_size
        self.num_kernel = num_kernel
        self.batch_norm = batch_norm
        self.output_size = output_size

        self.conv1 = nn.Conv2d(3, num_kernel, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(num_kernel)
        self.bn2 = nn.BatchNorm2d(num_kernel)
        self.conv2 = nn.Conv2d(num_kernel, num_kernel, 10)
        self.fc1 = nn.Linear(num_kernel*68*68, 100) 
        self.fc2 = nn.Linear(100, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 24)
        self.fc1_bn = nn.BatchNorm1d(100)
        self.fc2_bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        # print(x.shape)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # print(x.shape)
        x = x.view(-1, 68*68*self.num_kernel) 
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