import pickle

import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
import pdb
import matplotlib.pyplot as plt



# class Discrete_Policy(nn.Module):
#     def __init__(self, state_dim, output_dim):
#         super(Discrete_Policy, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 64)
#         self.fc2 = nn.Linear(64, 128)
#         self.fc3 = nn.Linear(128, 64)
#         self.fc4 = nn.Linear(64, output_dim)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.relu(self.fc3(x))
#         x = self.fc4(x)
#         x = torch.softmax(x, dim=1)
#         return x

class Discrete_Policy(nn.Module):
    def __init__(self, state_dim, output_dim):
        super(Discrete_Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return torch.softmax(x, dim=1)


class Continuous_Policy(nn.Module):
    def __init__(self, state_dim, output_dim):
        super(Continuous_Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return x