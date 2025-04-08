import pickle

import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
import pdb
import matplotlib.pyplot as plt



class Discrete_Policy(nn.Module):
    def __init__(self, state_dim, output_dim):
        super(Discrete_Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = torch.softmax(x, dim=1)
        return x

