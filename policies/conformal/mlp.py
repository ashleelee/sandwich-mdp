import pickle

import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
import pdb
import matplotlib.pyplot as plt

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
        # x = torch.sigmoid(self.fc4(x))
        return x
