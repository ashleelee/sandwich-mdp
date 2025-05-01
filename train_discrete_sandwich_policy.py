import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
import pickle
import pdb
from data_classes import Step, Episode
from policies.conformal.mlp import Discrete_Policy
import matplotlib.pyplot as plt

with open('all_episodes_gen.pkl', 'rb') as f:
    all_episodes = pickle.load(f)

# construct dataset
all_X = []
all_Y = []
for episode_idx in range(len(all_episodes)):
    steps_data = all_episodes[episode_idx].steps
    for timestep in range(len(steps_data)):
        state = np.array(steps_data[timestep].state_before)
        action = steps_data[timestep].action
        all_X.append(state)
        all_Y.append(action)

all_X = np.array(all_X)
all_Y = np.array(all_Y)




# split data into train, test, val
indices = np.arange(len(all_X))
np.random.shuffle(indices)
indices = list(indices)
x_data = all_X[indices]
y_data = all_Y[indices]
x_train, y_train, x_test, y_test, x_val, y_val = (x_data[:int(0.8 * len(x_data))], y_data[:int(0.8 * len(y_data))],
                                                  x_data[int(0.8 * len(x_data)):int(0.9 * len(x_data))],
                                                  y_data[int(0.8 * len(y_data)):int(0.9 * len(y_data))],
                                                  x_data[int(0.9 * len(x_data)):], y_data[int(0.9 * len(y_data)):])

train_loader = DataLoader(TensorDataset(torch.tensor(x_data), torch.tensor(y_data)),
                              batch_size=32,
                              shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(x_val), torch.tensor(y_val)),
                        batch_size=32,
                        shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(x_test), torch.tensor(y_test)),
                        batch_size=32,
                        shuffle=True)

N_action_classes = 38
discrete_policy = Discrete_Policy(state_dim=x_train.shape[1], output_dim=N_action_classes)


# define the optimizer
optimizer = optim.Adam(discrete_policy.parameters(), lr=0.0005)
# loss_fn
loss_fn = torch.nn.CrossEntropyLoss()

# training loop
N_eps = 150
train_losses = []
for epoch in range(N_eps):
    total_loss = 0
    for x, y in train_loader:
        x = x.float()
        y = y.long()
        optimizer.zero_grad()
        output = discrete_policy(x)

        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_losses.append(total_loss)
    if epoch % 50 == 0:
        print(f'epoch {epoch}, loss {total_loss}')


# eval on train
for x, y in train_loader:
    x = x.float()
    y = y.long()
    optimizer.zero_grad()
    output = discrete_policy(x)
    print("predicted", torch.argmax(output, dim=1))
    print("target", y)
    break

# torch save
torch.save(discrete_policy.state_dict(), 'discrete_policy.pth')

plt.plot(range(len(train_losses)), train_losses)
plt.show()








