import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
import pickle
import pdb
from jam_data_classes import TimeStep, Episode
from policies.conformal.mlp import Continuous_Policy
import matplotlib.pyplot as plt

with open('jam_all_episodes_gen.pkl', 'rb') as f:
    all_episodes = pickle.load(f)

# construct dataset
all_X = []
all_Y = []
for episode_idx in range(len(all_episodes)):
    steps_data = all_episodes[episode_idx].steps
    for timestep in range(len(steps_data)-4):
        state = np.array(steps_data[timestep].state_v[6:])
        state_t1 = np.array(steps_data[timestep+1].state_v[6:])
        state_t2 = np.array(steps_data[timestep+2].state_v[6:])
        action = steps_data[timestep+3].action
        state = np.concatenate((state, state_t1, state_t2))
        all_X.append(state)
        all_Y.append(action)
        # pdb.set_trace()

all_X = np.array(all_X)
all_Y = np.array(all_Y)

# normalize by min max of X and Y
min_X = np.min(all_X, axis=0)
max_X = np.max(all_X, axis=0)
all_X = (all_X - min_X) / (max_X - min_X)
min_Y = np.min(all_Y, axis=0)
max_Y = np.max(all_Y, axis=0)
all_Y = (all_Y - min_Y) / (max_Y - min_Y)

print("nans in all_X", np.isnan(all_X).sum())
print("nans in all_Y", np.isnan(all_Y).sum())

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
                              batch_size=256,
                              shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(x_val), torch.tensor(y_val)),
                        batch_size=32,
                        shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(x_test), torch.tensor(y_test)),
                        batch_size=32,
                        shuffle=True)

action_dim = 3
cont_policy = Continuous_Policy(state_dim=x_train.shape[1], output_dim=action_dim)
print(x_train.shape[1])



# define the optimizer
optimizer = optim.Adam(cont_policy.parameters(), lr=0.0005)
# loss_fn
loss_fn = torch.nn.MSELoss()

# training loop
N_eps = 1500
train_losses = []
for epoch in range(N_eps):
    total_loss = 0
    for x, y in train_loader:
        x = x.float()
        y = y.float()
        optimizer.zero_grad()
        output = cont_policy(x)

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
    y = y.float()
    optimizer.zero_grad()
    output = cont_policy(x)
    print("predicted", output)
    print("target", y)
    break

# torch save
torch.save(cont_policy.state_dict(), 'cont_policy.pth')

plt.plot(range(len(train_losses)), train_losses)
plt.show()








