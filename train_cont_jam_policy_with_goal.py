import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import matplotlib.pyplot as plt
from jam_data_classes import TimeStep, Episode
from policies.conformal.mlp import Continuous_Policy

# load data
with open('jam_all_episodes_with_goal_train_merged.pkl', 'rb') as f:
    all_episodes = pickle.load(f)

print(f"Loaded {len(all_episodes)} episodes")

# construct dataset
all_X = []
all_Y = []
for ep_idx, ep in enumerate(all_episodes):
    steps = ep.steps
    for t in range(len(steps) - 4):
        # Use full state vector (no slicing like [6:])
        s0 = np.array(steps[t].state_v, dtype=np.float32)
        s1 = np.array(steps[t+1].state_v, dtype=np.float32)
        s2 = np.array(steps[t+2].state_v, dtype=np.float32)
        action = np.array(steps[t+3].action, dtype=np.float32)

        # concatenate states of 3 timestep
        state = np.concatenate((s0, s1, s2))
        all_X.append(state)
        all_Y.append(action)

all_X = np.array(all_X, dtype=np.float32)
all_Y = np.array(all_Y, dtype=np.float32)
print(f"Dataset shapes: X={all_X.shape}, Y={all_Y.shape}")

# normalize by min max of X and Y, + 1e-8 avoid division by zero
min_X = np.min(all_X, axis=0)
max_X = np.max(all_X, axis=0)
all_X = (all_X - min_X) / (max_X - min_X + 1e-8)
min_Y = np.min(all_Y, axis=0)
max_Y = np.max(all_Y, axis=0)
all_Y = (all_Y - min_Y) / (max_Y - min_Y + 1e-8)

# save the min max for normalize and denormalize later
np.savez("minmax_stats_with_goal.npz",
         x_min=min_X, x_max=max_X,
         y_min=min_Y, y_max=max_Y)

# sanity check on the normalized data
print("NaNs in X:", np.isnan(all_X).sum(), "NaNs in Y:", np.isnan(all_Y).sum())

# split into train / val / test
indices = np.arange(len(all_X))
np.random.shuffle(indices)
indices = list(indices)
split1 = int(0.8 * len(indices))
split2 = int(0.9 * len(indices))

x_train = all_X[indices[:split1]]
y_train = all_Y[indices[:split1]]
x_val = all_X[indices[split1:split2]]
y_val = all_Y[indices[split1:split2]]
x_test = all_X[indices[split2:]]
y_test = all_Y[indices[split2:]]

print(f"Split: train={len(x_train)} val={len(x_val)} test={len(x_test)}")

# Dataloaders
train_loader = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train)), batch_size=256, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(x_val), torch.tensor(y_val)), batch_size=64, shuffle=False)
test_loader = DataLoader(TensorDataset(torch.tensor(x_test), torch.tensor(y_test)), batch_size=64, shuffle=False)


# model set up
state_dim = all_X.shape[1] # 51
action_dim = all_Y.shape[1] # 3
cont_policy = Continuous_Policy(state_dim=state_dim, output_dim=action_dim)
print(f"State dim={state_dim}, Action dim={action_dim}")

# define the optimizer
optimizer = optim.Adam(cont_policy.parameters(), lr=5e-6)

# define the loss function
loss_fn = torch.nn.MSELoss()

# training loop
N_eps = 1000
train_losses = []

for epoch in range(N_eps):
    total_loss = 0.0
    for x, y in train_loader:
        x = x.float()
        y = y.float()
        optimizer.zero_grad()
        output = cont_policy(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_losses.append(total_loss / len(train_loader))

    if epoch % 50 == 0:
        print(f"Epoch {epoch:4d}, Loss: {total_loss / len(train_loader):.4f}")

# quick eval on a batch of train
x, y = next(iter(train_loader))
x = x.float()
y = y.float()
output = cont_policy(x)
print("\nSample prediction:")
print("Predicted:", output[0].detach().numpy())
print("Target:", y[0].detach().numpy())

cont_policy.eval()
with torch.no_grad():
    xb, yb = next(iter(train_loader))
    pred = cont_policy(xb.float())
    print("pred min/max:", pred.min().item(), pred.max().item())

import torch.nn.functional as F
y_mean = torch.tensor(y_train).float().mean(dim=0, keepdim=True)
with torch.no_grad():
    yb = torch.tensor(y_val).float()
    mse_baseline = F.mse_loss(y_mean.expand_as(yb), yb).item()
print("Baseline (mean predictor) val MSE:", mse_baseline)


# save model
torch.save(cont_policy.state_dict(), 'cont_policy_with_goal_3.pth')

# plot training loss
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.show()
