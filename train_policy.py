import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
import pickle
import pdb
import os
import matplotlib.pyplot as plt

from jam_data_classes import TimeStep, Episode
from policies.conformal.mlp import Continuous_Policy
import matplotlib.pyplot as plt

shape = "zigzag_70"

# load data
with open(f'data/jam_train_data_all_{shape}.pkl', 'rb') as f:
    all_episodes = pickle.load(f)   # list[Episode] or list[dict]


# construct dataset: X = concat of 3 states, Y = future action
all_X = []
all_Y = []

for ep in all_episodes:
    # episode can be an Episode object or a dict
    if hasattr(ep, "steps"):
        steps = ep.steps
    else:
        steps = ep["steps"]

    if len(steps) < 4:
        continue

    for t in range(len(steps) - 4):
        s0_raw = steps[t]
        s1_raw = steps[t + 1]
        s2_raw = steps[t + 2]
        a_raw  = steps[t + 3]

        # step can be TimeStep or dict
        if hasattr(s0_raw, "state"):
            # new version: TimeStep.state
            s0 = np.array(s0_raw.state, dtype=np.float32)
            s1 = np.array(s1_raw.state, dtype=np.float32)
            s2 = np.array(s2_raw.state, dtype=np.float32)
            action  = np.array(a_raw.action, dtype=np.float32)
        else:
            # dict version: support both "state" and "state_v"
            def extract_state(d):
                if "state" in d:
                    return np.array(d["state"], dtype=np.float32)
                else:
                    raise KeyError("No 'state' or 'state_v' key in timestep dict")

            s0 = extract_state(s0_raw)
            s1 = extract_state(s1_raw)
            s2 = extract_state(s2_raw)
            action  = np.array(a_raw["action"], dtype=np.float32)

        state = np.concatenate([s0, s1, s2], axis=0)
        all_X.append(state)
        all_Y.append(action)

all_X = np.array(all_X, dtype=np.float32)
all_Y = np.array(all_Y, dtype=np.float32)

print("all_X.shape:", all_X.shape)
print("all_Y.shape:", all_Y.shape)

# min max normalization and save stats
min_X = all_X.min(axis=0)
max_X = all_X.max(axis=0)
range_X = max_X - min_X
range_X[range_X == 0] = 1.0   # avoid division by zero

X_norm = (all_X - min_X) / range_X

min_Y = all_Y.min(axis=0)
max_Y = all_Y.max(axis=0)
range_Y = max_Y - min_Y
range_Y[range_Y == 0] = 1.0

Y_norm = (all_Y - min_Y) / range_Y

print("nans in X_norm:", np.isnan(X_norm).sum())
print("nans in Y_norm:", np.isnan(Y_norm).sum())


# split data into train, test, val
indices = np.arange(len(all_X))
np.random.shuffle(indices)

x_data = X_norm[indices]
y_data = Y_norm[indices]

N = len(x_data)
n_train = int(0.8 * N)
n_val   = int(0.9 * N)

# split
x_train, y_train = x_data[:n_train],          y_data[:n_train]
x_val,   y_val   = x_data[n_train:n_val],     y_data[n_train:n_val]
x_test,  y_test  = x_data[n_val:],            y_data[n_val:]

# dataloaders use the splits
train_loader = DataLoader(
    TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
    batch_size=256,
    shuffle=True,
)

val_loader = DataLoader(
    TensorDataset(torch.tensor(x_val), torch.tensor(y_val)),
    batch_size=32,
    shuffle=False,
)

test_loader = DataLoader(
    TensorDataset(torch.tensor(x_test), torch.tensor(y_test)),
    batch_size=32,
    shuffle=False,
)

action_dim = 3
cont_policy = Continuous_Policy(state_dim=x_train.shape[1], output_dim=action_dim)
print("state_dim:", x_train.shape[1])

# policy and optimizer
action_dim = 3
state_dim = x_train.shape[1]   # should be 36
cont_policy = Continuous_Policy(state_dim=state_dim, output_dim=action_dim)

optimizer = optim.Adam(cont_policy.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()


# training loop
N_eps = 1000
train_losses = []
val_losses = []

for epoch in range(N_eps):
    cont_policy.train()
    total_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.float()
        y_batch = y_batch.float()

        optimizer.zero_grad()
        y_pred = cont_policy(x_batch)
        # y_pred = torch.clamp(y_pred, 0.0, 1.0) #clamp to [0, 1] when computing loss
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_losses.append(total_loss)

    # simple validation loss every epoch
    cont_policy.eval()
    total_val = 0.0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.float()
            y_batch = y_batch.float()
            y_pred = cont_policy(x_batch)
            loss = loss_fn(y_pred, y_batch)
            total_val += loss.item()
    val_losses.append(total_val)

    if epoch % 50 == 0:
        print(f"epoch {epoch}, train_loss {total_loss:.4f}, val_loss {total_val:.4f}")

# for epoch in range(N_eps):
#     cont_policy.train()
#     total_loss = 0.0

#     for x_batch, y_batch in train_loader:
#         x_batch = x_batch.float()
#         y_batch = y_batch.float()

#         optimizer.zero_grad()
#         y_pred = cont_policy(x_batch)
#         loss = loss_fn(y_pred, y_batch)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(cont_policy.parameters(), 1.0)
#         optimizer.step()

#         total_loss += loss.item()

#     mean_train = total_loss / len(train_loader)
#     train_losses.append(mean_train)

#     # validation
#     cont_policy.eval()
#     total_val = 0.0
#     with torch.no_grad():
#         for x_batch, y_batch in val_loader:
#             x_batch = x_batch.float()
#             y_batch = y_batch.float()
#             y_pred = cont_policy(x_batch)
#             y_pred = torch.clamp(y_pred, 0.0, 1.0)
#             loss = loss_fn(y_pred, y_batch)
#             total_val += loss.item()

#     mean_val = total_val / len(val_loader)
#     val_losses.append(mean_val)

#     if epoch % 50 == 0:
#         print(f"epoch {epoch}, train_loss {mean_train:.4f}, val_loss {mean_val:.4f}")



# eval on train
cont_policy.eval()
with torch.no_grad():
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.float()
        y_batch = y_batch.float()
        y_pred = cont_policy(x_batch)
        print("predicted (normalized)", y_pred[0])
        print("target    (normalized)", y_batch[0])
        break


# save policy and normalization stats
os.makedirs("trained_policy", exist_ok=True)

torch.save(cont_policy.state_dict(), f"trained_policy/cont_policy_{shape}.pth")

np.savez(
    f"trained_policy/norm_stats_{shape}.npz",
    min_X=min_X,
    max_X=max_X,
    min_Y=min_Y,
    max_Y=max_Y
)

print("Saved policy and norm_stats to trained_policy/")


# plot training curve
plt.figure()
plt.plot(range(len(train_losses)), train_losses, label="train")
plt.plot(range(len(val_losses)), val_losses, label="val")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("Training and validation loss")
plt.show()
