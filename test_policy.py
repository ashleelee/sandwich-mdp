import numpy as np
import torch
from policies.conformal.mlp import Continuous_Policy
import pickle

# ---------------------------------------------------------
# 1. Load the same episodes used during training
# ---------------------------------------------------------
with open("data/jam_train_data_all.pkl", "rb") as f:
    all_episodes = pickle.load(f)

# ---------------------------------------------------------
# 2. Load normalization stats saved during training
# ---------------------------------------------------------
norm_stats = np.load("trained_policy/norm_stats.npz")

min_X = norm_stats["min_X"]
max_X = norm_stats["max_X"]
min_Y = norm_stats["min_Y"]
max_Y = norm_stats["max_Y"]

range_X = max_X - min_X
range_Y = max_Y - min_Y
range_X[range_X == 0] = 1.0
range_Y[range_Y == 0] = 1.0

print("Loaded normalization stats:")
print("min_X:", min_X.shape)
print("max_X:", max_X.shape)
print("min_Y:", min_Y.shape)
print("max_Y:", max_Y.shape)
print()

# ---------------------------------------------------------
# 3. Build a few input samples exactly like training
# ---------------------------------------------------------
X_test = []
Y_test = []

for ep in all_episodes:
    steps = ep["steps"] if isinstance(ep, dict) else ep.steps
    if len(steps) < 4:
        continue

    for t in range(len(steps) - 3):

        def get_state(s):
            if hasattr(s, "state"):
                return np.array(s.state, dtype=np.float32)
            if "state" in s:
                return np.array(s["state"], dtype=np.float32)
            if "state_v" in s:
                return np.array(s["state_v"], dtype=np.float32)
            raise ValueError("No state field found")

        s0 = get_state(steps[t])
        s1 = get_state(steps[t + 1])
        s2 = get_state(steps[t + 2])

        a = steps[t + 3]["action"] if isinstance(steps[t + 3], dict) else steps[t + 3].action

        x = np.concatenate([s0, s1, s2], axis=0)
        X_test.append(x)
        Y_test.append(a)

X_test = np.array(X_test)
Y_test = np.array(Y_test)

print("Dataset shapes:", X_test.shape, Y_test.shape)
print()

# ---------------------------------------------------------
# 4. Load trained policy
# ---------------------------------------------------------
state_dim = 36
action_dim = 3

policy = Continuous_Policy(state_dim=state_dim, output_dim=action_dim)
policy.load_state_dict(torch.load("trained_policy/cont_policy.pth"))
policy.eval()

print("Model loaded.")
print()

# ---------------------------------------------------------
# 5. Pick random samples and test the policy
# ---------------------------------------------------------
for i in range(5):
    idx = np.random.randint(0, len(X_test))

    x_raw = X_test[idx]
    y_true = Y_test[idx]

    # normalize same as during training
    x_norm = (x_raw - min_X) / range_X
    x_tensor = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        y_norm_pred = policy(x_tensor).numpy()[0]

    # unnormalize back
    y_pred = y_norm_pred * range_Y + min_Y

    print(f"Sample {i}")
    print("   raw state min/max:", x_raw.min(), x_raw.max())
    print("   normalize min/max:", x_norm.min(), x_norm.max())
    print("   predicted action :", y_pred)
    print("   true action      :", y_true)
    print()
