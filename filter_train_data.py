import pickle
import math
import json

in_path = "data/jam_train_data_all_triangle.pkl"
out_pkl = "data/jam_train_data_all_triangle_filtered.pkl"
out_json = "data/jam_train_data_all_triangle_filtered.json"

THRESH = 3

def should_drop(prev_action, curr_action, thresh=THRESH):
    """
    Return True if we want to drop curr_action.
    Condition:
      positions are close AND gripper value is the same
    """
    # If gripper changes, never drop
    if prev_action[2] != curr_action[2]:
        return False

    # Otherwise check distance in (x, y)
    dx = curr_action[0] - prev_action[0]
    dy = curr_action[1] - prev_action[1]
    dist = math.hypot(dx, dy)
    return dist <= thresh

with open(in_path, "rb") as f:
    data = pickle.load(f)

for ep in data:
    steps = ep.get("steps", [])
    if not steps:
        continue

    new_steps = [steps[0]]  # always keep first step

    for i in range(1, len(steps)):
        prev = new_steps[-1]    # last kept step
        curr = steps[i]

        if not should_drop(prev["action"], curr["action"]):
            new_steps.append(curr)
        # else: drop curr

    # reindex step_id
    for new_id, step in enumerate(new_steps):
        step["step_id"] = new_id

    ep["steps"] = new_steps

# save filtered pickle
with open(out_pkl, "wb") as f:
    pickle.dump(data, f)

# save filtered JSON
with open(out_json, "w") as f:
    json.dump(data, f, indent=2)

print("Saved filtered pickle:", out_pkl)
print("Saved filtered JSON:", out_json)
