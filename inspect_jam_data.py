import pickle
import json
import numpy as np

INPUT_PKL = "jam_all_episodes_with_goal_train_merged.pkl"
OUTPUT_JSON = "jam_data_full.json"

def episode_to_dict(ep, idx):
    ep_dict = {
        "episode_id": getattr(ep, "episode_id", idx),
        "num_steps": len(ep.steps),
        "steps": []
    }
    for step_i, step in enumerate(ep.steps):
        state = np.round(np.array(step.state_v, dtype=float), 4).tolist()
        action = np.round(np.array(step.action, dtype=float), 4).tolist()
        ep_dict["steps"].append({
            "step_id": step_i,
            "state_dim": len(state),
            "action_dim": len(action),
            "state": state,
            "action": action
        })
    return ep_dict

if __name__ == "__main__":
    with open(INPUT_PKL, "rb") as f:
        episodes = pickle.load(f)

    print(f"Loaded {len(episodes)} episodes")

    all_data = {
        "meta": {
            "total_episodes": len(episodes),
            "state_dim": len(episodes[0].steps[0].state_v),
            "action_dim": len(episodes[0].steps[0].action)
        },
        "episodes": []
    }

    for i, ep in enumerate(episodes):
        all_data["episodes"].append(episode_to_dict(ep, i))

    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"Saved full dataset to {OUTPUT_JSON}")
