import pickle
import glob

# find all matching pkl files
pkl_files = sorted(glob.glob("jam_all_episodes_with_goal_train_*.pkl"))  
# or glob.glob("sessions/*.pkl") if they’re in a folder

all_episodes = []

for fname in pkl_files:
    with open(fname, "rb") as f:
        episodes = pickle.load(f)
    print(f"Loaded {len(episodes)} episodes from {fname}")
    all_episodes.extend(episodes)

print(f"Total combined episodes: {len(all_episodes)}")

# save merged file
with open("jam_all_episodes_with_goal_train_merged.pkl", "wb") as f:
    pickle.dump(all_episodes, f)

print("Merged -> jam_all_episodes_with_goal_train_merged.pkl")
