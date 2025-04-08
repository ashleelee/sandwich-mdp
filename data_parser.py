import pickle
import matplotlib.pyplot as plt

class Step:
    def __init__(self, step_id, state_before, action, state_after, context, robot_prediction):
        self.step_id = step_id
        self.state_before = [int(x) for x in state_before]
        self.action = int(action)
        self.state_after = [int(x) for x in state_after]
        self.context = context  # "robot_independent", "robot_asked", or "human_intervened"
        self.robot_prediction = None if context == "robot_independent" else int(robot_prediction)

    def to_dict(self):
        return self.__dict__
    
class Episode:
    def __init__(self, episode_id):
        self.episode_id = episode_id
        self.steps = []

    def add_step(self, step):
        self.steps.append(step)

    def to_dict(self):
        return {
            "episode_id": self.episode_id,
            "steps": [step.to_dict() for step in self.steps]
        }

# load the data
with open("all_episodes.pkl", "rb") as f:
    all_episodes = pickle.load(f)


# save the percentage of each type of actions
episode_stats = [] 

for episode in all_episodes:
    total_steps = len(episode.steps)
    counts = {
        "robot_independent": 0,
        "robot_asked": 0,
        "human_intervened": 0
    }

    for step in episode.steps:
        counts[step.context] += 1

    # calculate percentages for each episode
    episode_stats.append({
        "independent": counts["robot_independent"] / total_steps * 100,
        "asked": counts["robot_asked"] / total_steps * 100,
        "intervened": counts["human_intervened"] / total_steps * 100
    })



# extract episode numbers and percentages
episodes = list(range(len(episode_stats))) 
independent = [ep["independent"] for ep in episode_stats]
asked = [ep["asked"] for ep in episode_stats]
intervened = [ep["intervened"] for ep in episode_stats]

# plot
plt.figure(figsize=(10, 6))
plt.plot(episodes, independent, label="Robot Independent")
plt.plot(episodes, asked, label="Robot Asked")
plt.plot(episodes, intervened, label="Human Intervened")

plt.xlabel("Episode Number")
plt.ylabel("Percentage of Steps (%)")
plt.title("Intervention Types per Episode")
plt.xticks(episodes) 
# plt.xticks(ticks=range(0, len(episodes)+1, 5)) 
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

