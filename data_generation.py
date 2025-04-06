import pickle
import json
import random
import numpy as np
from d_ingredients import dict_ingredients
from actions import dict_actions

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

    def save_to_json(self, filename):
        with open(filename, "w") as f:
            json.dump(
                self.to_dict(),
                f,
                indent=4,  # Keep overall structure readable
                separators=(",", ": "),  # Add spacing for readability
                ensure_ascii=False
            )

def update_state(state, action):
    # mapping of actions to corresponding state indices
    action_to_state_index = {
        0: 0,                               # plate
        1: 1,  2: 1,                        # bottom bread
        3: 2,  4: 2,                        # regular mayo
        5: 3,  6: 3,                        # veggie mayo
        7: 4,  8: 4,  9: 4, 10: 4, 11: 4,   # tomato
        12: 5, 13: 5, 14: 5, 15: 5, 16: 5,  # avocado
        17: 6, 18: 6,                       # lettuce
        19: 7, 20: 7, 21: 7, 22: 7, 23: 7,  # eggs
        24: 8, 25: 8,                       # ham
        26: 9, 27: 9, 28: 9,                # bacon
        29: 10, 30: 10, 31: 10,             # plant-based meat
        32: 11, 33: 11,                     # cheese
        34: 12, 35: 12,                     # pepper
        36: 13, 37: 13                      # top bread
    }

    high = np.array([
            2,  # number of plate states
            3,  # number of bottom bread states
            3,  # number of regular mayo states
            3,  # number of veggie mayo states
            6,  # number of tomato states
            6,  # number of avocado states
            3,  # number of lettuce states
            6,  # number of eggs states
            3,  # number of ham states
            4,  # number of bacon states
            4,  # number of plant-based meat states
            3, # number of cheese states
            3,  # number of pepper states
            3   # number of top bread states
        ], dtype=np.int32) - 1 # subtracting 1 = the max value for each index
    
    # check if action is mapped to a state index (safety check)
    if action in action_to_state_index:
        state_index = action_to_state_index[action]
        
        # handle special cases (multiple ways to process tomato, avocado, and eggs)
        if action == 9: state[state_index] = 3
        elif action == 10: state[state_index] = 4
        elif action == 11: state[state_index] = 5
        elif action == 14: state[state_index] = 3
        elif action == 15: state[state_index] = 4
        elif action == 16: state[state_index] = 5
        elif action == 21: state[state_index] = 3
        elif action == 22: state[state_index] = 4
        elif action == 23: state[state_index] = 5
        else:
            # check we do not exceed max allowed value for the state
            state[state_index] = min(state[state_index] + 1, high[state_index])

def sample_action(state, last_action, pref, performed_actions, step_id):

    # ensure first three actions: plate, bottom bread (take out & place)
    if last_action is None:
        return 0
    if last_action == 0:
        return 1
    if last_action == 1:
        return 2
    
    # continue previous action
    fixed_transitions = {
        8: 10,
        9: 11,
        13: 15,
        14: 16,
        20: 22,
        21: 23
    }
    random_transitions = {
        7: [8, 9],
        12: [13, 14],
        19: [20, 21]
    }
    if last_action in fixed_transitions:
        return fixed_transitions[last_action]
    if last_action in random_transitions:
        return random.choice(random_transitions[last_action])
    elif last_action in {3, 5, 17, 24, 26, 27, 29, 30, 32, 34, 36}:
        return last_action + 1
    
    # if steps go over 20, place top bread
    if step_id >= 20:
        return 36
    
    # start a new action sequence
    valid_starts = [3, 5, 7, 12, 17, 19, 24, 26, 29, 32, 34]
    choices = []
    for a in valid_starts:
        if a not in performed_actions:
            choices.append(a)
    
    # handle vegetarian preference (exclude meat/mayo options)
    if pref == 5:
        excluded = {3, 24, 26}
        choices = [a for a in choices if a not in excluded]

    pref_to_action = {
        0: 24,  # ham
        1: 26,  # bacon
        2: 29,  # plant-based meat
        3: 19,  # egg
        4: 32,  # cheese
    }

    preferred_action = pref_to_action.get(pref)
    if preferred_action in choices:
        return preferred_action
    return random.choice(choices)




if __name__ == "__main__":
    keywords = ["ham", "bacon", "plant-based meat", "egg", "cheese", "vegetarian"]
    print("Select the number corresponding to your preference:")
    for i, keyword in enumerate(keywords):
        print(f"{i}: {keyword}")
    pref = int(input("Enter preference: "))
    num_episodes = int(input("Enter the number of episodes: "))
    
    all_episodes = []
    
    for episode_num in range(num_episodes):
        episode = Episode(episode_num)
        done = False

        # initialize the episode
        step_id = -1
        state = [0] * len(dict_ingredients)
        performed_actions = set() 
        last_action = None

        while not done:
            step_id += 1
            action = sample_action(state, last_action, pref, performed_actions, step_id)
            last_action = action
            performed_actions.add(action)
            state_before = state.copy()
            update_state(state, action)
            state_after = state.copy()
            step = Step(step_id, state_before, action, state_after, 
                        context="human_intervened", robot_prediction=action)
            episode.add_step(step)
            print(dict_actions[action])
            print(state_before)
            print(state_after)
            done = state[-1] == 2

        all_episodes.append(episode)
        print(f"Episode {episode_num} ended. Total: {step_id} steps.")
        print("-----------------")
            
    with open("all_episodes_gen.pkl", "wb") as f:
        pickle.dump(all_episodes, f)

    print("All episodes saved to all_episodes.pkl")



