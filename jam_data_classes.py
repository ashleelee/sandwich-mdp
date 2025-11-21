# classes for logging data
import numpy as np

class TimeStep:
    def __init__(self, step_id, action, state, state_img_path, context, robot_prediction):
        self.step_id = step_id
        self.action = action
        self.state = state
        self.state_img = state_img_path
        self.context = context
        self.robot_prediction = robot_prediction

    def to_dict(self):
        def to_python(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, (np.float32, np.float64, np.int32, np.int64)):
                return x.item()
            return x

        return {
            "step_id": int(self.step_id),
            "action": to_python(self.action),
            "state": to_python(self.state),
            "state_img": self.state_img,
            "context": self.context,
            "robot_prediction": to_python(self.robot_prediction),
        }


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