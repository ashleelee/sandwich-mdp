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