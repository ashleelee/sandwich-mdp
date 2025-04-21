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
    
class TimeStep:
    def __init__(self, step_id, action, state_v, state_img_path, context, robot_prediction):
        self.step_id = step_id
        self.action = action
        self.state_v = state_v
        self.state_img = state_img_path
        self.context = context  # "robot_independent", "robot_asked", or "human_intervened"
        self.robot_prediction = robot_prediction