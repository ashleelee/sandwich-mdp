import pickle
import json
import random
import numpy as np
import pygame
import os

def render_state(screen, jam_lines, save_path):
    screen.fill((255, 255, 255))

    bread_img = pygame.image.load("img_c/bread.png").convert_alpha()
    bread_img = pygame.transform.scale(bread_img, (550, 550)) 
    bread_pos = ((700 - bread_img.get_width()) // 2,
                 (700 - bread_img.get_height()) - 30)
    screen.blit(bread_img, bread_pos)

    if len(jam_lines) > 1:
        pygame.draw.lines(screen, (255, 0, 0), False, jam_lines, jam_width)  

    pygame.image.save(screen.subsurface((0, 0, 700, 700)), save_path)


# Global constants for jam coverage
jam_width = 40
box_positions = [
    (160, 155),  # Box 1
    (350, 155),  # Box 2
    (160, 275),  # Box 3
    (350, 275),  # Box 4
    (160, 395),  # Box 5
    (350, 395),  # Box 6
    (160, 515),  # Box 7
    (350, 515)   # Box 8
]
box_size = (190, 120)
box_area = box_size[0] * box_size[1]

def update_state(state, action, jam_lines):

    state[6] = action[0];
    state[7] = action[1];
    state[8] = action[2];

    # Check if robot is close enough to the piping bag and in 'hold' state
    robot_x, robot_y = state[6], state[7]
    bag_x, bag_y = state[2], state[3]
    gripper_state = state[8]

    distance_to_bag = ((robot_x - bag_x) ** 2 + (robot_y - bag_y) ** 2) ** 0.5

    if distance_to_bag <= 40 and gripper_state == 0.5:
        state[9] = 1  # holding_bag
    
    if state[8] == 1 and state[9] == 1:
        jam_point = (int(state[6] + 35), int(state[7] + 70))
        if jam_point not in jam_lines: 
            jam_lines.append(jam_point)
            update_jam_coverage_area(state, jam_lines)

def update_jam_coverage_area(state, jam_lines):
    global jam_width, box_positions, box_size, box_area
    box_areas_covered = [0.0] * 8
    w = jam_width

    if len(jam_lines) < 2:
        return

    for i in range(len(jam_lines) - 1):
        x1, y1 = jam_lines[i]
        x2, y2 = jam_lines[i + 1]

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        seg_area = max(dx, dy) * w

        min_x = min(x1, x2) - w / 2
        min_y = min(y1, y2) - w / 2
        max_x = max(x1, x2) + w / 2
        max_y = max(y1, y2) + w / 2

        for j, (bx, by) in enumerate(box_positions):
            box_min_x = bx
            box_min_y = by
            box_max_x = bx + box_size[0]
            box_max_y = by + box_size[1]

            overlap_min_x = max(min_x, box_min_x)
            overlap_max_x = min(max_x, box_max_x)
            overlap_min_y = max(min_y, box_min_y)
            overlap_max_y = min(max_y, box_max_y)

            if overlap_min_x < overlap_max_x and overlap_min_y < overlap_max_y:
                overlap_w = overlap_max_x - overlap_min_x
                overlap_h = overlap_max_y - overlap_min_y
                overlap_area = overlap_w * overlap_h

                seg_rect_area = (max_x - min_x) * (max_y - min_y)
                if seg_rect_area > 0:
                    box_areas_covered[j] += (overlap_area / seg_rect_area) * seg_area

    for i in range(8):
        coverage_ratio = min(box_areas_covered[i] / box_area, 1.0)
        state[10 + i] = coverage_ratio

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

if __name__ == "__main__":
    keywords = ["cross", "spiral", "square", "zig-zag"]
    print("Select the number corresponding to your preference:")
    for i, keyword in enumerate(keywords):
        print(f"{i}: {keyword}")
    pref = int(input("Enter preference: "))
    num_episodes = int(input("Enter the number of episodes: "))
    save_images_input = input("Save images for each step? (y/n): ").strip().lower()
    save_images = save_images_input == "y"

    if save_images:
        if os.path.exists("jam_state_img"):
            for filename in os.listdir("jam_state_img"):
                file_path = os.path.join("jam_state_img", filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        else:
            os.makedirs("jam_state_img", exist_ok=True)
        pygame.init()
        screen = pygame.display.set_mode((700, 700))
        pygame.display.set_caption("Jam Render")
    else:
        screen = None
    
    all_episodes = []
    if pref == 0:
        from jam_sample_action_syn_data.jam_sample_action_cross import jam_sample_actions
    elif pref == 1:
        from jam_sample_action_syn_data.jam_sample_action_spiral import jam_sample_actions
    elif pref == 2:
        from jam_sample_action_syn_data.jam_sample_action_square import jam_sample_actions
    else:  # pref == 3
        from jam_sample_action_syn_data.jam_sample_action_zigzag import jam_sample_actions

    for episode_num in range(num_episodes):
        episode = Episode(episode_num)
        jam_lines = []

        # Initialize environment state
        state = np.array([
            90.0, 90.0,      # start_position
            610.0, 90.0,     # piping_bag_initial_pos
            200.0, 580.0,    # bread_endpoint
            90.0, 90.0,      # robot_position (same as start)
            0,               # gripper_state (open)
            0,               # holding_bag (not holding)
            *[0.0] * 8       # jam_coverage
        ], dtype=np.float32)

        for step_id, base_action in enumerate(jam_sample_actions):
            state_before = state.copy()

            # Add random dx, dy in [5, 10]
            dx = random.randint(-15, 15)
            dy = random.randint(-15, 15)

            action = base_action.copy()
            if (action[2] == 1):
                action[0] += dx
                action[1] += dy

            if save_images:
                img_filename = f"jam_state_img/episode{episode_num}_step{step_id}.png"
                render_state(screen, jam_lines, img_filename)
            else:
                img_filename = None

            update_state(state, action, jam_lines)
            # state_after = state.copy()

            step = TimeStep(
                step_id=step_id,
                action=action,
                state_v=state_before,
                state_img_path=img_filename,
                context="human_intervened",
                robot_prediction=None
            )
            episode.add_step(step)

        all_episodes.append(episode)
        print(f"Episode {episode_num} completed. Steps: {len(jam_sample_actions)}")

        print("-----------------")
            
    with open("jam_all_episodes_gen.pkl", "wb") as f:
        pickle.dump(all_episodes, f)

    print("All episodes saved to jam_all_episodes_gen.pkl")
    if save_images:
        pygame.quit()