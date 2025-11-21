import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import pickle
import time
import random
from jam_data_classes import TimeStep, Episode
import os
import json

# init variables for pygame
FPS = 30
VIEWPORT_W = 900
VIEWPORT_H = 700
JAM_WIDTH = 40
DISTANCE_TO_HOLD_BAG = 50
DISTANCE_TO_HOLD_ROBOT_ARM = 40

class JamSpreadingEnv(gym.Env):
    
    def __init__(self):

        """
        ### Action Space
        The action space is continuous with 3 values: absolute x and y positions and a gripper state for the robot.
        
        ### Observation Space
        The observation is a 12-dimensional continuous vector capturing the task state, 
        including robot position and gripper state, whether the robot is holding the bag, 
        and jam coverage across 8 bread segments.

        ### Rewards
        A reward system is not set up for this environment, as it is designed for IL.

        ### Starting State
        The environment starts with the robot at the predefined start position. 
        The gripper is open and not holding the piping bag. 
        The piping bag is placed at its initial location. 
        No jam is spread on the bread.

        ### Episode Termination
        The episode ends when the robot releases the piping bag at its initial (piping bag's) location.
        """

        super().__init__()

        # init pygame variables
        self.screen: pygame.Surface = None
        self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        self.clock = None
        self.load_images()
        self.jam_lines = []
        self.jam_width = JAM_WIDTH
        self.piping_bag_x, self.piping_bag_y= 610.0, 90.0
        self.initializeBoxes()

        

        # state and action variables 
        self.state = np.array([
            90.0, 90.0,      # robot_position (same as start)
            0,             # gripper_state (open)
            0,             # holding_bag (not holding)
            *[0.0] * 8       # jam_coverage
        ], dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([700.0, 700.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=np.array([0.0] * 12, dtype=np.float32),
            high=np.array(
                [700.0, 700.0,   # robot_x, robot_y
                1.0,            # gripper_state
                1.0,            # holding_bag
                *[1.0] * 8      # jam_coverage
                ],
                dtype=np.float32
            ),
            dtype=np.float32
        )

        self.done = False
        self.last_gripper_state = 0.5
        self.been_on_bread = False # check if it's been on bread to avoid completing right after picking up the piping bag

        # logging data
        self.action_log = []
        self.save_interval = 0.0  # seconds
        self.last_save_time = time.time()
        self.save_counter = 0
    
    def load_images(self):
        self.robot_images = {
            "open": pygame.image.load("img_c/open.png").convert_alpha(),
            "hold": pygame.image.load("img_c/hold.png").convert_alpha(),
            "clutch": pygame.image.load("img_c/clutch.png").convert_alpha(),
        }

        self.piping_bag_images = {
            "normal": pygame.image.load("img_c/normal.png").convert_alpha(),
            "squeezed": pygame.image.load("img_c/squeezed.png").convert_alpha(),
        }
        for key in self.robot_images:
            self.robot_images[key] = pygame.transform.scale(self.robot_images[key], (175, 175))  # adjust size as needed
        for key in self.piping_bag_images:
            self.piping_bag_images[key] = pygame.transform.scale(self.piping_bag_images[key], (175, 175))  # adjust size as needed

    def initializeBoxes(self):
        self.box_positions = [
            (160, 155),  # Box 1
            (350, 155),  # Box 2
            (160, 275),  # Box 3
            (350, 275),  # Box 4
            (160, 395),  # Box 5
            (350, 395),  # Box 6
            (160, 515),  # Box 7
            (350, 515)   # Box 8
        ]

        # Each box has the same width and height
        self.box_size = (190, 120)  # (width = 190, height = 120)

        # Precompute box area for coverage calculations
        self.box_area = self.box_size[0] * self.box_size[1]  # 190 * 120 = 22800


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = np.array([
            90.0, 90.0,      # robot_position (same as start)
            0,             # gripper_state (open)
            0,             # holding_bag (not holding)
            *[0.0] * 8       # jam_coverage
        ], dtype=np.float32)

        # reset pygame variables
        self.jam_lines = []

        self.done = False
        self.been_on_bread = False
        self.last_gripper_state = 0.5

        return self.state, {}
    
    def step(self, action):

        # update the state after the action is performed
        self.update_state(action)
        self.action_log.append(np.array(action, dtype=np.float32))

        # check if bread endpoints is hit
        if not self.been_on_bread:
            self.check_been_on_bread()

        print("been on bread", self.been_on_bread)

        # check if final state is reached
        done = self.is_done()

        # pygame screen change
        return self.state, 0, done, False, {} 

    def update_state(self, action):

        self.state[0] = action[0]; # robot_x
        self.state[1] = action[1]; # robot_y
        self.state[2] = action[2]; # gripper state

        # Check if robot is close enough to the piping bag and in 'hold' state
        robot_x, robot_y = self.state[0], self.state[1]
        gripper_state = self.state[2]
        bag_x, bag_y = self.piping_bag_x, self.piping_bag_y

        distance_to_bag = ((robot_x - bag_x) ** 2 + (robot_y - bag_y) ** 2) ** 0.5

        if distance_to_bag <= DISTANCE_TO_HOLD_BAG and gripper_state == 0.5:
            self.state[3] = 1  # holding_bag
        
        if self.state[2] == 1 and self.state[3] == 1: # if holding bag + gripper state is squeeze
            jam_point = (int(self.state[0] + 35), int(self.state[1] + 70)) # jam point offset from the robot arm
            if jam_point not in self.jam_lines: 
                self.jam_lines.append(jam_point)
                self.update_jam_coverage_area()
    
    def update_jam_coverage_area(self):
        box_areas_covered = [0.0] * 8
        w = self.jam_width

        if len(self.jam_lines) < 2:
            return  # need at least 2 points for a line

        for i in range(len(self.jam_lines) - 1):
            x1, y1 = self.jam_lines[i]
            x2, y2 = self.jam_lines[i + 1]

            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            # Estimate jammed area for this segment
            seg_area = max(dx, dy) * w

            # Bounding rect of the segment
            min_x = min(x1, x2) - w // 2
            min_y = min(y1, y2) - w // 2
            max_x = max(x1, x2) + w // 2
            max_y = max(y1, y2) + w // 2
            seg_rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)

            for j, (bx, by) in enumerate(self.box_positions):
                box_rect = pygame.Rect(bx, by, *self.box_size)
                intersection = seg_rect.clip(box_rect)

                if intersection.width > 0 and intersection.height > 0:
                    # Approximate proportion of jam area in this box
                    overlap_area = intersection.width * intersection.height
                    box_areas_covered[j] += (overlap_area / seg_rect.width / seg_rect.height) * seg_area

        # Normalize and update state
        for i in range(8):
            coverage_ratio = min(box_areas_covered[i] / self.box_area, 1.0)
            self.state[4 + i] = coverage_ratio


    def ready_to_help(self):
        """
        Returns True if the user clicks within 30px of the robot or piping bag tip,
        depending on whether the bag is being held.
        """
        mouse_x, mouse_y = pygame.mouse.get_pos()
        holding_bag = self.state[3]
        robot_x, robot_y = self.state[0], self.state[1]

        # Process Pygame events
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                dist = ((mouse_x - robot_x) ** 2 + (mouse_y - robot_y) ** 2) ** 0.5
                return dist <= DISTANCE_TO_HOLD_ROBOT_ARM
            elif event.type == pygame.QUIT:
                pygame.quit()
                exit()
        
        return False
    
    def check_intervene_click(self):
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                button_width, button_height = 120, 40
                button_x = 700 + (200 - button_width) // 2
                button_y = (700 - button_height) // 2
                button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
                return button_rect.collidepoint(mouse_x, mouse_y)
            elif event.type == pygame.QUIT:
                pygame.quit()
                exit()
        return False

    def get_help(self):
        # Get mouse position
        x, y = pygame.mouse.get_pos()

        # Get current gripper state
        current_gripper = self.state[2]

        # Check for mouse click and cycle gripper state
        gripper = current_gripper
        
        for event in pygame.event.get():
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if current_gripper == 0:
                    gripper = 0.5
                elif current_gripper == 0.5:
                    if self.last_gripper_state == 1:
                        gripper = 0
                    else:
                        gripper = 1.0
                else:  # current_gripper == 1.0
                    gripper = 0.5
                self.last_gripper_state = current_gripper
                
            elif event.type == pygame.QUIT:
                pygame.quit()
                exit()

        return np.array([x, y, gripper], dtype=np.float32)

    
    def is_done(self):
        robot_x, robot_y = self.state[0], self.state[1]
        end_x, end_y = self.piping_bag_x, self.piping_bag_y # ending pos is the init pos of piping bag as well

        # tip of the piping bag
        tip_x = robot_x + 35
        tip_y = robot_y + 70

        # distance from tip to bread
        distance = ((robot_x - end_x) ** 2 + (robot_y - end_y) ** 2) ** 0.5

        print("distance:", distance)
        gripper_state = self.state[2]

        if distance <= DISTANCE_TO_HOLD_BAG and self.been_on_bread and gripper_state == 0:
            print("done")
            self.done = True
            return True
        return False
    
    def check_been_on_bread(self):
        # tip of the piping bag relative to robot
        robot_x, robot_y = self.state[0], self.state[1]
        tip_x = robot_x + 35
        tip_y = robot_y + 70

        bread_left = 160
        bread_top = 200
        bread_width = self.box_size[0] * 2   # 2 columns
        bread_height = self.box_size[1] * 4  # 4 rows

        bread_rect = pygame.Rect(bread_left, bread_top, bread_width, bread_height)

        if bread_rect.collidepoint(tip_x, tip_y):
            self.been_on_bread = True

    
    def draw_robot(self):
        gripper_state = self.state[2]
        if gripper_state == 0:
            robot_img = self.robot_images["open"]
        elif gripper_state == 0.5:
            robot_img = self.robot_images["hold"]
        else:
            robot_img = self.robot_images["clutch"]
        robot_pos = (int(self.state[0]), int(self.state[1]))  # robot x, y
        robot_rect = robot_img.get_rect(center=robot_pos)
        self.screen.blit(robot_img, robot_rect)
    
    def draw_piping_bag(self):
        holding_bag = self.state[3]
        gripper_state = self.state[2]

        squeezed = (holding_bag == 1 and gripper_state == 1)
        bag_img = self.piping_bag_images["squeezed" if squeezed else "normal"]

        if holding_bag == 0:
            # bag at its fixed initial position + offset
            bag_pos = (int(self.piping_bag_x), int(self.piping_bag_y))
            bag_rect = (bag_pos[0]-125, bag_pos[1]-87)
            self.screen.blit(bag_img, bag_rect)
        else:
            # attached to robot
            bag_pos = (int(self.state[0]), int(self.state[1]))
            bag_rect = bag_img.get_rect(center=bag_pos)
            self.screen.blit(bag_img, bag_rect)
    
    def draw_jam(self):
        if len(self.jam_lines) > 1:
            pygame.draw.lines(self.screen, (255, 102, 102), False, self.jam_lines, self.jam_width)  

    def draw_coverage_text(self):
        font = pygame.font.SysFont("Arial", 20)
        x = 10
        y_start = VIEWPORT_H - 180  # start higher so it fits 8 lines upward

        for i in range(8):
            coverage = self.state[4 + i]
            text_surface = font.render(f"Box {i+1}: {coverage:.2f}", True, (0, 0, 0))
            self.screen.blit(text_surface, (x, y_start + i * 20))
    
    def draw_box_dividers(self):
        # Bread bounding box
        bread_left = 160
        bread_top = 155
        box_w, box_h = self.box_size

        # Vertical divider (splits left/right)
        x_mid = bread_left + box_w
        pygame.draw.line(self.screen, (0, 0, 0), (x_mid, bread_top), (x_mid, bread_top + box_h * 4), 3)

        # Horizontal dividers (between 4 rows)
        for i in range(1, 4):
            y = bread_top + box_h * i
            pygame.draw.line(self.screen, (0, 0, 0), (bread_left, y), (bread_left + box_w * 2, y), 3)
    
    def draw_intervene_button(self):
        # Draw Intervene button
        button_width, button_height = 120, 40
        button_x = 700 + (200 - button_width) // 2  # horizontally center in the 200px wide panel
        button_y = (700 - button_height) // 2  # vertically center in the 700px high panel

        button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
        pygame.draw.rect(self.screen, (255, 224, 161), button_rect)  # Button color

        # Draw Intervene text
        font = pygame.font.SysFont("Arial", 20, bold=True)
        text_surf = font.render("Intervene", True, (203, 91, 59))  # Darker orange text
        text_rect = text_surf.get_rect(center=button_rect.center)
        self.screen.blit(text_surf, text_rect)

    def draw_intervene_ready_text(self):
        font = pygame.font.SysFont("Arial", 14, bold=True)
        text_lines = [
            "Please click on the dot on",
            "the robot arm and guide",
            "me with your cursor."
        ]
        color = (139, 69, 19)  # A brownish color similar to your image
        x = 715  # Slightly indented inside the side panel
        y_start = 300  # Start vertical position

        for i, line in enumerate(text_lines):
            text_surf = font.render(line, True, color)
            text_rect = text_surf.get_rect(center=(x + 85, y_start + i * 30))  # center in the side panel
            self.screen.blit(text_surf, text_rect)

    def draw_episode_num(self, episode_num):
        font = pygame.font.SysFont("Arial", 16, bold=True)
        ep_text = font.render(f"Episode: {episode_num}", True, (0, 0, 0))
        ep_rect = ep_text.get_rect(topright=(90, 25))
        self.screen.blit(ep_text, ep_rect)


    def render(self, episode_num, step_id, screen="help"):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
            # put these above in main?
        if self.clock is None:
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        
        # fill the screen with white bg
        self.screen.fill((255, 255, 255))

        # draw side panel rectangle (bottom-left)
        side_panel_rect = pygame.Rect(700, 0, 200, 700)
        pygame.draw.rect(self.screen, (247, 243, 238), side_panel_rect)
        if(screen != "ready"):
            self.draw_intervene_button()
        else:
            self.draw_intervene_ready_text()

        # draw bread image
        bread_img = pygame.image.load("img_c/bread.png").convert_alpha()
        bread_img = pygame.transform.scale(bread_img, (550, 550)) 
        bread_pos = ((700 - bread_img.get_width()) // 2,
                    (VIEWPORT_H - bread_img.get_height())-30)
        self.screen.blit(bread_img, bread_pos)

        # draw the initial pos box at top left
        box_color = (255, 191, 64)   # Orange-yellow
        box_size = 80
        start_pos = (50, 50)               # x, y for top-left corner

        # draw jam
        self.draw_jam()

        # save image here
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval:
            filename = f"jam_state_img/train/episode{episode_num}_step{step_id}.png"
            pygame.image.save(self.screen.subsurface((0, 0, 700, 700)), filename)
            self.save_counter += 1
            self.last_save_time = current_time

        """draw the rest"""

         # draw the bowl
        bowl_img = pygame.image.load("img_c/bowl.png").convert_alpha()
        bowl_img = pygame.transform.scale(bowl_img, (100, 100))
        self.screen.blit(bowl_img, (560, 75))

        # draw piping bag
        self.draw_piping_bag()

        # draw robot
        self.draw_robot()

        end_x, end_y = self.piping_bag_x, self.piping_bag_y
        pygame.draw.circle(self.screen, (255, 165, 0), (end_x, end_y), 10) 

        # Draw current screen state text
        font = pygame.font.SysFont("Arial", 16, bold=True)
        state_text = font.render(f"Mode: {screen}", True, (0, 0, 0))  # Black text
        self.screen.blit(state_text, (10, 10))  # top-left corner
        self.draw_episode_num(episode_num)

        self.clock.tick(FPS)
        pygame.display.flip()

def controlled_delay(delay_time):
    # non-blocking delay while keeping pygame responsive
    start_time = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start_time < delay_time:
        pygame.event.pump() 

def wait_for_space():
    print("Press SPACE to start the episode")
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    waiting = False
                    break

        pygame.time.delay(20)   # small sleep so we do not burn CPU


def collect_training_data():
    # num_episodes = int(input("Enter number of episodes to run: "))
    episodes_per_batch = 10

    batch_idx = int(input("Enter batch index (1, 2, 3, ...): "))
    num_episodes = episodes_per_batch

    start_episode_id = (batch_idx - 1) * episodes_per_batch
    
    save_dir = "data"
    os.makedirs(save_dir, exist_ok=True)

    base_name = f"jam_train_data_{batch_idx}"
    pkl_path = os.path.join(save_dir, base_name + ".pkl")
    json_path = os.path.join(save_dir, base_name + ".json")

    pygame.init()
    pygame.display.init()
    env = JamSpreadingEnv()

    all_episodes = []

    for local_ep in range(num_episodes):
        global_ep_id = start_episode_id + local_ep

        obs, _ = env.reset()
        done = False
        episode = Episode(episode_id=global_ep_id)
        step_id = 0

        print(f"Starting episode {global_ep_id}")

        # optional: draw first frame and wait for space
        env.render(global_ep_id, step_id, screen="help")
        wait_for_space()

        while not done:
            action = env.get_help()
            obs, _, done, _, _ = env.step(action)

            img_path = f"jam_state_img/train/episode{global_ep_id}_step{step_id}.png"

            timestep = TimeStep(
                step_id=step_id,
                action=action.copy(),
                state=obs.copy(),
                state_img_path=img_path,
                context="human_control",
                robot_prediction=None
            )
            episode.add_step(timestep)

            env.render(global_ep_id, step_id, screen="help")
            step_id += 1
            time.sleep(1 /10) # 10 timesteps per second

        print(f"Episode {global_ep_id} ended.")
        all_episodes.append(episode)
        controlled_delay(3000)


    # save all episodes as dicts
    data_to_save = [ep.to_dict() for ep in all_episodes]

    # pickle
    with open(pkl_path, "wb") as f:
        pickle.dump(data_to_save, f)

    # json
    with open(json_path, "w") as f:
        json.dump(data_to_save, f, indent=2)

    print(f"Saved pickle to {pkl_path}")
    print(f"Saved json to {json_path}")

    env.close()
    if pygame.get_init():
        pygame.quit()

    print("Environment closed successfully.")


if __name__ == "__main__":
    collect_training_data()