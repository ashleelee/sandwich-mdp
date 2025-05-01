import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import pickle
import time
import random
# from jam_sample_actions import jam_sample_actions
from jam_data_classes import TimeStep, Episode
import os

# init variables for pygame
FPS = 30
VIEWPORT_W = 900
VIEWPORT_H = 700

# Load the pickled episodes
with open("jam_all_episodes_gen.pkl", "rb") as f:
    episodes = pickle.load(f)

# Extract episode 0 actions
ep0 = episodes[0]
jam_sample_actions = [step.action.tolist() for step in ep0.steps]

class JamSpreadingEnv(gym.Env):
    
    def __init__(self):

        """
        ### Action Space
        The action space is continuous with 3 values: absolute x and y positions and a gripper state for the robot.
        
        ### Observation Space
        The observation is a 18-dimensional continuous vector capturing the task state, 
        including the start position, piping bag location, the bread's endpoint, 
        robot position and gripper state, whether the robot is holding the bag, 
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
        self.jam_width = 40
        self.initializeBoxes()
        self.action_log = []

        #delete later
        self.save_interval = 3.0  # seconds
        self.last_save_time = time.time()
        self.save_counter = 0


        self.state = np.array([
            90.0, 90.0,      # start_position
            610.0, 90.0,     # piping_bag_initial_pos
            200.0, 580.0,    # bread_endpoint
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
            low=np.array([0.0] * 18, dtype=np.float32),
            high=np.array(
                [700.0, 700.0,   # start_position * change
                700.0, 700.0,   # piping_bag_initial_pos * change
                700.0, 700.0,   # bread_endpoint * change
                700.0, 700.0,   # robot_position
                1.0,            # gripper_state
                1.0,            # holding_bag
                *[1.0] * 8      # jam_coverage
                ],
                dtype=np.float32
            ),
            dtype=np.float32
        )

        self.done = False
        self.hit_bread_endpoints = False
        self.last_gripper_state = 0.5
    
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
        self.box_size = (190, 120)  # (Width, Height)

        # Precompute box area for coverage calculations
        self.box_area = self.box_size[0] * self.box_size[1]  # 190 * 120 = 22800


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = np.array([
            90.0, 90.0,      # start_position
            610.0, 90.0,     # piping_bag_initial_pos
            200.0, 580.0,    # bread_endpoint
            90.0, 90.0,      # robot_position (same as start)
            0,             # gripper_state (open)
            0,             # holding_bag (not holding)
            *[0.0] * 8       # jam_coverage
        ], dtype=np.float32)

        # reset pygame variables
        self.jam_lines = []

        self.done = False
        self.hit_bread_endpoints = False
        self.last_gripper_state = 0.5

        return self.state, {}
    
    def step(self, action):

        # update the state after the action is performed
        self.update_state(action)
        self.action_log.append(np.array(action, dtype=np.float32))

        # check if bread endpoints is hit
        self.check_hit_bread_endpoints()

        # check if final state is reached
        done = self.is_done()

        # pygame screen change
        return self.state, 0, done, False, {} 

    def update_state(self, action):

        self.state[6] = action[0];
        self.state[7] = action[1];
        self.state[8] = action[2];

        # Check if robot is close enough to the piping bag and in 'hold' state
        robot_x, robot_y = self.state[6], self.state[7]
        bag_x, bag_y = self.state[2], self.state[3]
        gripper_state = self.state[8]

        distance_to_bag = ((robot_x - bag_x) ** 2 + (robot_y - bag_y) ** 2) ** 0.5

        if distance_to_bag <= 40 and gripper_state == 0.5:
            self.state[9] = 1  # holding_bag
        
        if self.state[8] == 1 and self.state[9] == 1:
            jam_point = (int(self.state[6] + 35), int(self.state[7] + 70))
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
            self.state[10 + i] = coverage_ratio


    def ready_to_help(self):
        """
        Returns True if the user clicks within 30px of the robot or piping bag tip,
        depending on whether the bag is being held.
        """
        mouse_x, mouse_y = pygame.mouse.get_pos()
        holding_bag = self.state[9]
        robot_x, robot_y = self.state[6], self.state[7]

        # Process Pygame events
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                dist = ((mouse_x - robot_x) ** 2 + (mouse_y - robot_y) ** 2) ** 0.5
                return dist <= 30
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
        current_gripper = self.state[8]

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
        robot_x, robot_y = self.state[6], self.state[7]
        end_x, end_y = 610, 140

        # tip of the piping bag
        tip_x = robot_x + 35
        tip_y = robot_y + 70

        # distance from tip to bread
        distance = ((tip_x - end_x) ** 2 + (tip_y - end_y) ** 2) ** 0.5

        gripper_state = self.state[8]

        if distance <= 25 and self.hit_bread_endpoints and gripper_state == 0:
            print("done")
            self.done = True
            return True
        return False
    
    def check_hit_bread_endpoints(self):
        robot_x, robot_y = self.state[6], self.state[7]
        bread_x, bread_y = self.state[4], self.state[5]

        # tip of the piping bag
        tip_x = robot_x + 35
        tip_y = robot_y + 70

        distance = ((tip_x - bread_x) ** 2 + (tip_y - bread_y) ** 2) ** 0.5
        if distance <= 25:
            print("hit!")
            self.hit_bread_endpoints = True
    
    def draw_robot(self):
        gripper_state = self.state[8]
        if gripper_state == 0:
            robot_img = self.robot_images["open"]
        elif gripper_state == 0.5:
            robot_img = self.robot_images["hold"]
        else:
            robot_img = self.robot_images["clutch"]
        robot_pos = (int(self.state[6]), int(self.state[7]))  # robot x, y
        robot_rect = robot_img.get_rect(center=robot_pos)
        self.screen.blit(robot_img, robot_rect)
    
    def draw_piping_bag(self):
        holding_bag = self.state[9]
        bag_img = self.piping_bag_images["squeezed" if holding_bag == 1 and self.state[8] == 1 else "normal"]
        if holding_bag == 0:
            bag_pos = (int(self.state[2]), int(self.state[3]))  # initial piping bag position
            bag_rect = (bag_pos[0]-125, bag_pos[1]-87)
            self.screen.blit(bag_img, bag_rect)
        else:
            bag_pos = (int(self.state[6]), int(self.state[7])) # robot pos
            bag_rect = bag_img.get_rect(center=bag_pos)
        self.screen.blit(bag_img, bag_rect)
    
    def draw_jam(self):
        if len(self.jam_lines) > 1:
            pygame.draw.lines(self.screen, (255, 0, 0), False, self.jam_lines, self.jam_width)  

    def draw_coverage_text(self):
        font = pygame.font.SysFont("Arial", 20)
        x = 10
        y_start = VIEWPORT_H - 180  # start higher so it fits 8 lines upward

        for i in range(8):
            coverage = self.state[10 + i]
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



    def render(self, screen="help"):
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

        self.draw_intervene_button()

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
        # pygame.draw.rect(self.screen, box_color, (*start_pos, box_size, box_size), width=7)
        
       
        # self.draw_box_dividers()

        # draw jam
        self.draw_jam()

        # save image here
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval:
            filename = f"jam_state_img/test_jam_pic{self.save_counter}.png"
            pygame.image.save(self.screen.subsurface((0, 0, 700, 700)), filename)
            # print(f"Saved image: {filename}")
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

        # self.draw_coverage_text()

        bread_x, bread_y = int(self.state[4]), int(self.state[5])
        pygame.draw.circle(self.screen, (255, 165, 0), (bread_x, bread_y), 25)

        end_x, end_y = int(self.state[2]), int(self.state[3])
        pygame.draw.circle(self.screen, (255, 165, 0), (610, 140), 25) 

        # Draw current screen state text
        font = pygame.font.SysFont("Arial", 24, bold=True)
        state_text = font.render(f"Mode: {screen}", True, (0, 0, 0))  # Black text
        self.screen.blit(state_text, (10, 10))  # top-left corner

        self.clock.tick(FPS)
        pygame.display.flip()

def controlled_delay(delay_time):
    # non-blocking delay while keeping pygame responsive
    start_time = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start_time < delay_time:
        pygame.event.pump() 

# classes for logging data
class TimeStep:
    def __init__(self, step_id, action, state_v, state_img_path, context, robot_prediction):
        self.step_id = step_id
        self.action = action
        self.state_v = state_v
        self.state_img = state_img_path
        self.context = context  # "robot_independent", "robot_asked", or "human_intervened"
        self.robot_prediction = robot_prediction
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

def create_syn_data(actions):
    import os
    save_path = os.path.join("jam_sample_action_syn_data", "jam_sample_action_cross.py")

    with open(save_path, "w") as f:
        f.write("import numpy as np\n\n")
        f.write("jam_sample_actions = [\n")
        for action in actions:
            rounded = [round(float(x), 2) for x in action]  # ensure float64 for nice formatting
            f.write(f"    np.array({rounded}, dtype=np.float32),\n")
        f.write("]\n")
    
if __name__ == "__main__":
    num_episodes = int(input("Enter number of episodes to run: "))
    all_episodes = []
    pygame.init()
    pygame.display.init()
    env = JamSpreadingEnv()
    action = None
    context = None

    for episode_num in range(num_episodes):
        # actions = []
        obs, _ = env.reset()
        done = False
        next_action_idx = 0
        next_action = jam_sample_actions[next_action_idx]
        help_start_time = None
        screen_state = "robot"
        last_help_check_time = time.time()
        episode = Episode(episode_num)
        step_id = 0 

        while not done:
            
            robot_prediction = next_action

            if screen_state == "robot":
                robot_prediction = next_action
                need_help = False
                # if time.time() - last_help_check_time >= 3.0:
                #     need_help = random.random() < 0.5

                if env.check_intervene_click():
                    screen_state = "ready"
                    context = "human_intervened"
                elif need_help:
                    screen_state = "ready"
                    context = "robot_asked"
                else:
                    # obs, _, done, _, _ = env.step(next_action)
                    action = next_action
                    context = "robot_independent"

                    next_action_idx += 1
                    if next_action_idx < len(jam_sample_actions):
                        next_action = jam_sample_actions[next_action_idx]
                    screen_state = "robot"

            elif screen_state == "ready":
                if env.ready_to_help():
                    screen_state = "help"
                    help_start_time = time.time()
                else:
                    env.render(screen_state)
                    time.sleep(1 / 10)
                    continue

            elif screen_state == "help":
                # let human intervene
                action = env.get_help()
                # obs, _, done, _, _ = env.step(action)

                # check if help time has expired (3 seconds)
                if help_start_time and time.time() - help_start_time >= 3.0:
                    # find nearest (x, y) in jam_sample_actions to robot position
                    robot_x, robot_y = env.state[6], env.state[7]
                    min_dist = float('inf')
                    best_idx = next_action_idx  # fallback to current index

                    for i in range(len(jam_sample_actions)):
                        act_x, act_y = jam_sample_actions[i][0], jam_sample_actions[i][1]
                        dist = ((act_x - robot_x) ** 2 + (act_y - robot_y) ** 2) ** 0.5
                        if dist < min_dist:
                            min_dist = dist
                            best_idx = i

                    # set the new next_action to the closest
                    next_action_idx = best_idx
                    next_action = jam_sample_actions[next_action_idx]
                    last_help_check_time = time.time()
                    screen_state = "robot"

            if action is not None:
                obs, _, done, _, _ = env.step(action)
                # actions.append(action)
                # print(robot_prediction)
                # print(context)
            env.render(screen_state)
            time.sleep(1 / 10)

        print(f"Episode {episode_num} ended.")
        controlled_delay(3000)
        # create_syn_data(actions)

    
    env.close()  # Close the Gym environment
    del env  # Explicitly delete the environment

    if pygame.get_init():
        pygame.quit()
    
    print("Environment closed successfully.")
