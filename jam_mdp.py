import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import pickle
import time
import random
import math

from jam_data_classes import TimeStep, Episode
import os
from policies.conformal.mlp import Continuous_Policy
import torch, pdb

# init variables for pygame
FPS = 30
VIEWPORT_W = 900
VIEWPORT_H = 700
JAM_WIDTH = 40
DISTANCE_TO_HOLD_BAG = 30
DISTANCE_TO_HOLD_ROBOT_ARM = 40
UNCERTAINTY_THRESHOLD = 20

# stats for policy
shape = "square"

# policy path
POLICY_PATH = f"trained_policy/cont_policy_{shape}.pth"

# load normalization stats
norm_stats = np.load(f"trained_policy/norm_stats_{shape}.npz")
min_X = norm_stats["min_X"]
max_X = norm_stats["max_X"]
min_Y = norm_stats["min_Y"]
max_Y = norm_stats["max_Y"]

# avoid division by zero
range_X = max_X - min_X
range_X[range_X == 0] = 1.0
range_Y = max_Y - min_Y
range_Y[range_Y == 0] = 1.0

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
        self.piping_bag_x, self.piping_bag_y= 120.0, 90.0
        self.bowl_x, self.bowl_y = self.piping_bag_x - 50, 75.0
        self.initializeBoxes()

        # state and action variables 
        self.state = np.array([
            90.0, 90.0,      # robot_position (same as start)
            0.5,             # gripper_state (open)
            1,             # holding_bag (not holding)
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

        self.uncertainty_history = []
    
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
            0.5,             # gripper_state (open)
            1,             # holding_bag (not holding)
            *[0.0] * 8       # jam_coverage
        ], dtype=np.float32)

        # reset pygame variables
        self.jam_lines = []

        self.done = False
        self.been_on_bread = False
        self.last_gripper_state = 0.5

        # save for uncertainty graphing
        self.uncertainty_history = []

        return self.state, {}
    
    def step(self, action):

        # update the state after the action is performed
        self.update_state(action)
        self.action_log.append(np.array(action, dtype=np.float32))

        # check if bread endpoints is hit
        if not self.been_on_bread:
            self.check_been_on_bread()

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
        Returns True if the user clicks within 40px of the robot or piping bag tip,
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

    def get_help(self):
        # Get mouse position
        x, y = pygame.mouse.get_pos()

        # Get current gripper state
        current_gripper = self.state[2]

        # Check for mouse click and cycle gripper state
        gripper = current_gripper
        on_bread = self.is_on_bread(x, y)
        
        for event in pygame.event.get():
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if not on_bread:
                    if current_gripper == 0:
                        gripper = 0.5
                    elif gripper == 0.5:
                        gripper = 0
                    else:
                        gripper = 0.5

                else:
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
        end_x, end_y = 80, 80 # ending pos is the init pos of piping bag as well

        # distance from tip to bread
        distance = ((robot_x - end_x) ** 2 + (robot_y - end_y) ** 2) ** 0.5

        gripper_state = self.state[2]
        if (distance <= DISTANCE_TO_HOLD_BAG and self.been_on_bread and gripper_state == 0.5):
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
        bread_top = 200 # lower bread top to lower sensitivity
        bread_width = self.box_size[0] * 2   # 2 columns
        bread_height = self.box_size[1] * 4  # 4 rows

        bread_rect = pygame.Rect(bread_left, bread_top, bread_width, bread_height)

        if bread_rect.collidepoint(tip_x, tip_y):
            self.been_on_bread = True
    
    def is_on_bread(self, x, y):
        # The bread area boundaries defined in check_been_on_bread
        bread_left = 100
        bread_top = 100 
        bread_width = self.box_size[0] * 2   # 2 columns
        bread_height = self.box_size[1] * 4  # 4 rows

        bread_rect = pygame.Rect(bread_left, bread_top, bread_width, bread_height)

        # Check if the tip (mouse) position is within the bread area
        return bread_rect.collidepoint(x, y)

    
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
            pygame.draw.lines(self.screen, (255, 0, 0), False, self.jam_lines, self.jam_width)  
    
    def draw_button(self, label):
        rect = self._panel_button_rect()
        pygame.draw.rect(self.screen, (255, 224, 161), rect)
        font = pygame.font.SysFont("Arial", 20, bold=True)
        txt = font.render(label, True, (203, 91, 59))
        self.screen.blit(txt, txt.get_rect(center=rect.center))

    def draw_intervene_ready_text(self):
        panel_x = 700
        panel_width = 200
        center_x = panel_x + panel_width // 2

        # same color as mode text
        text_color = (163, 74, 53)

        # fonts
        big_font = pygame.font.SysFont("Arial", 18, bold=True)
        med_font = pygame.font.SysFont("Arial", 18, bold=False)

        # lines and emphasis plan
        lines = [
            ("Please",       med_font),
            ("click on the dot", big_font),
            ("on the robot arm", med_font),
            ("to guide me",      med_font),
            ("with your cursor", med_font),
        ]

        # starting y slightly above your screenshot's text block
        y_start = 480
        line_spacing = 28

        for i, (text, font) in enumerate(lines):
            surf = font.render(text, True, text_color)
            rect = surf.get_rect(center=(center_x, y_start + i * line_spacing))
            self.screen.blit(surf, rect)
    
    def draw_takeover_text(self):
        panel_x = 700
        panel_width = 200
        center_x = panel_x + panel_width // 2

        # colors
        alert_color = (180, 40, 30)      # stronger red tone
        body_color  = (163, 74, 53)      # same as Mode text

        # fonts
        title_font = pygame.font.SysFont("Arial", 20, bold=True)
        body_font  = pygame.font.SysFont("Arial", 16, bold=False)

        lines = [
            ("Robot needs help!", title_font, alert_color),
            ("Click on the dot",  body_font,  body_color),
            ("on the robot arm",  body_font,  body_color),
            ("to take over.",     body_font,  body_color),
        ]

        y_start = 480
        spacing = 28

        for i, (text, font, color) in enumerate(lines):
            surf = font.render(text, True, color)
            rect = surf.get_rect(center=(center_x, y_start + i * spacing))
            self.screen.blit(surf, rect)

    def draw_episode_num(self, episode_num):
        font = pygame.font.SysFont("Arial", 16, bold=True)
        ep_text = font.render(f"Episode: {episode_num}", True, (0, 0, 0))
        ep_rect = ep_text.get_rect(topright=(90, 25))
        self.screen.blit(ep_text, ep_rect)

    def draw_side_panel(self, episode_num, mode_text):
        # panel geometry
        panel_x = 700
        panel_width = 200
        panel_height = 700
        panel_rect = pygame.Rect(panel_x, 0, panel_width, panel_height)

        # colors
        panel_bg   = (250, 245, 237)   # very light beige
        text_color = (163, 74, 53)     # brownish red

        # draw background
        pygame.draw.rect(self.screen, panel_bg, panel_rect)

        center_x = panel_rect.centerx

        # fonts
        title_font = pygame.font.SysFont("Arial", 20, bold=True)
        value_font = pygame.font.SysFont("Arial", 18, bold=False)

        # --- Episode block ---
        label = title_font.render("Episode:", True, text_color)
        label_rect = label.get_rect(center=(center_x, 290))
        self.screen.blit(label, label_rect)

        ep_text = value_font.render(f"{episode_num}/10", True, text_color)
        ep_rect = ep_text.get_rect(center=(center_x, 320))
        self.screen.blit(ep_text, ep_rect)

        # --- Mode block ---
        mode_label = title_font.render("Mode:", True, text_color)
        mode_label_rect = mode_label.get_rect(center=(center_x, 380))
        self.screen.blit(mode_label, mode_label_rect)

        mode_surf = value_font.render(mode_text, True, text_color)
        mode_rect = mode_surf.get_rect(center=(center_x, 410))
        self.screen.blit(mode_surf, mode_rect)
    
    def draw_human_control(self):
        panel_x = 700
        panel_width = 200
        center_x = panel_x + panel_width // 2

        text_color = (163, 74, 53)   

        # fonts
        title_font = pygame.font.SysFont("Arial", 18, bold=True)
        body_font  = pygame.font.SysFont("Arial", 16)

        # message lines
        lines = [
            ('Press "SPACE"', title_font),
            ("to return to",  body_font),
            ("robot control", body_font),
        ]

        # place this slightly below the middle of the side panel
        y_start = 480
        spacing = 26

        for i, (text, font) in enumerate(lines):
            surf = font.render(text, True, text_color)
            rect = surf.get_rect(center=(center_x, y_start + i * spacing))
            self.screen.blit(surf, rect)


    def draw_episode_complete_text(self):
        panel_x = 700
        panel_width = 200
        center_x = panel_x + panel_width // 2

        text_color = (34, 115, 76)

        title_font = pygame.font.SysFont("Arial", 22, bold=True)
        body_font  = pygame.font.SysFont("Arial", 18)

        lines = [
            ("Episode", title_font),
            ("Complete!", title_font),
        ]

        # near the bottom of the side panel
        y_start = 560       # adjust slightly if needed
        spacing = 28

        for i, (text, font) in enumerate(lines):
            surf = font.render(text, True, text_color)
            rect = surf.get_rect(center=(center_x, y_start + i * spacing))
            self.screen.blit(surf, rect)


    def _panel_button_rect(self):
        button_width, button_height = 150, 40
        panel_x = 700
        panel_width = 200
        panel_height = 700

        x = panel_x + (panel_width - button_width) // 2
        y = int(panel_height * 0.75) - button_height // 2  
        return pygame.Rect(x, y, button_width, button_height)

    def check_button_click(self):
        rect = self._panel_button_rect()
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if rect.collidepoint(*event.pos):
                    return True
            elif event.type == pygame.QUIT:
                pygame.quit()
                exit()
        return False

    def draw_uncertainty_graph(self, surface, threshold=UNCERTAINTY_THRESHOLD):
        hist = self.uncertainty_history
        if len(hist) < 2:
            return

        # graph placement
        graph_width = 200
        graph_height = 100
        origin_x = 700
        origin_y = 130

        # determine scaling
        max_hist = max(hist)
        max_u = max(max_hist, threshold, 1.0)   # ensure nonzero scale

        # draw background and border
        pygame.draw.rect(surface, (30, 30, 30),
                        (origin_x, origin_y, graph_width, graph_height))
        pygame.draw.rect(surface, (200, 200, 200),
                        (origin_x, origin_y, graph_width, graph_height), 1)

        # build curve points
        n = len(hist)
        points = []
        for i, u in enumerate(hist):
            x = origin_x + i / (n - 1) * graph_width
            y = origin_y + graph_height - (u / max_u) * graph_height
            points.append((x, y))

        # draw uncertainty curve (green)
        pygame.draw.lines(surface, (0, 255, 0), False, points, 2)

        # draw threshold line (white)
        threshold_y = origin_y + graph_height - (threshold / max_u) * graph_height
        pygame.draw.line(surface, (255, 255, 255),
                        (origin_x, threshold_y),
                        (origin_x + graph_width, threshold_y), 2)

        # print latest uncertainty value
        latest_u = hist[-1]
        panel_center_x = origin_x + graph_width / 2

        font = pygame.font.SysFont("Arial", 16)
        font.set_bold(True)     # make the text bold
        status_font = pygame.font.SysFont("Arial", 16, bold=True)

        text1 = font.render(f"uncertainty={latest_u:.2f}", True, (163, 74, 53))
        text2 = font.render(f"threshold={threshold:.1f}", True, (163, 74, 53))

        text1_rect = text1.get_rect()
        text2_rect = text2.get_rect()

        panel_center_x = origin_x + graph_width / 2

        text1_rect.centerx = panel_center_x
        text1_rect.y = origin_y - 80

        text2_rect.centerx = panel_center_x
        text2_rect.y = origin_y - 60

        surface.blit(text1, text1_rect)
        surface.blit(text2, text2_rect)

        # status bar 
        bar_x = origin_x                
        bar_y = origin_y - 24           # slightly above graph
        bar_w = graph_width             
        bar_h = 24                      # height of the bar

        if latest_u > threshold:
            label_text = "Robot Uncertain"
            bg_color = (255, 210, 210)   # light red
            fg_color = (180, 40, 40)     # red
        else:
            label_text = "Robot Certain"
            bg_color = (210, 255, 210)   # light green
            fg_color = (40, 120, 40)     # green

        # draw full-width bar
        pygame.draw.rect(surface, bg_color, (bar_x, bar_y, bar_w, bar_h))

        # centered label text
        status_font = pygame.font.SysFont("Arial", 16, bold=True)
        label_surf = status_font.render(label_text, True, fg_color)
        label_rect = label_surf.get_rect(center=(bar_x + bar_w // 2, bar_y + bar_h // 2))
        surface.blit(label_surf, label_rect)



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

        ############# SIDE PANEL #############

        # choose mode text from screen_state
        if screen_state == "start":
            mode_text = "Robot Operating"
        elif screen_state == "robot":
            mode_text = "Robot Operating"
        elif screen_state == "ready_help":
            mode_text = "Robot Requesting Help"
        elif screen_state == "ready_intervene":
            mode_text = "Human Taking Over"
        elif screen_state == "help":
            mode_text = "Human Assisting"
        elif screen_state == "done":
            mode_text = "Complete"
        else:
            mode_text = screen_state

        self.draw_side_panel(episode_num, mode_text)

        # draw the button for this screen_state
        if screen_state == "start":
            self.draw_button("START")
        elif screen_state == "robot":
            self.draw_button("INTERVENE")
        elif screen_state == "ready_help":
            self.draw_takeover_text()
        elif screen_state == "ready_intervene":
            self.draw_intervene_ready_text()
        elif screen_state == "help":
            self.draw_human_control()
        elif screen_state == "done":
            self.draw_episode_complete_text()
        
        ############# MAIN PANEL #############

        # draw bread image
        bread_img = pygame.image.load("img_c/bread.png").convert_alpha()
        bread_img = pygame.transform.scale(bread_img, (550, 550)) 
        bread_pos = ((700 - bread_img.get_width()) // 2,
                    (VIEWPORT_H - bread_img.get_height())-30)
        self.screen.blit(bread_img, bread_pos)

        # draw jam
        self.draw_jam()

        # save image here
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval:
            filename = f"jam_state_img/test/episode{episode_num}_step{step_id}.png"
            pygame.image.save(self.screen.subsurface((0, 0, 700, 700)), filename)
            self.save_counter += 1
            self.last_save_time = current_time

        """draw the rest"""

         # draw the bowl
        bowl_img = pygame.image.load("img_c/bowl.png").convert_alpha()
        bowl_img = pygame.transform.scale(bowl_img, (100, 100))
        self.screen.blit(bowl_img, (self.bowl_x, self.bowl_y))

        # draw piping bag
        self.draw_piping_bag()

        # draw robot
        self.draw_robot()

        self.draw_uncertainty_graph(self.screen, UNCERTAINTY_THRESHOLD)

        self.clock.tick(FPS)
        pygame.display.flip()

def controlled_delay(delay_time):
    # non-blocking delay while keeping pygame responsive
    start_time = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start_time < delay_time:
        pygame.event.pump()

def get_prediction(obs, obs_prev1, obs_prev2, policy_model):
    """
    obs, obs_prev1, obs_prev2: 12-d states from env (robot_x, robot_y, gripper, holding_bag, 8 coverage)
    We concat [prev2, prev1, current] → 36-d input, normalize with saved stats, run policy, unnormalize action.
    """
    # ensure numpy float32
    s0 = np.array(obs_prev2, dtype=np.float32)
    s1 = np.array(obs_prev1, dtype=np.float32)
    s2 = np.array(obs,        dtype=np.float32)

    input_obs = np.concatenate([s0, s1, s2], axis=0)  # shape (36,)

    # normalize
    x_norm = (input_obs - min_X) / range_X

    state_tensor = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        y_norm = policy_model(state_tensor).cpu().numpy()[0]   # shape (3,)

    # unnormalize action
    action = y_norm * range_Y + min_Y

    return action.astype(np.float32)

def similar_state(current_state, prev_state):

    current_robot_x = current_state[0]
    current_robot_y = current_state[1]

    prev_robot_x = prev_state[0]
    prev_robot_y = prev_state[1]

    dist = ((current_robot_x - prev_robot_x) ** 2 + (current_robot_y - prev_robot_y) ** 2) ** 0.5
    
    return dist <= 10


if __name__ == "__main__":
    num_episodes = int(input("Enter number of episodes to run: "))
    all_episodes = []

    # init pygame and environment
    pygame.init()
    pygame.display.init()
    env = JamSpreadingEnv()

    # history of scalar uncertainties for plotting
    uncertainty_history = []

    action = None
    context = None

    # load trained init policy
    action_dim = 3
    state_dim = 36
    cont_policy = Continuous_Policy(state_dim=state_dim, output_dim=action_dim)
    cont_policy.load_state_dict(torch.load(POLICY_PATH))
    cont_policy.eval()

    # loop over deployment episodes
    for episode_num in range(num_episodes):
        
        # reset variables for each episode
        obs, _ = env.reset()
        obs_prev1 = obs
        obs_prev2 = obs
        done = False
        next_action_idx = 0

        # init for UI
        screen_state = "start" 
        
        # init for data logging
        episode = Episode(episode_num)
        step_id = 0

        # uncertain quant init
        q_lo = np.array([0.1, 0.1, 0.1])
        q_hi = np.array([0.1, 0.1, 0.1])

        # learning rate for quantile tracking
        stepsize = 0.2

        # desired coverage level α
        alpha_desired = 0.8

        list_of_uncertainties = []
        list_of_residuals = []
        history_upper_residuals = []
        history_lower_residuals = []
        B_t_lookback_window = 100

        uncertainty_history = []

        # initial render to ensure window exists
        env.render(episode_num, step_id, screen_state)
        env.help_remaining_time = 0.0

        # count number of consecutive similar state (increase the noise added)
        stuck_count = 0

        while not done:
            # start screen
            if screen_state == "start":
                if env.check_button_click():
                    screen_state = "robot"
                continue
            
            # -------------------------------
            # Get robot predicted action & uncertainty
            # -------------------------------

            # update stuck counter
            if similar_state(obs, obs_prev1):
                stuck_count += 1
            else:
                stuck_count = 0

            # get base action from policy on true obs
            base_action = get_prediction(obs, obs_prev1, obs_prev2, cont_policy)

            # add noise if stuck
            if stuck_count > 0:
                if stuck_count >= 13:
                    sigma = 15.0 
                elif stuck_count >= 10:
                    sigma = 10.0   # quite large noise
                elif stuck_count >= 7:
                    sigma = 5.0    # medium noise
                else:
                    sigma = 0.0

                if sigma > 0.0:
                    noise_xy = np.random.normal(loc=0.0, scale=sigma, size=2)
                    base_action[0] += noise_xy[0]
                    base_action[1] += noise_xy[1]

            # if(similar_state(obs, obs_prev1)):
            #     # noise_xy = np.random.normal(loc=0.0, scale=5.0, size=2)
            #     # base_action[0] += noise_xy[0]
            #     # base_action[1] += noise_xy[1]

            #     base_action[0] += random.uniform(-5, 5)
            #     base_action[1] += random.uniform(-10, 10)
            
            robot_prediction = base_action


            # if(similar_state(obs, obs_prev1)):
            #     obs_offset = obs.copy()
            #     obs_offset[0] += random.randint(-10, 10)
            #     obs_offset[1] += random.randint(-5, 5)
            #     robot_prediction = get_prediction(obs_offset, obs_prev1, obs_prev2, cont_policy)
            # else:
            #     robot_prediction = get_prediction(obs, obs_prev1, obs_prev2, cont_policy)

            # set predicted action to a valid action, if needed
            robot_prediction[0] = np.clip(robot_prediction[0], 0.0, 700.0)
            robot_prediction[1] = np.clip(robot_prediction[1], 0.0, 700.0)
            valid_gripper = np.array([0.0, 0.5, 1.0])
            robot_prediction[2] = valid_gripper[ # set gripper value to {0, 0.5, 1}
                np.argmin(np.abs(valid_gripper - robot_prediction[2]))
            ]

            print("robot_prediction:", ["{:.4f}".format(x) for x in robot_prediction])

            uncertainty_at_timestep = np.linalg.norm(q_hi + q_lo)
            print("uncertainty_at_timestep", uncertainty_at_timestep)
            
            # update history for plotting
            uncertainty_history.append(uncertainty_at_timestep)

            # keep only last N points so it does not grow forever
            max_points = 150
            if len(uncertainty_history) > max_points:
                uncertainty_history = uncertainty_history[-max_points:]

            # give env access to the history
            env.uncertainty_history = uncertainty_history

            # ---------------------------------------------------
            # "robot" state: robot is currently controlling the arm
            # Decide whether to keep going, ask for help, or accept human intervention
            # ---------------------------------------------------
            if screen_state == "robot":
                
                need_help = uncertainty_at_timestep > UNCERTAINTY_THRESHOLD

                ########### human intervenes ############
                if env.check_button_click():
                    screen_state = "ready_intervene"
                    context = "human_intervened"

                ########### robot queries ############
                elif need_help:
                    screen_state = "ready_help"
                    context = "robot_asked"

                ########### robot acts independently #######
                else:
                    action = robot_prediction
                    context = "robot_independent"
                    screen_state = "robot"

            # ---------------------------------------------------
            # "ready_intervene" and "ready_help" are pre help states
            # Wait until env.ready_to_help() reports that human started controlling
            # ---------------------------------------------------
            elif screen_state == "ready_intervene":
                if env.ready_to_help():
                    screen_state = "help"
                else:
                    env.render(episode_num, step_id, screen_state)
                    time.sleep(1 / 10)
                    continue

            elif screen_state == "ready_help":
                if env.ready_to_help():
                    screen_state = "help"
                else:
                    env.render(episode_num, step_id, screen_state)
                    time.sleep(1 / 10)
                    continue

            # ---------------------------------------------------
            # "help" state: human is providing expert action a_t^h
            # Update IQT and coverage
            # ---------------------------------------------------
            elif screen_state == "help":
                # handle keyboard events in help state
                exit_human_control = False
                if pygame.event.peek(pygame.KEYDOWN):
                    for event in pygame.event.get(pygame.KEYDOWN):
                        if event.key == pygame.K_SPACE:
                            exit_human_control = True

                # if human return control back to robot
                if exit_human_control:
                    screen_state = "robot" # return to robot control
                    continue

                # -------------------------------
                # Residuals and coverage check
                # -------------------------------

                # get expert action from human
                action = env.get_help()

                # pdb.set_trace()
                expert_y = action
                y_pred = robot_prediction

                shi_upper_residual = expert_y - y_pred
                slo_lower_residual = y_pred - expert_y
                list_of_residuals.append(np.linalg.norm(abs(expert_y - y_pred)))

                # # Check coverage, compute err_lo, err_hi
                # if shi_upper_residual > q_hi, then err_hi = 1
                # if slo_lower_residual > q_lo, then err_lo = 1
                err_hi = np.zeros(action_dim)
                err_lo = np.zeros(action_dim)
                for i in range(action_dim):
                    if shi_upper_residual[i] > q_hi[i]:
                        err_hi[i] = 1
                    if slo_lower_residual[i] > q_lo[i]:
                        err_lo[i] = 1
                print("err_hi", err_hi)
                print("err_lo", err_lo)
                covered = sum(err_hi) + sum(err_lo)
                if covered > 0:
                    covered = 0
                else:
                    covered = 1
                print("covered", covered)

                B_hi = np.ones(action_dim) * 0.01
                B_lo = np.ones(action_dim) * 0.01

                if len(history_upper_residuals) > 0:
                    B_hi = np.max(history_upper_residuals, axis=0)
                    B_lo = np.max(history_lower_residuals, axis=0)

                # Append the current residuals to the history
                history_upper_residuals.append(shi_upper_residual)
                history_lower_residuals.append(slo_lower_residual)

                # If the history is longer than the lookback window, pop the oldest residuals
                if len(history_upper_residuals) > B_t_lookback_window:
                    history_upper_residuals.pop(0)
                    history_lower_residuals.pop(0)

                # IQT Update - update quantile tracking parameters
                q_hi = q_hi + (stepsize) * B_hi * (err_hi - alpha_desired)
                q_lo = q_lo + (stepsize) * B_lo * (err_lo - alpha_desired)

            
            # ---------------------------------------
            # Step the environment if we have an action
            # ---------------------------------------
            obs_prev2 = obs_prev1
            obs_prev1 = obs

            if action is not None:
                obs, _, done, _, _ = env.step(action)
                img_path = f"jam_state_img/test/episode{episode_num}_step{step_id}.png"
                timestep = TimeStep(
                    step_id=step_id,
                    action=action,
                    state=obs,
                    state_img_path=img_path,
                    context=context,
                    robot_prediction=robot_prediction
                )
                episode.add_step(timestep)
                if(done): screen_state = "done"

                # render current UI state and slow simulation to 10 FPS
                env.render(episode_num, step_id, screen_state)
                step_id += 1
                time.sleep(1 / 10)

        
        print(f"Episode {episode_num} ended.")
        controlled_delay(3000) # Small pause between episodes (3 seconds)

    env.close()  # Close the Gym environment
    del env  # Explicitly delete the environment

    if pygame.get_init():
        pygame.quit()

    print("Environment closed successfully.")