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
import json

# init variables for pygame
FPS = 30
VIEWPORT_W = 900
VIEWPORT_H = 700
JAM_WIDTH = 40
DISTANCE_TO_HOLD_BAG = 20
DISTANCE_TO_HOLD_ROBOT_ARM = 40
UNCERTAINTY_THRESHOLD = 20
shape = "zigzag"


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
                    print("not on bread!")
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

        # draw threshold line (red)
        threshold_y = origin_y + graph_height - (threshold / max_u) * graph_height
        pygame.draw.line(surface, (255, 0, 0),
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
        # if screen_state == "start":
        #     mode_text = "Robot Operating"
        # elif screen_state == "robot":
        #     mode_text = "Robot Operating"
        # elif screen_state == "ready_help":
        #     mode_text = "Robot Requesting Help"
        # elif screen_state == "ready_intervene":
        #     mode_text = "Human Taking Over"
        # elif screen_state == "help":
        #     mode_text = "Human Assisting"
        # elif screen_state == "done":
        #     mode_text = "Complete"
        # else:
        #     mode_text = screen_state

        self.draw_side_panel(episode_num, "human assisting")

        # draw the button for this screen_state
        # if screen_state == "start":
        #     self.draw_button("START")
        # elif screen_state == "robot":
        #     self.draw_button("INTERVENE")
        # elif screen_state == "ready_help":
        #     self.draw_takeover_text()
        # elif screen_state == "ready_intervene":
        #     self.draw_intervene_ready_text()
        # elif screen_state == "help":
        #     self.draw_human_control()
        # elif screen_state == "done":
        #     self.draw_episode_complete_text()
        
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
            filename = f"jam_state_img/train_{shape}/episode{episode_num}_step{step_id}.png"
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

            img_path = f"jam_state_img/train_{shape}/episode{global_ep_id}_step{step_id}.png"

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
            time.sleep(1/10) # 10 timesteps per second

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

def rollout_from_file():
    data_path = "data/jam_train_data_all_triangle_filtered.pkl"

    with open(data_path, "rb") as f:
        all_episodes = pickle.load(f)   # list of dicts

    print(f"Loaded {len(all_episodes)} episodes from {data_path}")

    # choose which episode to replay
    ep_idx = int(input(f"Enter episode index to replay [0 .. {len(all_episodes)-1}]: "))
    ep_dict = all_episodes[ep_idx]
    steps = ep_dict["steps"]

    pygame.init()
    pygame.display.init()
    env = JamSpreadingEnv()

    # reset env before replay
    obs, _ = env.reset()
    done = False
    step_id = 0

    print(f"Ready to replay episode {ep_idx} with {len(steps)} steps")
    env.render(ep_idx, step_id, screen="help")
    wait_for_space()

    for step_id, step in enumerate(steps):
        if done:
            break

        # stored as list; convert to numpy in case env expects that
        action = np.array(step["action"], dtype=np.float32)

        obs, _, done, _, _ = env.step(action)

        # you can choose any screen string that your render uses
        env.render(ep_idx, step_id, screen="help")

        time.sleep(1/10)   # same rate as data collection

    print(f"Replay of episode {ep_idx} finished.")

    env.close()
    if pygame.get_init():
        pygame.quit()

    print("Environment closed successfully.")


if __name__ == "__main__":
    # rollout_from_file()
    collect_training_data()
