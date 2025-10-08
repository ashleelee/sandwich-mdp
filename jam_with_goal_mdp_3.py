import os
import time
import random
import pickle
from typing import Optional, Tuple, List

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import torch

from jam_data_classes import TimeStep, Episode
from policies.conformal.mlp import Continuous_Policy

# =========================
# Global constants & config
# =========================

FPS = 30
VIEWPORT_W = 1000
VIEWPORT_H = 800

# Control Toggle 
HUMAN_ALWAYS_CONTROL = True

# Side panel geometry
SIDE_PANEL_X = 700
SIDE_PANEL_W = 200
SIDE_PANEL_H = 700

# ==== Multi-spread UI/logic ====
SPREAD_TYPES = ["jam", "peanut", "nutella", "avocado"]
SPREAD_COLOR = {
    "jam":      (220, 80, 100),
    "peanut":   (227, 152, 60),
    "nutella":  (110, 80, 40),
    "avocado":  (80, 170, 90),
}

# Top tray bar and bag placement
TOP_TRAY_H = 120
TOP_TRAY_RECT = pygame.Rect(0, 0, VIEWPORT_W, TOP_TRAY_H)

# Where the four bags sit on the top tray (tweak x's as you like)
BAG_ANCHORS = {
    "jam":      (180,  65),
    "peanut":   (300,  65),
    "nutella":  (420,  65),
    "avocado":  (540,  65),
}

PICKUP_RADIUS = 40  # same feel as your single-bag pickup
# Tray docking behavior
DOCK_EPSILON = 20   # how close to anchor counts as "on tray"


# ======================
# Helper / utility funcs
# ======================

def controlled_delay(delay_ms: int) -> None:
    """Non-blocking delay while keeping pygame responsive."""
    start_time = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start_time < delay_ms:
        pygame.event.pump()


def get_prediction(
    obs: np.ndarray,
    obs_prev1: np.ndarray,
    obs_prev2: np.ndarray,
    policy_model: Continuous_Policy,
) -> np.ndarray:
    """Compute policy action with your original normalization."""
    min_X = np.array([
        np.float32(90.0), np.float32(78.0), np.float32(0.0), np.float32(0.0),
        np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0),
        np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0),
        np.float32(92.0), np.float32(74.0), np.float32(0.0), np.float32(0.0),
        np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0),
        np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0),
        np.float32(92.0), np.float32(74.0), np.float32(0.0), np.float32(0.0),
        np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0),
        np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0)
    ])
    max_X = np.array([
        np.float32(573.0), np.float32(524.0), np.float32(1.0), np.float32(1.0),
        np.float32(0.25068787), np.float32(0.075533986), np.float32(0.25659528),
        np.float32(0.6491279), np.float32(0.52741855), np.float32(0.40159684),
        np.float32(0.40303788), np.float32(0.4491095), np.float32(573.0),
        np.float32(524.0), np.float32(1.0), np.float32(1.0), np.float32(0.25068787),
        np.float32(0.075533986), np.float32(0.25659528), np.float32(0.6491279),
        np.float32(0.52741855), np.float32(0.40159684), np.float32(0.40303788),
        np.float32(0.4491095), np.float32(573.0), np.float32(524.0),
        np.float32(1.0), np.float32(1.0), np.float32(0.25068787),
        np.float32(0.075533986), np.float32(0.25659528), np.float32(0.6491279),
        np.float32(0.52741855), np.float32(0.40159684), np.float32(0.40303788),
        np.float32(0.4491095)
    ])
    min_Y = np.array([np.float32(136.0), np.float32(74.0), np.float32(0.0)])
    max_Y = np.array([np.float32(573.0), np.float32(524.0), np.float32(1.0)])

    input_obs = np.concatenate((obs_prev2[6:], obs_prev1[6:], obs[6:]), axis=0)
    input_obs = (input_obs - min_X) / (max_X - min_X)

    with torch.no_grad():
        state_tensor = torch.tensor(np.array(input_obs)).float().unsqueeze(0)
        action_pred = policy_model(state_tensor)
        action_pred = action_pred.detach().numpy() * (max_Y - min_Y) + min_Y

    return action_pred[0]


# ==========================
# Environment implementation
# ==========================

class JamSpreadingEnv(gym.Env):
    """
    Pygame-based environment for jam spreading.
    Action: [x, y, gripper] (continuous)
    Observation: 18-D vector.
    """
    JAM_WIDTH = 40

    def __init__(self) -> None:
        super().__init__()

        # Pygame init
        if not pygame.get_init():
            pygame.init()
            pygame.display.init()
        self.screen: pygame.Surface = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        self.clock: pygame.time.Clock = pygame.time.Clock()

        # Drawing & state holders
        self.save_interval = 0.0
        self.last_save_time = time.time()
        self.save_counter = 0


        self.jam_lines: List[Tuple[int, int]] = []
        # Multi-bag state
        self.spread_lines = {name: [] for name in SPREAD_TYPES}  # per-spread jam traces
        self.active_spread = "jam"                                # which bag user is interacting with
        self.held_spread = None                                   # None or one of SPREAD_TYPES
        self.bag_current_pos = dict(BAG_ANCHORS)                  # each bag's current location (follows robot when held)



        # State vector layout:
        # [0:2]=start, [2:4]=bag_init, [4:6]=bread_end,
        # [6:8]=robot_xy, [8]=gripper, [9]=holding, [10:18]=coverage(8)
        self.state = np.array([
            90.0, 90.0,
            610.0, 90.0,
            200.0, 580.0,
            90.0, 90.0,
            0.0,
            0.0,
            *([0.0] * 8)
        ], dtype=np.float32)

        # Spaces
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([700.0, 700.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array([0.0] * 18, dtype=np.float32),
            high=np.array([
                700.0, 700.0,
                700.0, 700.0,
                700.0, 700.0,
                700.0, 700.0,
                1.0,
                1.0,
                *([1.0] * 8)
            ], dtype=np.float32),
            dtype=np.float32
        )

        # Flags
        self.done = False
        self.hit_bread_endpoints = False
        self.last_gripper_state = 0.5

        # Bread grid geometry
        self.box_positions = [
            (160, 155), (350, 155),
            (160, 275), (350, 275),
            (160, 395), (350, 395),
            (160, 515), (350, 515)
        ]
        self.box_size = (190, 120)
        self.box_area = self.box_size[0] * self.box_size[1]

        # Assets
        self._load_images()

        

    # ---------- assets ----------

    def _load_images(self) -> None:
        def _load(path: str, size: Optional[Tuple[int, int]] = None) -> pygame.Surface:
            surf = pygame.image.load(path).convert_alpha()
            return pygame.transform.scale(surf, size) if size else surf

        self.robot_images = {
            "open":   _load("img_c/open.png", (175, 175)),
            "hold":   _load("img_c/hold.png", (175, 175)),
            "clutch": _load("img_c/clutch.png", (175, 175)),
        }

        # Per-spread bag art:
        # jam uses original filenames; peanut/nutella/avocado use suffixes _b/_c/_d
        self.piping_bag_images = {
            "jam": {
                "normal":   _load("img_c/normal.png",   (175, 175)),
                "squeezed": _load("img_c/squeezed.png", (175, 175)),
            },
            "peanut": {
                "normal":   _load("img_c/normal_b.png",   (175, 175)),
                "squeezed": _load("img_c/squeezed_b.png", (175, 175)),
            },
            "nutella": {
                "normal":   _load("img_c/normal_c.png",   (175, 175)),
                "squeezed": _load("img_c/squeezed_c.png", (175, 175)),
            },
            "avocado": {
                "normal":   _load("img_c/normal_d.png",   (175, 175)),
                "squeezed": _load("img_c/squeezed_d.png", (175, 175)),
            },
        }

        self.bread_img = _load("img_c/bread.png", (550, 550))
        self.bowl_img  = _load("img_c/bowl.png",  (100, 100))


    # ---------- gym API ----------

    def reset(self, seed: Optional[int] = None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.state[:] = np.array([
            90.0, 90.0,
            610.0, 90.0,
            200.0, 580.0,
            90.0, 90.0,
            0.0,
            0.0,
            *([0.0] * 8)
        ], dtype=np.float32)

        self.jam_lines.clear()
        self.spread_lines = {name: [] for name in SPREAD_TYPES}
        self.active_spread = "jam"
        self.held_spread = None
        self.bag_current_pos = dict(BAG_ANCHORS)
                               

        self.done = False
        self.hit_bread_endpoints = False
        self.last_gripper_state = 0.5
        return self.state, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self._update_state(action)
        self._check_hit_bread_endpoints()
        done = self._is_done()
        return self.state, 0.0, done, False, {}

    # ---------- internals ----------

    def _update_state(self, action: np.ndarray) -> None:
        # Desired new robot pose/gripper from action
        desired_x, desired_y, desired_g = float(action[0]), float(action[1]), float(action[2])

        # Enforce: outside tray, you cannot switch/open (0.0). Force to hold (0.5).
        if not self._over_tray(desired_x, desired_y) and desired_g == 0.0:
            desired_g = 0.5

        # Apply to state
        self.state[6], self.state[7], self.state[8] = desired_x, desired_y, desired_g
        robot_x, robot_y = self.state[6], self.state[7]
        gripper_state = self.state[8]

        # Keep state[2:4] in sync for backwards compatibility (points at the active bag on the tray)
        ax, ay = self.bag_current_pos[self.active_spread]
        self.state[2], self.state[3] = ax, ay

        # If not holding anything yet, allow pickup by proximity:
        # - If the bag is on its tray, pickup radius is tested at the TRAY anchor.
        # - If the bag is off-tray (you dropped it), pickup is tested at the BAG'S CURRENT location.
        if self.held_spread is None and gripper_state == 0.5:
            tip_x, tip_y = robot_x, robot_y
            for name in SPREAD_TYPES:
                if self._is_on_tray(name):
                    ax, ay = BAG_ANCHORS[name]
                    dist = np.hypot(tip_x - ax, tip_y - ay)
                else:
                    bx, by = self.bag_current_pos[name]
                    dist = np.hypot(tip_x - bx, tip_y - by)

                if dist <= PICKUP_RADIUS:
                    self.held_spread = name
                    self.active_spread = name
                    self.state[9] = 1.0
                    break

        # If holding a bag, make its position follow the robot and (optionally) deposit jam
        if self.held_spread is not None:
            # bag follows robot hand
            self.bag_current_pos[self.held_spread] = (robot_x, robot_y)

            if self.state[8] == 1.0:  # clutching
                jam_point = (int(robot_x + 35), int(robot_y + 70))
                if not self.hit_bread_endpoints:
                    # legacy combined trace
                    if (not self.jam_lines) or (jam_point != self.jam_lines[-1]):
                        self.jam_lines.append(jam_point)
                        self._update_jam_coverage_area()
                    # per-spread trace
                    lines = self.spread_lines[self.held_spread]
                    if (not lines) or (jam_point != lines[-1]):
                        lines.append(jam_point)

        # --- Release logic: only drop if the TIP is inside the tray ---
        if self.held_spread is not None and gripper_state == 0.0:
            name = self.held_spread

            # tip of the piping bag based on robot pose
            tip_x = float(robot_x + 35)
            tip_y = float(robot_y + 70)

            if TOP_TRAY_RECT.collidepoint(int(tip_x), int(tip_y)):
                # Allowed to put down: snap back to that bag's tray anchor
                ax, ay = BAG_ANCHORS[name]
                self.bag_current_pos[name] = (ax, ay)
                self.state[9] = 0.0       # no longer holding
                self.held_spread = None
            else:
                # Not allowed to put down outside the tray: keep holding.
                # Force gripper back to hold so the state remains consistent.
                self.state[8] = 0.5       # hold
                self.state[9] = 1.0       # holding
                # self.held_spread remains unchanged




    def _update_jam_coverage_area(self) -> None:
        if len(self.jam_lines) < 2:
            return
        w = self.JAM_WIDTH
        box_areas_covered = [0.0] * 8
        for (x1, y1), (x2, y2) in zip(self.jam_lines[:-1], self.jam_lines[1:]):
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            seg_area = max(dx, dy) * w
            min_x = min(x1, x2) - w // 2
            min_y = min(y1, y2) - w // 2
            max_x = max(x1, x2) + w // 2
            max_y = max(y1, y2) + w // 2
            seg_rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)
            for j, (bx, by) in enumerate(self.box_positions):
                box_rect = pygame.Rect(bx, by, *self.box_size)
                inter = seg_rect.clip(box_rect)
                if inter.width > 0 and inter.height > 0:
                    overlap_area = inter.width * inter.height
                    box_areas_covered[j] += (overlap_area / (seg_rect.width * seg_rect.height)) * seg_area
        for i in range(8):
            self.state[10 + i] = min(box_areas_covered[i] / self.box_area, 1.0)

    def _is_done(self) -> bool:
        rx, ry = self.state[6], self.state[7]
        tip_x, tip_y = rx, ry
        end_x, end_y = 610, 140
        if np.hypot(tip_x - end_x, tip_y - end_y) <= 25 and self.hit_bread_endpoints and self.state[8] == 0.0:
            self.done = True
            print("done")
            return True
        return False

    def _check_hit_bread_endpoints(self) -> None:
        rx, ry = self.state[6], self.state[7]
        bx, by = self.state[4], self.state[5]
        tip_x, tip_y = rx + 35, ry + 70
        if np.hypot(tip_x - bx, tip_y - by) <= 25:
            self.hit_bread_endpoints = True

    # ---------- human input helpers (unchanged) ----------

    def ready_to_help(self) -> bool:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        rx, ry = self.state[6], self.state[7]
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                return np.hypot(mouse_x - rx, mouse_y - ry) <= 30.0
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
        return False

    def check_intervene_click(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                button_w, button_h = 120, 40
                bx = SIDE_PANEL_X + (SIDE_PANEL_W - button_w) // 2
                by = (SIDE_PANEL_H - button_h) // 2
                return pygame.Rect(bx, by, button_w, button_h).collidepoint(mx, my)
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
        return False
    
    def _over_tray(self, x: float, y: float) -> bool:
        return TOP_TRAY_RECT.collidepoint(int(x), int(y))


    def get_help(self) -> np.ndarray:
        # Cursor-controlled robot position
        x, y = pygame.mouse.get_pos()
        current = float(self.state[8])
        gripper = current

        over_tray = self._over_tray(self.state[6], self.state[7])

        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if over_tray:
                    # Full cycle 0.0 -> 0.5 -> 1.0 -> 0.0
                    cycle = [0.0, 0.5, 1.0]
                    try:
                        i = cycle.index(round(current, 1))
                    except ValueError:
                        i = 0
                    gripper = cycle[(i + 1) % len(cycle)]
                else:
                    # Outside the tray: only toggle between hold (0.5) and clutch (1.0).
                    if current <= 0.5:
                        # if open (0.0) or hold (0.5) -> go to hold (0.5) first, then clutch on next click
                        gripper = 0.5 if current == 0.0 else 1.0
                    else:
                        # clutch -> hold
                        gripper = 0.5
                self.last_gripper_state = current

            elif event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        return np.array([x, y, gripper], dtype=np.float32)


    def _is_on_tray(self, name: str) -> bool:
        bx, by = self.bag_current_pos[name]
        ax, ay = BAG_ANCHORS[name]
        return np.hypot(bx - ax, by - ay) <= DOCK_EPSILON


    # ---------- drawing ----------

    def draw_intervene_button(self) -> None:
        button_w, button_h = 120, 40
        bx = SIDE_PANEL_X + (SIDE_PANEL_W - button_w) // 2
        by = (SIDE_PANEL_H - button_h) // 2
        rect = pygame.Rect(bx, by, button_w, button_h)
        pygame.draw.rect(self.screen, (255, 224, 161), rect)
        font = pygame.font.SysFont("Arial", 20, bold=True)
        self.screen.blit(font.render("Intervene", True, (203, 91, 59)), font.render("Intervene", True, (203, 91, 59)).get_rect(center=rect.center))

    def draw_intervene_ready_text(self) -> None:
        font = pygame.font.SysFont("Arial", 14, bold=True)
        color = (139, 69, 19)
        lines = ["Please click on the dot on", "the robot arm and guide", "me with your cursor."]
        x = SIDE_PANEL_X + 15
        y0 = 300
        for i, line in enumerate(lines):
            surf = font.render(line, True, color)
            self.screen.blit(surf, (x, y0 + i * 24))

    def _draw_robot(self) -> None:
        g = self.state[8]
        key = "open" if g == 0.0 else ("hold" if g == 0.5 else "clutch")
        img = self.robot_images[key]
        pos = (int(self.state[6]), int(self.state[7]))
        self.screen.blit(img, img.get_rect(center=pos))

    def _draw_piping_bag(self) -> None:
        for name in SPREAD_TYPES:
            color = SPREAD_COLOR[name]
            imgs = self.piping_bag_images[name]
            squeezed = (name == self.held_spread and self.state[8] == 1.0)
            img = imgs["squeezed" if squeezed else "normal"]

            if name == self.held_spread:
                # Draw at robot
                bag_pos = (int(self.state[6]), int(self.state[7]))
                self.screen.blit(img, img.get_rect(center=bag_pos))
            else:
                # Draw at its current world position (on tray or wherever it was dropped)
                bx, by = self.bag_current_pos[name]
                self.screen.blit(img, (int(bx) - 125, int(by) - 87))

            # # Optional: frame for non-held bags
            # if name != self.held_spread:
            #     bx, by = self.bag_current_pos[name]
            #     frame = pygame.Rect(int(bx) - 35, int(by) + 25, 70, 70)
            #     pygame.draw.rect(self.screen, color, frame, 2)


    def _draw_trays(self) -> None:
        """Draw the top tray bar and four static docking rectangles at anchors."""
        # Top tray bar background (optional – make it obvious)
        pygame.draw.rect(self.screen, (235, 235, 235), TOP_TRAY_RECT)

        # Static docking rectangles at anchors (never move)
        for name, (ax, ay) in BAG_ANCHORS.items():
            color = SPREAD_COLOR[name]
            # Use same geometry you used before for the frame under the bag image
            dock_rect = pygame.Rect(int(ax) - 35, int(ay) + 25, 70, 70)
            pygame.draw.rect(self.screen, color, dock_rect, 2)



    def _draw_jam(self) -> None:
        # draw each spread in its own color
        for name, lines in self.spread_lines.items():
            if len(lines) > 1:
                pygame.draw.lines(self.screen, SPREAD_COLOR[name], False, lines, self.JAM_WIDTH)

        # (Optional) keep legacy red overlay as well:
        # if len(self.jam_lines) > 1:
        #     pygame.draw.lines(self.screen, (255, 0, 0), False, self.jam_lines, self.jam_width)


    def draw_episode_num(self, episode_num: int) -> None:
        font = pygame.font.SysFont("Arial", 16, bold=True)
        ep = font.render(f"Episode: {episode_num}", True, (0, 0, 0))
        self.screen.blit(ep, ep.get_rect(topright=(90, 25)))

    def render(self, episode_num: int, step_id: int, screen: str = "help") -> None:
        self.screen.fill((255, 255, 255))

        side_rect = pygame.Rect(SIDE_PANEL_X, 0, SIDE_PANEL_W, SIDE_PANEL_H)
        pygame.draw.rect(self.screen, (247, 243, 238), side_rect)
        if screen != "ready":
            self.draw_intervene_button()
        else:
            self.draw_intervene_ready_text()

        bread_pos = ((700 - self.bread_img.get_width()) // 2, (VIEWPORT_H - self.bread_img.get_height()) - 30)
        self.screen.blit(self.bread_img, bread_pos)

        self._draw_jam()

        # periodic capture (left 700x700 region)
        if (time.time() - self.last_save_time) >= self.save_interval:
            filename = f"jam_state_img/test/episode{episode_num}_step{step_id}.png"
            subsurf = self.screen.subsurface((0, 0, 700, 700))
            pygame.image.save(subsurf, filename)
            self.save_counter += 1
            self.last_save_time = time.time()

        self._draw_trays()

        # self.screen.blit(self.bowl_img, (560, 75))
        self._draw_piping_bag()
        self._draw_robot()

        bread_x, bread_y = int(self.state[4]), int(self.state[5])
        # pygame.draw.circle(self.screen, (255, 165, 0), (bread_x, bread_y), 25)
        # pygame.draw.circle(self.screen, (255, 165, 0), (610, 140), 25)

        font = pygame.font.SysFont("Arial", 16, bold=True)
        status = f"Spread: {self.active_spread}" + (f" (holding)" if self.held_spread else "")
        self.screen.blit(font.render(status, True, (0, 0, 0)), (10, 30))
        robot_x, robot_y = self.state[6], self.state[7]
        tip_x, tip_y = robot_x, robot_y  # same offset as in update_state

        # Draw the tip (small orange dot)
        pygame.draw.circle(self.screen, (255, 140, 0), (tip_x, tip_y), 6)

        # Draw pickup radius (transparent circle outline)
        pygame.draw.circle(self.screen, (0, 100, 255), (tip_x, tip_y), PICKUP_RADIUS, 2)

       
        # Pickup outlines at trays; if holding, draw circle at tip for the held bag
        for name, (ax, ay) in BAG_ANCHORS.items():
            color = SPREAD_COLOR[name]
            if name == self.held_spread:
                tip_x, tip_y = int(self.state[6]), int(self.state[7])
                pygame.draw.circle(self.screen, color, (tip_x, tip_y), int(PICKUP_RADIUS), 2)
            else:
                pygame.draw.circle(self.screen, color, (int(ax), int(ay)), int(PICKUP_RADIUS), 2)


        font = pygame.font.SysFont("Arial", 16, bold=True)
        self.screen.blit(font.render(f"Mode: {screen}", True, (0, 0, 0)), (10, 10))
        self.draw_episode_num(episode_num)

        self.clock.tick(FPS)
        pygame.display.flip()


# ================
# Main entry point
# ================

if __name__ == "__main__":
    # Load demo episodes (unchanged usage)
    with open("jam_all_episodes_gen.pkl", "rb") as f:
        episodes = pickle.load(f)
    ep0 = episodes[0]
    jam_sample_actions = [step.action.tolist() for step in ep0.steps]

    num_episodes = int(input("Enter number of episodes to run: "))

    pygame.init()
    pygame.display.init()

    env = JamSpreadingEnv()
    action: Optional[np.ndarray] = None
    context: Optional[str] = None

    # Load policy only when not human-only (safe CPU map; no behavior change)
    action_dim = 3
    state_dim = 36
    cont_policy: Optional[Continuous_Policy] = None
    if not HUMAN_ALWAYS_CONTROL:
        cont_policy = Continuous_Policy(state_dim=state_dim, output_dim=action_dim)
        try:
            state_dict = torch.load("cont_policy.pth", map_location="cpu", weights_only=True)  # newer torch
        except TypeError:
            state_dict = torch.load("cont_policy.pth", map_location="cpu")  # older torch
        cont_policy.load_state_dict(state_dict)
        cont_policy.eval()

    for episode_num in range(num_episodes):
        obs, _ = env.reset()
        obs_prev1 = obs.copy()
        obs_prev2 = obs.copy()

        done = False
        next_action_idx = 0
        next_action = jam_sample_actions[next_action_idx]
        help_start_time: Optional[float] = None
        screen_state = "robot"
        last_help_check_time = time.time()
        episode = Episode(episode_num)
        step_id = 0

        # Conformal tracking vars (unchanged)
        q_lo = np.array([0.1, 0.1, 0.1])
        q_hi = np.array([0.1, 0.1, 0.1])
        stepsize = 0.2
        alpha_desired = 0.8
        list_of_uncertainties: List[float] = []
        list_of_residuals: List[float] = []
        history_upper_residuals: List[np.ndarray] = []
        history_lower_residuals: List[np.ndarray] = []
        B_t_lookback_window = 100

        while not done:
            if HUMAN_ALWAYS_CONTROL:
                screen_state = "help"
                context = "human_intervened"
                action = env.get_help()
                robot_prediction = None

            else:
                robot_prediction = get_prediction(obs, obs_prev1, obs_prev2, cont_policy)
                print("robot_prediction", robot_prediction)

                if screen_state == "robot":
                    need_help = False
                    if time.time() - last_help_check_time >= 2.0:
                        uncertainty_at_timestep = np.linalg.norm(q_hi + q_lo)
                        print("uncertainty_at_timestep", uncertainty_at_timestep)
                        list_of_uncertainties.append(float(uncertainty_at_timestep))
                        need_help = uncertainty_at_timestep > 10

                    if env.check_intervene_click():
                        screen_state = "ready"
                        context = "human_intervened"
                    elif need_help:
                        screen_state = "ready"
                        context = "robot_asked"
                    else:
                        action = robot_prediction
                        context = "robot_independent"
                        screen_state = "robot"

                elif screen_state == "ready":
                    if env.ready_to_help():
                        screen_state = "help"
                        help_start_time = time.time()
                    else:
                        env.render(episode_num, step_id, screen_state)
                        time.sleep(1 / 10)
                        continue

                elif screen_state == "help":
                    action = env.get_help()

                    if help_start_time and time.time() - help_start_time >= 3.0:
                        # Snap back to nearest script point (unchanged)
                        rx, ry = env.state[6], env.state[7]
                        min_dist = float("inf")
                        best_idx = next_action_idx
                        for i in range(len(jam_sample_actions)):
                            ax, ay = jam_sample_actions[i][0], jam_sample_actions[i][1]
                            d = ((ax - rx) ** 2 + (ay - ry) ** 2) ** 0.5
                            if d < min_dist:
                                min_dist = d
                                best_idx = i
                        next_action_idx = best_idx
                        next_action = jam_sample_actions[next_action_idx]
                        last_help_check_time = time.time()
                        screen_state = "robot"

                    # Conformal stats (unchanged)
                    expert_y = action
                    y_pred = robot_prediction
                    shi_upper_residual = expert_y - y_pred
                    slo_lower_residual = y_pred - expert_y
                    list_of_residuals.append(float(np.linalg.norm(np.abs(expert_y - y_pred))))

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
                    covered = 0 if covered > 0 else 1
                    print("covered", covered)

                    B_hi = np.ones(action_dim) * 0.01
                    B_lo = np.ones(action_dim) * 0.01
                    if len(history_upper_residuals) > 0:
                        B_hi = np.max(history_upper_residuals, axis=0)
                        B_lo = np.max(history_lower_residuals, axis=0)

                    history_upper_residuals.append(shi_upper_residual)
                    history_lower_residuals.append(slo_lower_residual)
                    if len(history_upper_residuals) > B_t_lookback_window:
                        history_upper_residuals.pop(0)
                        history_lower_residuals.pop(0)

                    q_hi = q_hi + (stepsize) * B_hi * (err_hi - alpha_desired)
                    q_lo = q_lo + (stepsize) * B_lo * (err_lo - alpha_desired)

            # Book-keeping
            obs_prev2 = obs_prev1
            obs_prev1 = obs

            if action is not None:
                # Defensive guard that does not change your logic:
                # skip step if NaN/Inf; otherwise clamp into action box
                if not np.all(np.isfinite(action)):
                    print("Skipping step due to invalid action:", action)
                else:
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                    obs, _, done, _, _ = env.step(action)

                    img_path = f"jam_state_img/test/episode{episode_num}_step{step_id}.png"
                    timestep = TimeStep(
                        step_id=step_id,
                        action=action,
                        state_v=obs,
                        state_img_path=img_path,
                        context=context,
                        robot_prediction=robot_prediction,
                    )
                    episode.add_step(timestep)

            env.render(episode_num, step_id, screen_state)
            step_id += 1
            time.sleep(1 / 10)

        print(f"Episode {episode_num} ended.")
        controlled_delay(3000)

    env.close()
    del env
    if pygame.get_init():
        pygame.quit()
    print("Environment closed successfully.")
