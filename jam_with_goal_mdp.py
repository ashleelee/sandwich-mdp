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
HUMAN_ALWAYS_CONTROL = False   # run the robot policy
MODEL_PATH = "cont_policy_with_goal_2.pth"
STATS_PATH = "minmax_stats_with_goal.npz"

# Side panel geometry
SIDE_PANEL_X = 800
SIDE_PANEL_Y = 150
SIDE_PANEL_W = 200
SIDE_PANEL_H = VIEWPORT_H - SIDE_PANEL_Y

# Top tray bar and bag placement
TOP_TRAY_H = 150
TOP_TRAY_RECT = pygame.Rect(0, 0, VIEWPORT_W, TOP_TRAY_H)
TOP_TRAY_COLOR = (245, 215, 181)

# Spread
SPREAD_TYPES = ["jam", "peanut", "nutella", "avocado"]
SPREAD_COLOR = {
    "jam":      (220, 80, 100),
    "peanut":   (227, 152, 60),
    "nutella":  (110, 80, 40),
    "avocado":  (80, 170, 90),
}

# Where the four bags sit on the top tray 
BAG_Y = 140
BAG_ANCHORS = {
    "jam":      (420,  BAG_Y),
    "peanut":   (520,  BAG_Y),
    "nutella":  (620,  BAG_Y),
    "avocado":  (720,  BAG_Y),
}

PICKUP_RADIUS = 40  

# ===== Sidebar info widgets =====
INFO_BOX_OUTLINE = (238, 197, 144)
INFO_BOX_FILL    = (255, 244, 226)
INFO_TEXT_COLOR  = (102, 73, 45)

# =========================
# state layout (17-D)
# [ rx, ry, g, holding, coverage[8], spread_comp[4], bread_idx ]
# =========================
IDX_RX = 0
IDX_RY = 1
IDX_G  = 2
IDX_H  = 3
IDX_C0 = 4                 # coverage[0]..coverage[7]
IDX_SC0 = 12               # spread_comp[0]..spread_comp[3]
IDX_BI = 16                # bread index (float)

STATE_DIM = 17

# ======================
# Helper / utility funcs
# ======================

def controlled_delay(delay_ms: int) -> None:
    """Non-blocking delay while keeping pygame responsive."""
    start_time = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start_time < delay_ms:
        pygame.event.pump()


def get_prediction_prev(
    obs: np.ndarray,
    obs_prev1: np.ndarray,
    obs_prev2: np.ndarray,
    policy_model: Continuous_Policy,
    x_min: np.ndarray,
    x_max: np.ndarray,
    y_min: np.ndarray,
    y_max: np.ndarray,
) -> np.ndarray:
    """
    Build the 51-D input [s_t, s_{t+1}, s_{t+2}], apply the SAME min–max
    normalization used during training, run the policy, and de-normalize
    the predicted action back to env units (pixels, gripper in [0,1]).
    """
    x = np.concatenate([obs, obs_prev1, obs_prev2]).astype(np.float32)  # shape (51,)
    x_n = (x - x_min) / (x_max - x_min + 1e-8)

    with torch.no_grad():
        xt = torch.from_numpy(x_n).unsqueeze(0).float()  # (1, 51)
        y_n = policy_model(xt).cpu().numpy()[0]          # normalized action
    y = y_n * (y_max - y_min + 1e-8) + y_min
    print(y.astype(np.float32))            # de-normalized action
    return y.astype(np.float32)                          # expected: [x,y,g]

def get_prediction(
    obs: np.ndarray,
    obs_prev1: np.ndarray,
    obs_prev2: np.ndarray,
    policy_model: Continuous_Policy,
    x_min: np.ndarray,
    x_max: np.ndarray,
    y_min: np.ndarray,
    y_max: np.ndarray,
) -> np.ndarray:
    # x = np.concatenate([obs, obs_prev1, obs_prev2]).astype(np.float32)
    x = np.concatenate([obs_prev2, obs_prev1, obs]).astype(np.float32)
    x_n = (x - x_min) / x_range           # <= use safe range

    with torch.no_grad():
        xt = torch.from_numpy(x_n).unsqueeze(0).float()
        action_pred = policy_model(xt).cpu().numpy()[0]

    action_pred = np.clip(action_pred, 0.0, 1.0)          # normalized outputs were trained in [0,1]
    action_pred = action_pred * y_range + y_min             # <= use safe range

    return action_pred.astype(np.float32)



# ==========================
# Environment implementation
# ==========================

class JamSpreadingEnv(gym.Env):
    """
    Pygame-based environment for jam spreading.
    Action: [x, y, gripper] (continuous)
    Observation (12-D): [rx, ry, g, holding, coverage[8]]
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
        self.save_interval = 5.0
        self.last_save_time = time.time()
        self.save_counter = 0

        # Legacy / per-spread lines
        self.jam_lines: List[Tuple[int, int]] = []
        self.spread_lines = {name: [] for name in SPREAD_TYPES}  # per-spread jam traces
        self.active_spread = "jam"                                # which bag user is interacting with
        self.held_spread = None                                   # None or one of SPREAD_TYPES
        self.bag_current_pos = dict(BAG_ANCHORS)                  # each bag's current location (follows robot when held)

        # 17-D state
        self.state = np.array([
            90.0, 90.0,           # rx, ry
            0.0,                  # gripper
            0.0,                  # holding flag
            *([0.0] * 8),         # coverage
            *([0.0] * 4),         # spread composition [jam, peanut, nutella, avocado]
            0.0                   # bread index
        ], dtype=np.float32)

        # Spaces
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([VIEWPORT_W, VIEWPORT_H, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0] + [0.0]*8 + [0.0]*4 + [0.0], dtype=np.float32),
            high=np.array([VIEWPORT_W, VIEWPORT_H, 1.0, 1.0] + [1.0]*8 + [1.0]*4 + [10.0], dtype=np.float32),
            dtype=np.float32
        )

        # Goal
        self.goal = [1, 0, 0, 0]

        # Flags
        self.done = False
        self.last_gripper_state = 0.5

        # Assets
        self._load_images()

        # Bread grid geometry (for coverage)
        bread_w, bread_h = self.bread_img.get_size()
        bread_pos = ((700 - self.bread_img.get_width()) // 2,
             (VIEWPORT_H - self.bread_img.get_height()) - 30)
        bx0, by0 = bread_pos

        rows, cols = 4, 2
        cell_w = (bread_w-50) // cols   # ~250
        cell_h = bread_h // rows   # ~125

        self.box_size = (cell_w, cell_h)
        self.box_positions = []

        for r in range(rows):
            for c in range(cols):
                x = 25 + bx0 + c * cell_w
                y = by0 + r * cell_h
                self.box_positions.append((x, y))

        self.box_area = self.box_size[0] * self.box_size[1]

        

        # Click space to proceed to next episode
        self.manual_end = False

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

        # Per-spread bag art
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

        self.bread_img = _load("img_c/bread.png", (500, 500))
        self.bowl_img  = _load("img_c/bowl.png",  (100, 100))

    # ---------- gym API ----------

    def reset(self, seed: Optional[int] = None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.state = np.array([
            90.0, 90.0,           # rx, ry
            0.0,                  # gripper
            0.0,                  # holding flag
            *([0.0] * 8),         # coverage
            *([0.0] * 4),         # spread composition [jam, peanut, nutella, avocado]
            0.0                   # bread index
        ], dtype=np.float32)

        self.jam_lines.clear()
        self.spread_lines = {name: [] for name in SPREAD_TYPES}
        self.active_spread = "jam"
        self._update_goal()
        self.held_spread = None
        self.bag_current_pos = dict(BAG_ANCHORS)

        self.state[IDX_BI] = float(self.current_bread_index)

        self.done = False
        self.last_gripper_state = 0.5
        return self.state, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self._update_state(action)
        done = self._is_done()
        return self.state, 0.0, done, False, {}

    # ---------- internals ----------

    def _update_state(self, action: np.ndarray) -> None:
        # Desired new robot pose/gripper from action
        desired_x, desired_y, desired_g = float(action[0]), float(action[1]), float(action[2])

        # Enforce: outside tray, cannot set open (0.0). Force to hold (0.5).
        if not self._over_tray(desired_x, desired_y) and self.state[IDX_H] and desired_g == 0.0:
            desired_g = 0.5

        # Apply to state
        self.state[IDX_RX], self.state[IDX_RY], self.state[IDX_G] = desired_x, desired_y, desired_g
        robot_x, robot_y = self.state[IDX_RX], self.state[IDX_RY]
        gripper_state = self.state[IDX_G]

        # If not holding anything yet, allow pickup by proximity:
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
                    self.state[IDX_H] = 1.0
                    self._update_goal()
                    break

        # If holding a bag, make its position follow the robot and (optionally) deposit jam
        if self.held_spread is not None:
            # bag follows robot hand
            self.bag_current_pos[self.held_spread] = (robot_x, robot_y)

            if self.state[IDX_G] == 1.0:  # clutching
                jam_point = (int(robot_x + 35), int(robot_y + 70))
                # legacy combined trace
                if (not self.jam_lines) or (jam_point != self.jam_lines[-1]):
                    self.jam_lines.append(jam_point)
                    self._update_jam_coverage_area()
                # per-spread trace
                lines = self.spread_lines[self.held_spread]
                if (not lines) or (jam_point != lines[-1]):
                    lines.append(jam_point)
                    # update spread composition ratio
                    self._update_spread_composition()

        # --- Release logic: only drop if the TIP is inside this bag's dock square ---
        if self.held_spread is not None and gripper_state == 0.0:
            name = self.held_spread
            tip_x = float(robot_x + 35)
            tip_y = float(robot_y + 70)

            if self._bag_dock_rect(name).collidepoint(int(tip_x), int(tip_y)):
                # Allowed: snap back to that bag's tray anchor
                ax, ay = BAG_ANCHORS[name]
                self.bag_current_pos[name] = (ax, ay)
                self.state[IDX_H] = 0.0
                self.held_spread = None
            else:
                # Not allowed: keep holding
                self.state[IDX_G] = 0.5
                self.state[IDX_H] = 1.0
    
    def _update_goal(self) -> None:
        """Update the one-hot goal vector based on current active_spread."""
        self.goal = [1 if name == self.active_spread else 0 for name in SPREAD_TYPES]
    
    def _update_spread_composition(self) -> None:
        # Compute per-spread “painted” area inside bread, then normalize to fractions
        per_spread_area = {name: 0.0 for name in SPREAD_TYPES}
        for name, lines in self.spread_lines.items():
            if len(lines) < 2:
                continue
            for p1, p2 in zip(lines[:-1], lines[1:]):
                per_spread_area[name] += self._seg_area_inside_bread(p1, p2, self.JAM_WIDTH)

        total = sum(per_spread_area.values())
        if total <= 1e-8:
            comps = [0.0, 0.0, 0.0, 0.0]
        else:
            comps = [per_spread_area["jam"]/total,
                    per_spread_area["peanut"]/total,
                    per_spread_area["nutella"]/total,
                    per_spread_area["avocado"]/total]

        # Write into state
        self.state[IDX_SC0+0:IDX_SC0+4] = np.asarray(comps, dtype=np.float32)



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
                    # approximate fractional overlap allocation
                    denom = max(seg_rect.width * seg_rect.height, 1)
                    box_areas_covered[j] += (overlap_area / denom) * seg_area
        for i in range(8):
            self.state[IDX_C0 + i] = min(box_areas_covered[i] / self.box_area, 1.0)
    
    def _bread_union_rects(self) -> list[pygame.Rect]:
        # Return the 8 coverage box rects as the “toast” area
        return [pygame.Rect(int(bx), int(by), int(self.box_size[0]), int(self.box_size[1]))
                for (bx, by) in self.box_positions]

    def _seg_area_inside_bread(self, p1: tuple[int,int], p2: tuple[int,int], width: int) -> float:
        # Rectangle that approximates the thick line segment
        x1, y1 = p1; x2, y2 = p2
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        seg_len = max(dx, dy)
        seg_area = seg_len * width
        min_x = min(x1, x2) - width // 2
        min_y = min(y1, y2) - width // 2
        max_x = max(x1, x2) + width // 2
        max_y = max(y1, y2) + width // 2
        seg_rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)

        # Approximate intersection with toast by summing intersections with the 8 boxes
        inside_area = 0.0
        for box_rect in self._bread_union_rects():
            inter = seg_rect.clip(box_rect)
            if inter.width > 0 and inter.height > 0:
                # allocate proportionally from the seg_rect area to the thick line area
                denom = max(seg_rect.width * seg_rect.height, 1)
                inside_area += (inter.width * inter.height) / denom * seg_area
        return inside_area


    def _is_done(self) -> bool:
        if self.manual_end:
            self.done = True
            print("done (manual)")
            return True

        g_open = (self.state[IDX_G] == 0.0)
        not_holding = (self.state[IDX_H] == 0.0)
        coverage = self.state[IDX_C0:IDX_C0+8]
        done = g_open and not_holding and (np.mean(coverage) >= 0.9)
        self.done = bool(done)
        if self.done:
            print("done")
        return self.done


    # inside JamSpreadingEnv
    def _bag_dock_rect(self, name: str) -> pygame.Rect:
        ax, ay = BAG_ANCHORS[name]
        # same numbers as _draw_trays so logic == visuals
        return pygame.Rect(int(ax) - 35, int(ay) + 25, 70, 70)

    # ---------- human input helpers ----------

    def ready_to_help(self) -> bool:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        rx, ry = self.state[IDX_RX], self.state[IDX_RY]
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
                by = SIDE_PANEL_Y + (SIDE_PANEL_H - button_h) // 2
                return pygame.Rect(bx, by, button_w, button_h).collidepoint(mx, my)
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
        return False

    def _over_tray(self, x: float, y: float) -> bool:
        ix, iy = int(x) + 35, int(y) + 70
        for name in SPREAD_TYPES:
            if self._bag_dock_rect(name).collidepoint(ix, iy):
                return True
            
        return False

    def get_help(self) -> np.ndarray:
        # Cursor-controlled robot position
        x, y = pygame.mouse.get_pos()
        current = float(self.state[IDX_G])
        gripper = current

        over_tray = self._over_tray(self.state[IDX_RX], self.state[IDX_RY])

        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if over_tray:
                    # Full cycle 0.0 -> 0.5 -> 1.0 -> 0.0
                    cycle = [0.0, 0.5]
                    try:
                        i = cycle.index(round(current, 1))
                    except ValueError:
                        i = 0
                    gripper = cycle[(i + 1) % len(cycle)]
                else:
                    # Outside the tray: only toggle between hold (0.5) and clutch (1.0).
                    if current <= 0.5:
                        gripper = 0.5 if current == 0.0 else 1.0
                    else:
                        gripper = 0.5
                self.last_gripper_state = current

            elif event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        return np.array([x, y, gripper], dtype=np.float32)

    def _is_on_tray(self, name: str) -> bool:
        bx, by = self.bag_current_pos[name]
        ax, ay = BAG_ANCHORS[name]
        return np.hypot(bx - ax, by - ay) <= PICKUP_RADIUS

    def _render_toast_region(self, surface: pygame.Surface, bg_color=None) -> None:
        """
        Draw the toast area (bread + jam lines) into `surface`, whose size must match
        the captured region: (550, VIEWPORT_H - TOP_TRAY_H - 100).
        """
        # --- Region geometry (same as your subsurface slice) ---
        cap_x, cap_y = 75, TOP_TRAY_H + 100
        cap_w, cap_h = 550, (VIEWPORT_H - TOP_TRAY_H - 100)

        # Adjust background color to that of the top tray
        if TOP_TRAY_COLOR is not None:
            surface.fill(TOP_TRAY_COLOR)

        # --- Bread position (convert from screen coords to local coords in surface) ---
        bread_pos_screen = ((700 - self.bread_img.get_width()) // 2,
                            (VIEWPORT_H - self.bread_img.get_height()) - 30)
        bread_pos_local = (bread_pos_screen[0] - cap_x, bread_pos_screen[1] - cap_y)
        surface.blit(self.bread_img, bread_pos_local)

        # --- Draw jam for each spread, offsetting into local coords ---
        jam_width = self.JAM_WIDTH
        for name, lines in self.spread_lines.items():
            if len(lines) > 1:
                # Offset all points by (-cap_x, -cap_y)
                local_pts = [(px - cap_x, py - cap_y) for (px, py) in lines]
                pygame.draw.lines(surface, SPREAD_COLOR[name], False, local_pts, jam_width)

    def save_completed_toast(self, episode_num: int, bg_color=(245, 235, 220)) -> str:
        """
        Save the current toast **re-rendered** onto a custom background color.
        Default bg_color is a warm beige; pass None to keep transparency
        (PNG with alpha) if you create the surface with SRCALPHA.
        """
        os.makedirs("jam_state_img/completed", exist_ok=True)
        filename = f"jam_state_img/completed/{episode_num}.png"

        # Make an off-screen canvas the same size as your previous crop
        cap_w, cap_h = 550, (VIEWPORT_H - TOP_TRAY_H - 100)
        # Use SRCALPHA so we can support transparent backgrounds if bg_color=None
        canvas = pygame.Surface((cap_w, cap_h), pygame.SRCALPHA)

        # Re-render the toast area onto this canvas
        self._render_toast_region(canvas, bg_color=bg_color)

        # Save the off-screen render
        pygame.image.save(canvas, filename)
        print(f"Saved completed toast -> {filename}")
        return filename
    
    def _load_completed_thumbs(self) -> None:
        """Load all completed toasts as 75×75 thumbnails with 30px spacing."""
        self.completed_thumbs: list[pygame.Surface] = []
        folder = "jam_state_img/completed"
        if not os.path.exists(folder):
            return
        for fname in sorted(os.listdir(folder)):
            if fname.endswith(".png"):
                img = pygame.image.load(os.path.join(folder, fname)).convert_alpha()
                thumb = pygame.transform.scale(img, (100, 100))
                self.completed_thumbs.append(thumb)
    # ---------- drawing ----------

    def draw_intervene_button(self) -> None:
        button_w, button_h = 120, 40
        bx = SIDE_PANEL_X + (SIDE_PANEL_W - button_w) // 2
        by = SIDE_PANEL_Y + (SIDE_PANEL_H - button_h) // 2
        rect = pygame.Rect(bx, by, button_w, button_h)
        pygame.draw.rect(self.screen, (255, 224, 161), rect)
        font = pygame.font.SysFont("Arial", 20, bold=True)
        text = font.render("Intervene", True, (203, 91, 59))
        self.screen.blit(text, text.get_rect(center=rect.center))

    def _draw_info_row(self, x, y, label, value, font_label, font_value):
        label_surf = font_label.render(label, True, INFO_TEXT_COLOR)
        self.screen.blit(label_surf, (x, y))

        val_surf = font_value.render(str(value), True, INFO_TEXT_COLOR)
        pad_x, pad_y = 12, 6
        val_rect = val_surf.get_rect()
        val_rect.topleft = (x + 90, y - 2)
        bg_rect = pygame.Rect(val_rect.x - pad_x, val_rect.y - pad_y,
                            val_rect.w + 2*pad_x, val_rect.h + 2*pad_y)
        pygame.draw.rect(self.screen, INFO_BOX_FILL, bg_rect, border_radius=6)
        pygame.draw.rect(self.screen, INFO_BOX_OUTLINE, bg_rect, width=2, border_radius=6)
        self.screen.blit(val_surf, val_rect)

    def _draw_sidebar_info(self, episode_num: int, spread_name: str, mode_name: str):
        """Draw Toast Count / Spread / Mode blocks in the right sidebar."""
        font_label = pygame.font.SysFont("Arial", 18, bold=True)
        font_value = pygame.font.SysFont("Arial", 18)

        left_x = SIDE_PANEL_X + 20
        base_y = SIDE_PANEL_Y + 20

        self._draw_info_row(left_x, base_y +   0, "Count:", episode_num, font_label, font_value)
        self._draw_info_row(left_x, base_y +  66, "Spread:", spread_name, font_label, font_value)
        self._draw_info_row(left_x, base_y + 132, "Mode:", mode_name, font_label, font_value)


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
        g = self.state[IDX_G]
        key = "open" if g == 0.0 else ("hold" if g == 0.5 else "clutch")
        img = self.robot_images[key]
        pos = (int(self.state[IDX_RX]), int(self.state[IDX_RY]))
        self.screen.blit(img, img.get_rect(center=pos))

    def _draw_piping_bag(self) -> None:
        for name in SPREAD_TYPES:
            imgs = self.piping_bag_images[name]
            squeezed = (name == self.held_spread and self.state[IDX_G] == 1.0)
            img = imgs["squeezed" if squeezed else "normal"]

            if name == self.held_spread:
                # Draw at robot
                bag_pos = (int(self.state[IDX_RX]), int(self.state[IDX_RY]))
                self.screen.blit(img, img.get_rect(center=bag_pos))
            else:
                # Draw at its current world position (on tray or wherever it was dropped)
                bx, by = self.bag_current_pos[name]
                self.screen.blit(img, (int(bx) - 125, int(by) - 87))

    def _draw_trays(self) -> None:
        """Draw the top tray bar and four static docking rectangles at anchors."""
        pygame.draw.rect(self.screen, (245, 215, 181), TOP_TRAY_RECT)
        for name, (ax, ay) in BAG_ANCHORS.items():
            color = SPREAD_COLOR[name]
            dock_rect = self._bag_dock_rect(name)
            pygame.draw.rect(self.screen, color, dock_rect, 2)

    def _draw_jam(self) -> None:
        # draw each spread in its own color
        for name, lines in self.spread_lines.items():
            if len(lines) > 1:
                pygame.draw.lines(self.screen, SPREAD_COLOR[name], False, lines, self.JAM_WIDTH)

    def _draw_completed_thumbs(self) -> None:
        """
        Draw previously completed toast thumbnails along the top tray,
        each 75x75 with a 30px buffer.
        """
        if not hasattr(self, "completed_thumbs"):
            return
        x_start = 20  # left margin
        y_pos = 20    # keep them above the tray
        spacing = 75 + 30  # size + buffer

        for i, thumb in enumerate(self.completed_thumbs):
            self.screen.blit(thumb, (x_start + i * spacing, y_pos))
    
    def _draw_bread_boxes_debug(self) -> None:
        """
        Temporary debug overlay for bread coverage boxes.
        Uses self.box_positions (top-left coords) and self.box_size (w,h).
        """
        if not hasattr(self, "box_positions") or not hasattr(self, "box_size"):
            return

        w, h = self.box_size
        # Semi-transparent layer
        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        overlay.fill((0, 128, 255, 60))  # RGBA: light blue with alpha

        font = pygame.font.SysFont("Arial", 16, bold=True)

        for i, (bx, by) in enumerate(self.box_positions):
            # Fill (alpha) and outline
            self.screen.blit(overlay, (int(bx), int(by)))
            pygame.draw.rect(self.screen, (0, 100, 200), pygame.Rect(int(bx), int(by), w, h), width=2)

            # Index label in the corner
            label = font.render(str(i), True, (0, 60, 140))
            self.screen.blit(label, (int(bx) + 6, int(by) + 4))
    
    def _draw_state_tail_debug(self) -> None:
        """Display the last 5 elements of the state vector on screen for debugging."""
        tail = self.state[-5:]  # [comp1, comp2, comp3, comp4, bread_idx]
        font = pygame.font.SysFont("Arial", 18, bold=True)
        x, y = 20, 20  # top-left corner of the main window

        # background box for clarity
        bg_rect = pygame.Rect(x - 10, y - 10, 280, 150)
        pygame.draw.rect(self.screen, (250, 245, 230), bg_rect)
        pygame.draw.rect(self.screen, (150, 120, 90), bg_rect, width=2)

        # draw text lines
        lines = [
            f"jam:     {tail[0]:.2f}",
            f"peanut:  {tail[1]:.2f}",
            f"nutella: {tail[2]:.2f}",
            f"avocado: {tail[3]:.2f}",
            f"bread idx: {tail[4]:.0f}"
        ]
        for i, text in enumerate(lines):
            surf = font.render(text, True, (80, 50, 20))
            self.screen.blit(surf, (x, y + i * 25))




    def render(self, episode_num: int, step_id: int, screen: str = "help") -> None:
        self.screen.fill((255, 255, 255))

        side_rect = pygame.Rect(SIDE_PANEL_X, SIDE_PANEL_Y, SIDE_PANEL_W, SIDE_PANEL_H)
        pygame.draw.rect(self.screen, (247, 243, 238), side_rect)
        if screen != "ready":
            self.draw_intervene_button()
        else:
            self.draw_intervene_ready_text()
        # Sidebar info rows
        mode_label = "Help" if screen == "help" else ("Ready" if screen == "ready" else "Robot")
        self._draw_sidebar_info(
            episode_num=episode_num,
            spread_name=self.active_spread.capitalize(),
            mode_name=mode_label
        )


        bread_pos = ((700 - self.bread_img.get_width()) // 2, (VIEWPORT_H - self.bread_img.get_height()) - 30)
        self.screen.blit(self.bread_img, bread_pos)

        # self._draw_bread_boxes_debug()

        self._draw_jam()

        # periodic capture (left 700x700 region)
        if (time.time() - self.last_save_time) >= self.save_interval:
            os.makedirs("jam_state_img/test", exist_ok=True)
            filename = f"jam_state_img/test/episode{episode_num}_step{step_id}.png"
            subsurf = self.screen.subsurface((75, TOP_TRAY_H+100, 550, VIEWPORT_H - TOP_TRAY_H-100))
            pygame.image.save(subsurf, filename)
            self.save_counter += 1
            self.last_save_time = time.time()

        self._draw_trays()
        self._draw_completed_thumbs()
        self._draw_piping_bag()
        self._draw_robot()

        robot_x, robot_y = int(self.state[IDX_RX]), int(self.state[IDX_RY])

        # Draw the tip (small orange dot)
        pygame.draw.circle(self.screen, (255, 140, 0), (robot_x, robot_y), 6)

        # Draw pickup radius (transparent circle outline)
        pygame.draw.circle(self.screen, (0, 100, 255), (robot_x, robot_y), PICKUP_RADIUS, 2)

        # Pickup outlines at trays; if holding, draw circle at tip for the held bag
        for name, (ax, ay) in BAG_ANCHORS.items():
            color = SPREAD_COLOR[name]
            if name == self.held_spread:
                pygame.draw.circle(self.screen, color, (robot_x, robot_y), int(PICKUP_RADIUS), 2)
            else:
                pygame.draw.circle(self.screen, color, (int(ax), int(ay)), int(PICKUP_RADIUS), 2)

        # self._draw_state_tail_debug()
        

        self.clock.tick(FPS)
        pygame.display.flip()


# main entry point
if __name__ == "__main__":
    all_episodes = []
    num_episodes = int(input("Enter number of episodes to run: "))
    
    pygame.init()
    pygame.display.init()

    env = JamSpreadingEnv()

    # Show *something* even if the policy path errors after
    env.render(episode_num=0, step_id=0, screen="robot")
    pygame.event.pump()
    pygame.display.flip()
    pygame.time.wait(50)

    # load policy
    action_dim = 3
    state_dim = 17 * 3  
    cont_policy = Continuous_Policy(state_dim=state_dim, output_dim=action_dim)
    sd = torch.load(MODEL_PATH, map_location="cpu")
    cont_policy.load_state_dict(sd, strict=False)
    cont_policy.eval()

    if not HUMAN_ALWAYS_CONTROL:

        # store these to use in get_prediction
        stats = np.load(STATS_PATH)
        x_min = stats["x_min"].astype(np.float32)
        x_max = stats["x_max"].astype(np.float32)
        y_min = stats["y_min"].astype(np.float32)
        y_max = stats["y_max"].astype(np.float32)

        assert x_min.shape[0] == state_dim, f"x_min dim {x_min.shape[0]} != {state_dim}"
        assert x_max.shape[0] == state_dim, f"x_max dim {x_max.shape[0]} != {state_dim}"
        assert y_min.shape[0] == action_dim and y_max.shape[0] == action_dim

        x_range = (x_max - x_min).astype(np.float32)
        y_range = (y_max - y_min).astype(np.float32)

        const_x_mask = x_range <= 1e-8
        const_y_mask = y_range <= 1e-8

        if np.any(const_x_mask):
            idx = np.where(const_x_mask)[0]
            print("[WARN] Constant input feature(s) in stats at indices:", idx.tolist())
            # Avoid divide-by-zero; normalized value doesn't matter (always same), set scale to 1
            x_range[const_x_mask] = 1.0

        if np.any(const_y_mask):
            idx = np.where(const_y_mask)[0]
            print("[WARN] Constant target dim(s) in stats at indices:", idx.tolist())
            y_range[const_y_mask] = 1.0

    for episode_num in range(num_episodes):
        robot_prediction = None
        env.current_bread_index = episode_num

        obs, _ = env.reset()
        env._load_completed_thumbs()  # refresh the thumbnails for this episode
        obs_prev1 = obs.copy()
        obs_prev2 = obs.copy()

        done = False
        help_start_time: Optional[float] = None
        screen_state = "robot"
        last_help_check_time = time.time()
        episode = Episode(episode_num)
        space_was_down = False
        env.manual_end = False
        step_id = 0

        # Conformal tracking vars (kept; used only if not human-only)
        # q_lo = np.array([0.1, 0.1, 0.1])
        # q_hi = np.array([0.1, 0.1, 0.1])
        # stepsize = 0.2
        # alpha_desired = 0.8
        # list_of_uncertainties: List[float] = []
        # list_of_residuals: List[float] = []
        # history_upper_residuals: List[np.ndarray] = []
        # history_lower_residuals: List[np.ndarray] = []
        # B_t_lookback_window = 100

        while not done:
            if HUMAN_ALWAYS_CONTROL:
                screen_state = "help"
                context = "human_intervened"
                action = env.get_help()
                robot_prediction = None

                keys = pygame.key.get_pressed()
                if keys[pygame.K_SPACE] and not space_was_down:
                    env.save_completed_toast(episode_num)
                    env.manual_end = True
                    controlled_delay(3000)
            else:
                
                # pure robot mode, no ask-for-help logic
                action = get_prediction(
                    obs, obs_prev1, obs_prev2,
                    cont_policy, x_min, x_max, y_min, y_max
                )
                
                g = action[2]
                action[2] = min([0.0, 0.5, 1.0], key=lambda v: abs(v-g))
                
                # clip to env action space just in case
                action = np.clip(action, env.action_space.low, env.action_space.high)
                
                context = "robot_independent"
                screen_state = "robot"
                print(action)


            # else:
            #     robot_prediction = get_prediction(obs, obs_prev1, obs_prev2, cont_policy)
            #     print("robot_prediction", robot_prediction)

            #     if screen_state == "robot":
            #         need_help = False
            #         if time.time() - last_help_check_time >= 2.0:
            #             uncertainty_at_timestep = np.linalg.norm(q_hi + q_lo)
            #             print("uncertainty_at_timestep", uncertainty_at_timestep)
            #             list_of_uncertainties.append(float(uncertainty_at_timestep))
            #             need_help = uncertainty_at_timestep > 10

            #         if env.check_intervene_click():
            #             screen_state = "ready"
            #             context = "human_intervened"
            #         elif need_help:
            #             screen_state = "ready"
            #             context = "robot_asked"
            #         else:
            #             action = robot_prediction
            #             context = "robot_independent"
            #             screen_state = "robot"

            #     elif screen_state == "ready":
            #         if env.ready_to_help():
            #             screen_state = "help"
            #             help_start_time = time.time()
            #         else:
            #             env.render(episode_num, step_id, screen_state)
            #             time.sleep(1 / 10)
            #             continue

            #     elif screen_state == "help":
            #         action = env.get_help()

            #         if help_start_time and time.time() - help_start_time >= 3.0:
            #             # Snap back to nearest script point (unchanged)
            #             rx, ry = env.state[IDX_RX], env.state[IDX_RY]
            #             min_dist = float("inf")
            #             best_idx = next_action_idx
            #             for i in range(len(jam_sample_actions)):
            #                 ax, ay = jam_sample_actions[i][0], jam_sample_actions[i][1]
            #                 d = ((ax - rx) ** 2 + (ay - ry) ** 2) ** 0.5
            #                 if d < min_dist:
            #                     min_dist = d
            #                     best_idx = i
            #             next_action_idx = best_idx
            #             next_action = jam_sample_actions[next_action_idx]
            #             last_help_check_time = time.time()
            #             screen_state = "robot"

            #         # Conformal stats (unchanged)
            #         expert_y = action
            #         y_pred = robot_prediction
            #         shi_upper_residual = expert_y - y_pred
            #         slo_lower_residual = y_pred - expert_y
            #         list_of_residuals.append(float(np.linalg.norm(np.abs(expert_y - y_pred))))

            #         err_hi = np.zeros(action_dim)
            #         err_lo = np.zeros(action_dim)
            #         for i in range(action_dim):
            #             if shi_upper_residual[i] > q_hi[i]:
            #                 err_hi[i] = 1
            #             if slo_lower_residual[i] > q_lo[i]:
            #                 err_lo[i] = 1
            #         print("err_hi", err_hi)
            #         print("err_lo", err_lo)
            #         covered = sum(err_hi) + sum(err_lo)
            #         covered = 0 if covered > 0 else 1
            #         print("covered", covered)

            #         B_hi = np.ones(action_dim) * 0.01
            #         B_lo = np.ones(action_dim) * 0.01
            #         if len(history_upper_residuals) > 0:
            #             B_hi = np.max(history_upper_residuals, axis=0)
            #             B_lo = np.max(history_lower_residuals, axis=0)

            #         history_upper_residuals.append(shi_upper_residual)
            #         history_lower_residuals.append(slo_lower_residual)
            #         if len(history_upper_residuals) > B_t_lookback_window:
            #             history_upper_residuals.pop(0)
            #             history_lower_residuals.pop(0)

            #         q_hi = q_hi + (stepsize) * B_hi * (err_hi - alpha_desired)
            #         q_lo = q_lo + (stepsize) * B_lo * (err_lo - alpha_desired)

            # Book-keeping
            obs_prev2 = obs_prev1
            obs_prev1 = obs
            # action = [50, 50, 0]
            if action is not None:
                # Defensive guard that does not change your logic:
                # skip step if NaN/Inf; otherwise clamp into action box
                if not np.all(np.isfinite(action)):
                    print("Skipping step due to invalid action:", action)
                else:
                    # action = np.clip(action, env.action_space.low, env.action_space.high)
                    obs, _, done, _, _ = env.step(action)

                    os.makedirs("jam_state_img/test", exist_ok=True)
                    img_path = f"jam_state_img/test/episode{episode_num}_step{step_id}.png"
                    timestep = TimeStep(
                        step_id=step_id,
                        action=action,
                        state_v=obs,
                        state_img_path=img_path,
                        context=context,
                        robot_prediction=None if HUMAN_ALWAYS_CONTROL else robot_prediction,
                    )
                    episode.add_step(timestep)

            env.render(episode_num, step_id, screen_state)

            step_id += 1
            time.sleep(1 / 10)

        print(f"Episode {episode_num} ended.")
        all_episodes.append(episode)
        controlled_delay(3000)
    
    env.close()
    del env
    if pygame.get_init():
        pygame.quit()
    print("Environment closed successfully.")
