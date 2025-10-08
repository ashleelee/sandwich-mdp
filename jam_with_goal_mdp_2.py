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
        self.jam_lines: List[Tuple[int, int]] = []
        self.save_interval = 0.0
        self.last_save_time = time.time()
        self.save_counter = 0

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
        self.piping_bag_images = {
            "normal":   _load("img_c/normal.png", (175, 175)),
            "squeezed": _load("img_c/squeezed.png", (175, 175)),
        }
        self.bread_img = _load("img_c/bread.png", (550, 550))
        self.bowl_img = _load("img_c/bowl.png", (100, 100))

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
        self.state[6], self.state[7], self.state[8] = action[0], action[1], action[2]
        rx, ry = self.state[6], self.state[7]
        bx, by = self.state[2], self.state[3]
        g = self.state[8]

        # Pick up bag if close & holding
        if np.hypot(rx - bx, ry - by) <= 40.0 and g == 0.5:
            self.state[9] = 1.0

        # Deposit jam while clutching and holding
        if self.state[8] == 1.0 and self.state[9] == 1.0 and not self.hit_bread_endpoints:
            jp = (int(rx + 35), int(ry + 70))
            if not self.jam_lines or jp != self.jam_lines[-1]:
                self.jam_lines.append(jp)
                self._update_jam_coverage_area()

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
        tip_x, tip_y = rx + 35, ry + 70
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

    def get_help(self) -> np.ndarray:
        x, y = pygame.mouse.get_pos()
        current = float(self.state[8])
        gripper = current
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if current == 0.0:
                    gripper = 0.5
                elif current == 0.5:
                    gripper = 0.0 if self.last_gripper_state == 1.0 else 1.0
                else:
                    gripper = 0.5
                self.last_gripper_state = current
            elif event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
        return np.array([x, y, gripper], dtype=np.float32)

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
        holding = self.state[9]
        img = self.piping_bag_images["squeezed" if holding == 1.0 and self.state[8] == 1.0 else "normal"]
        if holding == 0.0:
            bx, by = int(self.state[2]), int(self.state[3])
            self.screen.blit(img, (bx - 125, by - 87))
        else:
            rx, ry = int(self.state[6]), int(self.state[7])
            self.screen.blit(img, img.get_rect(center=(rx, ry)))

    def _draw_jam(self) -> None:
        if len(self.jam_lines) > 1:
            pygame.draw.lines(self.screen, (255, 0, 0), False, self.jam_lines, self.JAM_WIDTH)

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

        self.screen.blit(self.bowl_img, (560, 75))
        self._draw_piping_bag()
        self._draw_robot()

        bread_x, bread_y = int(self.state[4]), int(self.state[5])
        pygame.draw.circle(self.screen, (255, 165, 0), (bread_x, bread_y), 25)
        pygame.draw.circle(self.screen, (255, 165, 0), (610, 140), 25)

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
