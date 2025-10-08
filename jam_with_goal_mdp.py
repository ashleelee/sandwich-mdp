import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import pickle
import time
import random
import os
import torch

from jam_data_classes import TimeStep, Episode
from policies.conformal.mlp import Continuous_Policy

# -----------------------------
# Layout / UI config
# -----------------------------
FPS = 30

# Canvas (left) + side panel (right)
CANVAS_W = 840          # wider canvas so the top strip is wider than the bread
PANEL_W  = 260
VIEWPORT_W = CANVAS_W + PANEL_W
VIEWPORT_H = 860        # taller window

# Top thumbnails strip
TOP_BAR_H = 130
THUMB_W, THUMB_H = 96, 68
THUMB_SPACING = 12
THUMB_MARGIN_X = 14

# Bread
BREAD_W, BREAD_H = 450, 450                  # (1) smaller bread
BREAD_TOP = TOP_BAR_H + 150
BREAD_X = (CANVAS_W - BREAD_W) // 2          # centered in canvas

# Bag + picking/placing logic
GRAB_RADIUS = 44      # distance to a bag home to grab
PLACE_RADIUS = 44     # distance to a bag home to "place" the bag

# Spreads: name, selector color, jam color, file suffix
SPREADS = [
    ("Jam",            (239,120,120), (255, 60, 60),  ""),
    ("Peanut Butter",  (245,178,82),  (188,140,77),   "_b"),
    ("Nutella",        (104,74,44),   (90, 60, 40),   "_c"),
    ("Avocado",        (152,223,152), (85,160, 85),   "_d"),
]

# Bag anchors under/near the top selector row (x is spread across the canvas)
BAG_ANCHORS = [
    (CANVAS_W//2 - 210, TOP_BAR_H - 50),
    (CANVAS_W//2 - 70,  TOP_BAR_H - 50),
    (CANVAS_W//2 + 70,  TOP_BAR_H - 50),
    (CANVAS_W//2 + 210, TOP_BAR_H - 50),
]

# spread selector row layout
SPREAD_ROW_Y = TOP_BAR_H + 70   # move this larger to push the row further down
SELECTOR_W = 78
SELECTOR_H = 78
BAG_ICON_SIZE = 100           # bag icon inside the selector box

# -----------------------------
# Load demo episode
# -----------------------------
with open("jam_all_episodes_gen.pkl", "rb") as f:
    episodes_pickle = pickle.load(f)
ep0 = episodes_pickle[0]
jam_sample_actions = [step.action.tolist() for step in ep0.steps]

# -----------------------------
# Env
# -----------------------------
class JamSpreadingEnv(gym.Env):

    def __init__(self):
        super().__init__()

        # Pygame display MUST be ready before convert_alpha
        pygame.init()
        pygame.display.init()
        self.screen: pygame.Surface = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        self.clock = pygame.time.Clock()

        # State & UI
        self.jam_lines = []
        self.jam_width = 36  # slightly slimmer lines to fit the smaller bread
        self.action_log = []
        self.round_thumbnails = []  # past breads
        self.current_spread_idx = 0  # UI selection highlight
        self.held_spread_idx = -1    # which bag the robot is actually holding (-1 = none)

        # Timing for saving images
        self.save_interval = 0.0
        self.last_save_time = time.time()
        self.save_counter = 0
        self._init_selectors()

        # Build sub-systems
        self._init_state()
        self._init_spaces()
        self._load_images()
        self._init_layout_and_boxes()  # boxes depend on bread rect
        # self._init_selectors()

        self.done = False
        self.hit_bread_endpoints = False
        self.last_gripper_state = 0.5

    # ---------- state and spaces ----------
    def _init_state(self):
        # dynamic endpoints: one point near bottom of bread (for "touch bread" goal)
        bread_end_x = BREAD_X + BREAD_W//2
        bread_end_y = BREAD_TOP + BREAD_H - 32

        # initial robot and a "generic" bag home (not actually used for logic now)
        self.state = np.array([
            90.0, 90.0,
            self.bag_home_positions[0][0], self.bag_home_positions[0][1],  # legacy fields
            bread_end_x, bread_end_y,
            90.0, 90.0,
            0.0, 0.0,
            *([0.0] * 8)
        ], dtype=np.float32)


    def _init_spaces(self):
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([CANVAS_W, VIEWPORT_H, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array([0.0] * 18, dtype=np.float32),
            high=np.array([
                CANVAS_W, VIEWPORT_H,   # start
                CANVAS_W, VIEWPORT_H,   # bag pos (legacy; unused)
                CANVAS_W, VIEWPORT_H,   # bread endpoint
                CANVAS_W, VIEWPORT_H,   # robot pos
                1.0,                    # gripper
                1.0,                    # holding
                *([1.0] * 8)
            ], dtype=np.float32),
            dtype=np.float32
        )

    def _load_images(self):
        # Robot hands
        self.robot_images = {
            "open":   pygame.transform.scale(pygame.image.load("img_c/open.png").convert_alpha(),   (175, 175)),
            "hold":   pygame.transform.scale(pygame.image.load("img_c/hold.png").convert_alpha(),   (175, 175)),
            "clutch": pygame.transform.scale(pygame.image.load("img_c/clutch.png").convert_alpha(), (175, 175)),
        }

        # Bags per spread
        self.piping_bag_images = []
        for _name, _ui, _jam, suf in SPREADS:
            nf = f"img_c/normal{suf}.png" if suf else "img_c/normal.png"
            sf = f"img_c/squeezed{suf}.png" if suf else "img_c/squeezed.png"
            self.piping_bag_images.append({
                "normal":   pygame.transform.scale(pygame.image.load(nf).convert_alpha(), (175, 175)),
                "squeezed": pygame.transform.scale(pygame.image.load(sf).convert_alpha(), (175, 175)),
            })

        # Bread + bowl
        self.bread_img = pygame.transform.scale(pygame.image.load("img_c/bread.png").convert_alpha(), (BREAD_W, BREAD_H))
        self.bowl_img  = pygame.transform.scale(pygame.image.load("img_c/bowl.png").convert_alpha(),  (100, 100))

    def _init_layout_and_boxes(self):
        # bread rect and dynamic 2x4 grid inside it
        self.bread_rect = pygame.Rect(BREAD_X, BREAD_TOP, BREAD_W, BREAD_H)

        # 2 columns, 4 rows with small gutters
        gutter = 10
        box_w = (BREAD_W // 2) - gutter
        box_h = (BREAD_H // 4) - gutter

        self.box_size = (box_w, box_h)
        self.box_positions = []
        for r in range(4):
            for c in range(2):
                x = BREAD_X + c * (box_w + gutter) + (gutter // 2)
                y = BREAD_TOP + r * (box_h + gutter) + (gutter // 2)
                self.box_positions.append((x, y))
        self.box_area = box_w * box_h

    def _init_selectors(self):
        # centers spaced across the canvas width
        centers_x = [
            CANVAS_W//2 - 210,
            CANVAS_W//2 - 70,
            CANVAS_W//2 + 70,
            CANVAS_W//2 + 210,
        ]
        self.selector_rects = []
        for cx in centers_x:
            rect = pygame.Rect(0, 0, SELECTOR_W, SELECTOR_H)
            rect.center = (cx, SPREAD_ROW_Y)
            self.selector_rects.append(rect)

        # use selector centers as the true "bag home" positions for grab/place logic
        self.bag_home_positions = [r.center for r in self.selector_rects]


    # ---------- gym api ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.jam_lines.clear()
        self.done = False
        self.hit_bread_endpoints = False
        self.last_gripper_state = 0.5
        self.held_spread_idx = -1
        self._init_state()                   # refresh endpoints
        return self.state, {}

    def step(self, action):
        self._update_state_with_action(action)
        self._maybe_grab_or_place_bag()
        self._check_hit_bread_endpoints()
        done = self._is_done()
        return self.state, 0.0, done, False, {}

    # ---------- interaction logic ----------
    def _update_state_with_action(self, action):
        self.state[6] = action[0]
        self.state[7] = action[1]
        self.state[8] = action[2]

        # leave legacy bag fields untouched (we render bag homes from BAG_ANCHORS)

        # draw jam when clutching with a held bag
        if self.state[8] == 1.0 and self.state[9] == 1.0 and not self.hit_bread_endpoints:
            jam_point = (int(self.state[6] + 35), int(self.state[7] + 70))
            if (not self.jam_lines) or jam_point != self.jam_lines[-1]:
                self.jam_lines.append(jam_point)
                self._update_jam_coverage_area()

    def _maybe_grab_or_place_bag(self):
        """ (2) & (4) Grab closest bag when holding; place bag at its home to finish. """
        rx, ry = self.state[6], self.state[7]
        g = self.state[8]     # 0 open, 0.5 hold, 1 clutch

        # GRAB
        if g == 0.5 and self.state[9] == 0.0:
            closest_i, closest_d = -1, 1e9
            for i, (hx, hy) in enumerate(self.bag_home_positions):
                d = ((rx - hx)**2 + (ry - hy)**2)**0.5
                if d < closest_d:
                    closest_i, closest_d = i, d
            if closest_d <= GRAB_RADIUS:
                self.state[9] = 1.0
                self.held_spread_idx = closest_i
                self.current_spread_idx = closest_i

        # PLACE
        if g == 0.0 and self.state[9] == 1.0 and self.held_spread_idx != -1:
            hx, hy = self.bag_home_positions[self.held_spread_idx]
            d = ((rx - hx)**2 + (ry - hy)**2)**0.5
            if d <= PLACE_RADIUS:
                self.state[9] = 0.0
                if self.hit_bread_endpoints:
                    self.done = True


    def _update_jam_coverage_area(self):
        if len(self.jam_lines) < 2:
            return
        box_areas_covered = [0.0] * 8
        w = self.jam_width

        for i in range(len(self.jam_lines) - 1):
            x1, y1 = self.jam_lines[i]
            x2, y2 = self.jam_lines[i + 1]
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

    def _is_done(self):
        # Done is now controlled by placing the bag back home *after* touching bread endpoint.
        return self.done

    def _check_hit_bread_endpoints(self):
        rx, ry = self.state[6], self.state[7]
        bread_x, bread_y = self.state[4], self.state[5]  # dynamic endpoint near bottom center of bread
        tip_x, tip_y = rx + 35, ry + 70
        if ((tip_x - bread_x)**2 + (tip_y - bread_y)**2)**0.5 <= 28:
            self.hit_bread_endpoints = True

    # ---------- input helpers ----------
    def _cycle_gripper_on_click(self, current):
        if current == 0.0:   return 0.5
        if current == 0.5:   return 1.0 if self.last_gripper_state != 1.0 else 0.0
        return 0.5

    def ready_to_help(self):
        # unchanged (click near robot dot)
        mouse_x, mouse_y = pygame.mouse.get_pos()
        rx, ry = self.state[6], self.state[7]
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                dist = ((mouse_x - rx)**2 + (mouse_y - ry)**2)**0.5
                return dist <= 30
            elif event.type == pygame.QUIT:
                pygame.quit(); exit()
        return False

    def check_intervene_click(self):
        # right-panel Intervene button + spread selector clicks
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                button_rect = pygame.Rect(CANVAS_W + (PANEL_W - 140)//2, (VIEWPORT_H - 40)//2, 140, 44)
                if button_rect.collidepoint(mx, my):
                    return True
                # selectors
                for i, r in enumerate(self.selector_rects):
                    if r.collidepoint(mx, my):
                        self.current_spread_idx = i
                        return False
            elif event.type == pygame.QUIT:
                pygame.quit(); exit()
        return False

    def get_help(self):
        x, y = pygame.mouse.get_pos()
        gr = self.state[8]
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                newg = self._cycle_gripper_on_click(gr)
                self.last_gripper_state = gr
                gr = newg
            elif event.type == pygame.QUIT:
                pygame.quit(); exit()
        return np.array([x, y, gr], dtype=np.float32)

    # ---------- drawing ----------
    def _draw_robot(self):
        g = self.state[8]
        img = self.robot_images["open"] if g == 0.0 else (self.robot_images["hold"] if g == 0.5 else self.robot_images["clutch"])
        rect = img.get_rect(center=(int(self.state[6]), int(self.state[7])))
        self.screen.blit(img, rect)

    def _draw_bags(self):
        """Draw the 4 bags at their homes; if holding one, draw that bag at the robot."""
        # draw home bags
        for i, (cx, cy) in enumerate(BAG_ANCHORS):
            bag_imgs = self.piping_bag_images[i]
            bag = bag_imgs["normal"]
            small = pygame.transform.scale(bag, (110, 110))
            self.screen.blit(small, small.get_rect(center=(cx, cy)))
        # draw held bag on robot (overwrites home visual)
        if self.state[9] == 1.0 and self.held_spread_idx != -1:
            bag_imgs = self.piping_bag_images[self.held_spread_idx]
            bag = bag_imgs["squeezed"] if self.state[8] == 1.0 else bag_imgs["normal"]
            pos = (int(self.state[6]), int(self.state[7]))
            self.screen.blit(bag, bag.get_rect(center=pos))

    def _draw_jam(self):
        if len(self.jam_lines) > 1 and self.held_spread_idx != -1:
            jam_color = SPREADS[self.held_spread_idx][2]
            pygame.draw.lines(self.screen, jam_color, False, self.jam_lines, self.jam_width)

    def _draw_top_thumbnails(self):
        if not self.round_thumbnails:
            return
        x = THUMB_MARGIN_X
        y = (TOP_BAR_H - THUMB_H)//2
        for img in self.round_thumbnails[-12:]:
            self.screen.blit(img, (x, y))
            pygame.draw.rect(self.screen, (176,148,120), pygame.Rect(x, y, THUMB_W, THUMB_H), 2, border_radius=8)
            x += THUMB_W + THUMB_SPACING

    def _draw_spread_selectors(self):
        font = pygame.font.SysFont("Arial", 16)
        for i, ((name, box_color, _jam, _suf), rect) in enumerate(zip(SPREADS, self.selector_rects)):
            # outer colored box
            pygame.draw.rect(self.screen, box_color, rect, width=6, border_radius=10)
            if i == self.current_spread_idx:
                pygame.draw.rect(self.screen, (0,0,0), rect, width=3, border_radius=10)

            # bag icon centered INSIDE the box
            bag = self.piping_bag_images[i]["normal"]
            bag_small = pygame.transform.smoothscale(bag, (BAG_ICON_SIZE, BAG_ICON_SIZE))
            self.screen.blit(bag_small, bag_small.get_rect(center=rect.center))

            # label under the box
            label = font.render(name, True, (0,0,0))
            self.screen.blit(label, (rect.centerx - label.get_width()//2, rect.bottom + 6))


    def draw_episode_num(self, episode_num):
        font = pygame.font.SysFont("Arial", 16, bold=True)
        ep_text = font.render(f"Episode: {episode_num}", True, (0,0,0))
        self.screen.blit(ep_text, (10, TOP_BAR_H + 6))

    def render(self, episode_num, step_id, screen="help"):
        # background
        self.screen.fill((255,255,255))

        # top bar (wider than bread; spans full CANVAS_W)
        pygame.draw.rect(self.screen, (234,220,200), pygame.Rect(0, 0, CANVAS_W, TOP_BAR_H))
        self._draw_top_thumbnails()

        # side panel
        side_panel_rect = pygame.Rect(CANVAS_W, 0, PANEL_W, VIEWPORT_H)
        pygame.draw.rect(self.screen, (247,243,238), side_panel_rect)

        # Intervene or instructions
        if screen != "ready":
            button_rect = pygame.Rect(CANVAS_W + (PANEL_W - 140)//2, (VIEWPORT_H - 40)//2, 140, 44)
            pygame.draw.rect(self.screen, (255,224,161), button_rect)
            t = pygame.font.SysFont("Arial", 20, bold=True).render("Intervene", True, (203,91,59))
            self.screen.blit(t, t.get_rect(center=button_rect.center))
        else:
            f = pygame.font.SysFont("Arial", 14, bold=True)
            for i, line in enumerate(["Click the dot on the robot arm", "then drag to guide it."]):
                s = f.render(line, True, (139,69,19))
                self.screen.blit(s, s.get_rect(center=(CANVAS_W + PANEL_W//2, 300 + i*28)))

        # bread
        self.screen.blit(self.bread_img, self.bread_rect.topleft)

        # selectors & bags
        self._draw_spread_selectors()
        self._draw_bags()

        # jam & robot
        self._draw_jam()
        self._draw_robot()

        # endpoint visual (small orange dot)
        pygame.draw.circle(self.screen, (255,165,0), (int(self.state[4]), int(self.state[5])), 22)

        # mode & ep number
        font = pygame.font.SysFont("Arial", 16, bold=True)
        self.screen.blit(font.render(f"Mode: {screen}", True, (0,0,0)), (10, TOP_BAR_H + 28))
        self.draw_episode_num(episode_num)

        # save current canvas portion if needed (left side only)
        t = time.time()
        if t - self.last_save_time >= self.save_interval:
            os.makedirs("jam_state_img/test", exist_ok=True)
            filename = f"jam_state_img/test/episode{episode_num}_step{step_id}.png"
            subsurf = self.screen.subsurface((0, 0, CANVAS_W, VIEWPORT_H))
            pygame.image.save(subsurf, filename)
            self.save_counter += 1
            self.last_save_time = t

        self.clock.tick(FPS)
        pygame.display.flip()

    def capture_round_thumbnail(self, episode_num, final_step_id):
        path = f"jam_state_img/test/episode{episode_num}_step{final_step_id}.png"
        if os.path.exists(path):
            surf = pygame.image.load(path).convert_alpha()
            # crop bread area inside the saved canvas (same coords)
            crop_rect = self.bread_rect.copy()
            bread_crop = surf.subsurface(crop_rect).copy()
            bread_small = pygame.transform.smoothscale(bread_crop, (THUMB_W, THUMB_H))
            self.round_thumbnails.append(bread_small)

# -----------------------------
# Misc helper
# -----------------------------
def controlled_delay(ms):
    start = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start < ms:
        pygame.event.pump()

# -----------------------------
# Policy helper (unchanged numerics)
# -----------------------------
def get_prediction(obs, obs_prev1, obs_prev2, policy_model):
    # your original normalization constants kept verbatim
    min_X = np.array([np.float32(90.0), np.float32(78.0), np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(
        0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(
        0.0), np.float32(92.0), np.float32(74.0), np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(
        0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(
        0.0), np.float32(92.0), np.float32(74.0), np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(
        0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0)])
    max_X = np.array([np.float32(573.0), np.float32(524.0), np.float32(1.0), np.float32(1.0), np.float32(0.25068787), np.float32(
        0.075533986), np.float32(0.25659528), np.float32(0.6491279), np.float32(0.52741855), np.float32(
        0.40159684), np.float32(0.40303788), np.float32(0.4491095), np.float32(573.0), np.float32(524.0), np.float32(
        1.0), np.float32(1.0), np.float32(0.25068787), np.float32(0.075533986), np.float32(0.25659528), np.float32(
        0.6491279), np.float32(0.52741855), np.float32(0.40159684), np.float32(0.40303788), np.float32(
        0.4491095), np.float32(573.0), np.float32(524.0), np.float32(1.0), np.float32(1.0), np.float32(
        0.25068787), np.float32(0.075533986), np.float32(0.25659528), np.float32(0.6491279), np.float32(
        0.52741855), np.float32(0.40159684), np.float32(0.40303788), np.float32(0.4491095)])
    min_Y = np.array([np.float32(136.0), np.float32(74.0), np.float32(0.0)])
    max_Y = np.array([np.float32(573.0), np.float32(524.0), np.float32(1.0)])

    input_obs = np.concatenate((obs_prev2[6:], obs_prev1[6:], obs[6:]), axis=0)
    input_obs = (input_obs - min_X) / (max_X - min_X)

    state_tensor = torch.tensor(np.array(input_obs)).float().unsqueeze(0)
    with torch.no_grad():
        action_pred = policy_model(state_tensor)
        action_pred = action_pred.detach().numpy() * (max_Y - min_Y) + min_Y
    return action_pred[0]

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    try:
        user_in = input("Enter number of episodes to run (press Enter for 10): ").strip()
        num_episodes = int(user_in) if user_in else 10
    except Exception:
        num_episodes = 10

    env = JamSpreadingEnv()

    # policy (unchanged)
    action_dim = 3
    state_dim = 36
    cont_policy = Continuous_Policy(state_dim=state_dim, output_dim=action_dim)
    cont_policy.load_state_dict(torch.load('cont_policy.pth'))
    cont_policy.eval()

    for episode_num in range(num_episodes):
        obs, _ = env.reset()
        obs_prev1 = obs.copy()
        obs_prev2 = obs.copy()
        done = False
        screen_state = "robot"
        last_help_check_time = time.time()
        episode = Episode(episode_num)
        step_id = 0

        # conformal params (kept)
        q_lo = np.array([0.1, 0.1, 0.1])
        q_hi = np.array([0.1, 0.1, 0.1])
        stepsize = 0.2
        alpha_desired = 0.8
        history_upper_residuals, history_lower_residuals = [], []
        B_t_lookback_window = 100

        while not done:
            robot_prediction = get_prediction(obs, obs_prev1, obs_prev2, cont_policy)

            if screen_state == "robot":
                need_help = False
                if time.time() - last_help_check_time >= 2.0:
                    uncertainty_at_timestep = np.linalg.norm(q_hi + q_lo)
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

            elif screen_state == "ready":
                if env.ready_to_help():
                    screen_state = "help"
                    help_start_time = time.time()
                else:
                    env.render(episode_num, step_id, screen_state)
                    time.sleep(1/10)
                    continue

            elif screen_state == "help":
                action = env.get_help()
                if time.time() - help_start_time >= 3.0:
                    last_help_check_time = time.time()
                    screen_state = "robot"

                # conformal updates
                expert_y = action
                y_pred = robot_prediction
                shi = expert_y - y_pred
                slo = y_pred - expert_y

                err_hi = (shi > q_hi).astype(float)
                err_lo = (slo > q_lo).astype(float)

                B_hi = np.ones(action_dim) * 0.01
                B_lo = np.ones(action_dim) * 0.01
                if history_upper_residuals:
                    B_hi = np.max(history_upper_residuals, axis=0)
                    B_lo = np.max(history_lower_residuals, axis=0)

                history_upper_residuals.append(shi)
                history_lower_residuals.append(slo)
                if len(history_upper_residuals) > B_t_lookback_window:
                    history_upper_residuals.pop(0)
                    history_lower_residuals.pop(0)

                q_hi = q_hi + stepsize * B_hi * (err_hi - alpha_desired)
                q_lo = q_lo + stepsize * B_lo * (err_lo - alpha_desired)

            # advance env
            obs_prev2 = obs_prev1
            obs_prev1 = obs
            if screen_state != "ready":
                obs, _, done, _, _ = env.step(action)
                img_path = f"jam_state_img/test/episode{episode_num}_step{step_id}.png"
                timestep = TimeStep(
                    step_id=step_id,
                    action=action,
                    state_v=obs,
                    state_img_path=img_path,
                    context=context,
                    robot_prediction=robot_prediction
                )
                episode.add_step(timestep)

            env.render(episode_num, step_id, screen_state)
            step_id += 1
            time.sleep(1/10)

        # when done, capture thumbnail and allow next round
        env.capture_round_thumbnail(episode_num, max(step_id-1, 0))
        print(f"Episode {episode_num} ended.")
        controlled_delay(1200)

    env.close()
    del env
    if pygame.get_init():
        pygame.quit()
    print("Environment closed successfully.")
