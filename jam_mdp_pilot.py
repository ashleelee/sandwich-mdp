"""
ConformalDAgger Study Loop
==========================
Runs N episodes × M rounds with policy retraining between rounds.
Total episodes collected: N × M.

Usage:
    python jam_mdp_study_merged.py

You will be prompted for:
    shape       — jam pattern used for base policy (square / triangle / zigzag / swirl)
    N           — episodes per round
    M           — number of rounds
    n_orig      — number of episodes to sample from the base training data for retraining
                  (e.g. 1.0 = equal amounts, 0.0 = study episodes only)

Outputs (created automatically):
    data/study/round_<r>.pkl                   — episodes collected in round r
    trained_policy/study/cont_policy_round_<r>.pth
    trained_policy/study/norm_stats_round_<r>.npz
    jam_state_img/study/round_<r>/             — per-step screenshots
"""

import math
import os
import pickle
import random
import time

import numpy as np
import pygame
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from jam_data_classes import TimeStep, Episode
from jam_mdp import (
    JamSpreadingEnv,
    controlled_delay,
    similar_state,
)
from policies.conformal.mlp import Continuous_Policy

ACTION_DIM  = 3
STATE_DIM   = 36
BUFFER_SIZE = 30
N_ENSEMBLE  = 3
UNCERTAINTY_THRESHOLD = 40
STEP_SLEEP  = 1 / 20   # seconds between steps

# ──────────────────────────────────────────────────────────────
# Policy helpers
# ──────────────────────────────────────────────────────────────

def load_policy_and_stats(policy_path, norm_stats_path):
    stats = np.load(norm_stats_path)
    min_X = stats["min_X"]
    max_X = stats["max_X"]
    min_Y = stats["min_Y"]
    max_Y = stats["max_Y"]

    range_X = max_X - min_X
    range_X[range_X == 0] = 1.0
    range_Y = max_Y - min_Y
    range_Y[range_Y == 0] = 1.0

    policy = Continuous_Policy(state_dim=STATE_DIM, output_dim=ACTION_DIM)
    policy.load_state_dict(torch.load(policy_path))
    policy.eval()

    return policy, min_X, max_X, min_Y, max_Y, range_X, range_Y


def load_ensemble(policy_paths, norm_stats_paths):
    """Load N_ENSEMBLE policies and return a list of (policy, min_X, range_X, min_Y, range_Y)."""
    members = []
    for pp, sp in zip(policy_paths, norm_stats_paths):
        policy, min_X, max_X, min_Y, max_Y, range_X, range_Y = load_policy_and_stats(pp, sp)
        members.append((policy, min_X, range_X, min_Y, range_Y))
    return members


def get_prediction(obs, obs_prev1, obs_prev2, policy, min_X, range_X, min_Y, range_Y):
    """Concatenate 3 consecutive states, normalize, run policy, unnormalize."""
    input_obs = np.concatenate(
        [np.array(obs_prev2, dtype=np.float32),
         np.array(obs_prev1, dtype=np.float32),
         np.array(obs,       dtype=np.float32)],
        axis=0,
    )
    x_norm = (input_obs - min_X) / range_X
    state_tensor = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        y_norm = policy(state_tensor).cpu().numpy()[0]
    return (y_norm * range_Y + min_Y).astype(np.float32)


def get_ensemble_prediction(obs, obs_prev1, obs_prev2, ensemble):
    """
    Run all ensemble members and return (mean_action, std_action).

    mean_action : element-wise mean across N_ENSEMBLE predictions  → used as robot_prediction
    std_action  : element-wise std  across N_ENSEMBLE predictions  → added to IQT uncertainty
    """
    preds = np.stack([
        get_prediction(obs, obs_prev1, obs_prev2, policy, min_X, range_X, min_Y, range_Y)
        for policy, min_X, range_X, min_Y, range_Y in ensemble
    ])                                  # shape: (N_ENSEMBLE, ACTION_DIM)
    return preds.mean(axis=0).astype(np.float32), preds.std(axis=0).astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Retraining
# ──────────────────────────────────────────────────────────────

DUPLICATE_THRESH = 3  # pixels; steps closer than this with same gripper are dropped

def _should_drop(prev_action, curr_action, thresh=DUPLICATE_THRESH):
    """True if curr_action is a near-duplicate of prev_action (same gripper, small xy move)."""
    if curr_action[2] != prev_action[2]:   # gripper changed — always keep
        return False
    dx = curr_action[0] - prev_action[0]
    dy = curr_action[1] - prev_action[1]
    return math.hypot(dx, dy) <= thresh


def _filter_steps(steps):
    """Remove near-duplicate consecutive steps from a step list."""
    def get_action(s):
        return np.array(s.action if hasattr(s, "action") else s["action"], dtype=np.float32)

    kept = [steps[0]]
    for s in steps[1:]:
        if not _should_drop(get_action(kept[-1]), get_action(s)):
            kept.append(s)
    return kept


def _build_dataset(episodes, human_only=False):
    """
    Convert a list of Episode objects into (X, Y) numpy arrays.

    Parameters
    ----------
    episodes   : list of Episode objects or dicts
    human_only : if True, only include steps where the human provided the action
                 (context == 'robot_asked' or 'human_intervened').
                 Use False for base training data which has no context labels.
    """
    all_X, all_Y = [], []

    for ep in episodes:
        steps = ep.steps if hasattr(ep, "steps") else ep["steps"]
        if len(steps) < 4:
            continue

        # ── #1: filter to human-labeled steps only (when requested) ──────
        if human_only:
            steps = [
                s for s in steps
                if (s.context if hasattr(s, "context") else s.get("context", ""))
                in ("robot_asked", "human_intervened")
            ]
            if len(steps) < 4:
                continue

        # ── #2: remove near-duplicate consecutive steps ───────────────────
        steps = _filter_steps(steps)
        if len(steps) < 4:
            continue

        for t in range(len(steps) - 4):
            def get_state(s):
                return np.array(
                    s.state if hasattr(s, "state") else s["state"], dtype=np.float32
                )
            def get_action(s):
                return np.array(
                    s.action if hasattr(s, "action") else s["action"], dtype=np.float32
                )
            state = np.concatenate(
                [get_state(steps[t]), get_state(steps[t + 1]), get_state(steps[t + 2])],
                axis=0,
            )
            all_X.append(state)
            all_Y.append(get_action(steps[t + 3]))

    return np.array(all_X, dtype=np.float32), np.array(all_Y, dtype=np.float32)


def retrain_policy(episode_buffer, round_idx, output_dir):
    """
    Train N_ENSEMBLE independent policies on the current episode buffer.
    All members share the same normalisation stats (same dataset).

    Returns
    -------
    policy_paths : list of N_ENSEMBLE .pth paths
    stats_path   : single .npz path (shared norm stats)
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"  [retrain] Training {N_ENSEMBLE} ensemble members on {len(episode_buffer)} buffered episodes")

    all_X, all_Y = _build_dataset(episode_buffer, human_only=False)
    print(f"  [retrain] Dataset: {all_X.shape[0]} steps total")

    # Normalise once — shared across all ensemble members
    min_X = all_X.min(axis=0);  max_X = all_X.max(axis=0)
    range_X = max_X - min_X;    range_X[range_X == 0] = 1.0
    min_Y = all_Y.min(axis=0);  max_Y = all_Y.max(axis=0)
    range_Y = max_Y - min_Y;    range_Y[range_Y == 0] = 1.0

    X_norm = (all_X - min_X) / range_X
    Y_norm = (all_Y - min_Y) / range_Y

    stats_path = os.path.join(output_dir, f"norm_stats_round_{round_idx}.npz")
    np.savez(stats_path, min_X=min_X, max_X=max_X, min_Y=min_Y, max_Y=max_Y)

    loss_fn = torch.nn.MSELoss()
    policy_paths = []

    for member_idx in range(N_ENSEMBLE):
        print(f"  [retrain] Member {member_idx + 1}/{N_ENSEMBLE}")

        # Each member gets its own shuffle → different train/val split
        idx = np.arange(len(all_X))
        np.random.shuffle(idx)
        X_m, Y_m = X_norm[idx], Y_norm[idx]

        n_train = int(0.8 * len(all_X))
        n_val   = int(0.9 * len(all_X))

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_m[:n_train]), torch.tensor(Y_m[:n_train])),
            batch_size=256, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_m[n_train:n_val]), torch.tensor(Y_m[n_train:n_val])),
            batch_size=32, shuffle=False,
        )

        policy = Continuous_Policy(state_dim=STATE_DIM, output_dim=ACTION_DIM)
        optimizer = optim.Adam(policy.parameters(), lr=1e-4)

        for epoch in range(1000):
            policy.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss_fn(policy(xb.float()), yb.float()).backward()
                optimizer.step()

            if epoch % 100 == 0:
                policy.eval()
                val_loss = sum(
                    loss_fn(policy(xb.float()), yb.float()).item()
                    for xb, yb in val_loader
                )
                print(f"    epoch {epoch:4d}  val_loss={val_loss:.4f}")

        policy_path = os.path.join(output_dir, f"cont_policy_round_{round_idx}_{member_idx + 1}.pth")
        torch.save(policy.state_dict(), policy_path)
        policy_paths.append(policy_path)
        print(f"  [retrain] Saved → {policy_path}")

    return policy_paths, stats_path


# ──────────────────────────────────────────────────────────────
# Pygame overlay screens
# ──────────────────────────────────────────────────────────────

def _draw_overlay(screen, lines, sub=None):
    """Draw a dark overlay with centered text lines."""
    screen.fill((30, 30, 30))
    font_big   = pygame.font.SysFont("Arial", 34, bold=True)
    font_small = pygame.font.SysFont("Arial", 20)
    y = 220
    for line in lines:
        surf = font_big.render(line, True, (255, 255, 255))
        screen.blit(surf, surf.get_rect(center=(450, y)))
        y += 60
    if sub:
        surf = font_small.render(sub, True, (160, 160, 160))
        screen.blit(surf, surf.get_rect(center=(450, y + 10)))
    pygame.display.flip()


def show_message_screen(screen, lines, sub=None):
    """Non-blocking: just draw and return immediately."""
    _draw_overlay(screen, lines, sub)


def wait_for_space(screen, lines, sub="Press SPACE to continue"):
    """Blocking: draw message and wait until SPACE is pressed."""
    _draw_overlay(screen, lines, sub)
    while True:
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                return
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit


# ──────────────────────────────────────────────────────────────
# Episode runner  (mirrors the main loop in jam_mdp.py)
# ──────────────────────────────────────────────────────────────

def run_round(env, N, M, round_idx, global_ep_offset, ensemble):
    """
    Run N episodes and return a list of Episode objects.

    Parameters
    ----------
    env              : JamSpreadingEnv (already initialised)
    N                : number of episodes this round
    M                : total number of rounds (used to compute total_episodes for display)
    round_idx        : 0-based round index (for logging)
    global_ep_offset : episode_num = global_ep_offset + local_ep
    ensemble         : list of (policy, min_X, range_X, min_Y, range_Y) from load_ensemble
    """
    total_episodes = N * M
    round_episodes = []

    for local_ep in range(N):
        episode_num = global_ep_offset + local_ep

        obs, _ = env.reset()
        obs_prev1 = obs.copy()
        obs_prev2 = obs.copy()
        done = False

        screen_state = "start"
        episode = Episode(episode_num)
        step_id = 0

        # IQT — reset per episode as agreed
        q_lo = np.array([1,1,1], dtype=np.float32)
        q_hi = np.array([1,1,1], dtype=np.float32)
        stepsize         = 0.2
        alpha_desired    = 0.8
        B_t_lookback     = 10
        history_upper    = []
        history_lower    = []
        uncertainty_hist = []

        action  = None
        context = None
        stuck_count = 0

        env.help_remaining_time = 0.0
        env.render(episode_num, step_id, screen_state, total_episodes)

        print(f"\n[Round {round_idx + 1}  Ep {local_ep + 1}/{N}  (global ep {episode_num})]")

        while not done:
            # ── start screen ──────────────────────────────
            if screen_state == "start":
                if env.check_button_click():
                    screen_state = "robot"
                continue

            # ── waiting for human to take over: freeze display, don't recompute ──
            if screen_state in ("ready_intervene", "ready_help"):
                if env.ready_to_help():
                    screen_state = "help"
                else:
                    env.render(episode_num, step_id, screen_state, total_episodes)
                    time.sleep(STEP_SLEEP)
                    continue

            # ── stuck detection ───────────────────────────
            if similar_state(obs, obs_prev1):
                stuck_count += 1
            else:
                stuck_count = 0

            # ── robot prediction (ensemble mean + std) ────
            mean_action, std_action = get_ensemble_prediction(
                obs, obs_prev1, obs_prev2, ensemble,
            )

            # add noise if stuck
            if stuck_count >= 13:
                sigma = 15.0
            elif stuck_count >= 10:
                sigma = 10.0
            elif stuck_count >= 7:
                sigma = 5.0
            else:
                sigma = 0.0

            if sigma > 0.0:
                noise_xy = np.random.normal(0.0, sigma, 2)
                mean_action[0] += noise_xy[0]
                mean_action[1] += noise_xy[1]

            robot_prediction = mean_action.copy()
            robot_prediction[0] = np.clip(robot_prediction[0], 0.0, 700.0)
            robot_prediction[1] = np.clip(robot_prediction[1], 0.0, 700.0)
            valid_gripper = np.array([0.0, 0.5, 1.0])
            robot_prediction[2] = valid_gripper[
                np.argmin(np.abs(valid_gripper - robot_prediction[2]))
            ]

            # ── uncertainty: ensemble std offset + IQT ────
            # y_pred = robot_prediction
            # ensemble_intv_lo = y_pred - (2 * std_action * q_lo)
            # ensemble_intv_hi = y_pred + (2 * std_action * q_hi)
            ensemble_intv_lo = (2 * std_action * q_lo)
            ensemble_intv_hi = (2 * std_action * q_hi)
            uncertainty = np.linalg.norm(ensemble_intv_lo + ensemble_intv_hi)
            uncertainty_hist.append(uncertainty)
            if len(uncertainty_hist) > 150:
                uncertainty_hist = uncertainty_hist[-150:]
            env.uncertainty_history = uncertainty_hist

            # ── state machine ─────────────────────────────
            human_action = None  # set below if human is controlling

            if screen_state == "robot":
                need_help = uncertainty > UNCERTAINTY_THRESHOLD
                if env.check_button_click():
                    screen_state = "ready_intervene"
                    context = "human_intervened"
                elif need_help:
                    screen_state = "ready_help"
                    context = "robot_asked"
                else:
                    action  = robot_prediction
                    context = "robot_independent"

            elif screen_state == "help":
                # check if human hands back control
                exit_human = False
                if pygame.event.peek(pygame.KEYDOWN):
                    for event in pygame.event.get(pygame.KEYDOWN):
                        if event.key == pygame.K_SPACE:
                            exit_human = True
                if exit_human:
                    screen_state = "robot"
                    env.prediction_overlay = None
                    continue

                # get expert action and update IQT
                action       = env.get_help()
                human_action = action
                expert_y     = action
                y_pred    = robot_prediction

                ensemble_intv_lo = y_pred - (2 * std_action * q_lo)
                ensemble_intv_hi = y_pred + (2 * std_action * q_hi)
                print("-------------------------------")
                print("y pred", y_pred)
                print("ensemble pred lo and hi", (ensemble_intv_lo, ensemble_intv_hi))
                print("q_lo, q_hi", q_lo, q_hi)

                env.prediction_overlay = {
                    "y_pred":   y_pred,
                    "intv_lo":  ensemble_intv_lo,
                    "intv_hi":  ensemble_intv_hi,
                }

                bool_miscoverage_lo = expert_y < ensemble_intv_lo
                bool_miscoverage_hi = expert_y > ensemble_intv_hi

                # import pdb; pdb.set_trace()

                # check if there is a miscoverage in any of the 3 dim
                err_lo = np.any(bool_miscoverage_lo)
                err_hi = np.any(bool_miscoverage_hi)
                print("err_lo, err_hi:", err_lo, err_hi)

                resid_lo = (y_pred - expert_y)/(2*std_action)
                resid_hi = (expert_y - y_pred)/(2*std_action)
                
                # replace negatives in resid_lo and reside_hi with 1.0
                resid_lo[resid_lo < 0] = 1.0
                resid_hi[resid_hi < 0] = 1.0

                B_hi = np.ones(ACTION_DIM) * 0.01
                B_lo = np.ones(ACTION_DIM) * 0.01

                if len(history_upper) > 0:
                    B_hi = np.max(history_upper, axis=0)
                    B_lo = np.max(history_lower, axis=0)

                history_upper.append(resid_hi)
                history_lower.append(resid_lo)
                if len(history_upper) > B_t_lookback:
                    history_upper.pop(0)
                    history_lower.pop(0)

                # IQT update 
                q_hi = q_hi + stepsize * B_hi * (err_hi - alpha_desired)
                q_lo = q_lo + stepsize * B_lo * (err_lo - alpha_desired)

                # IQT update (clamp to small positive floor)
                # q_hi = np.maximum(q_hi + stepsize * B_hi * (err_hi - alpha_desired), 0.01)
                # q_lo = np.maximum(q_lo + stepsize * B_lo * (err_lo - alpha_desired), 0.01)

            # ── step environment ──────────────────────────
            # ── log ───────────────────────────────────────
            log = (
                f"  step={step_id:4d}  unc={uncertainty:.3f}  "
                f"pred={[f'{x:.2f}' for x in robot_prediction]}"
            )
            if human_action is not None:
                log += f"  human={[f'{x:.2f}' for x in human_action]}"
            print(log)

            obs_prev2 = obs_prev1.copy()
            obs_prev1 = obs.copy()

            if action is not None:
                obs, _, done, _, _ = env.step(action)
                img_path = (
                    f"jam_state_img/{env.img_save_dir}/"
                    f"episode{episode_num}_step{step_id}.png"
                )
                episode.add_step(TimeStep(
                    step_id=step_id,
                    action=action,
                    state=obs.copy(),
                    state_img_path=img_path,
                    context=context,
                    robot_prediction=robot_prediction,
                    q_lo=q_lo.copy(),
                    q_hi=q_hi.copy(),
                    std_action=std_action.copy(),
                    intv_lo=(robot_prediction - 2 * std_action * q_lo).copy(),
                    intv_hi=(robot_prediction + 2 * std_action * q_hi).copy(),
                ))
                if done:
                    screen_state = "done"
                env.render(episode_num, step_id, screen_state, total_episodes)
                step_id += 1
                time.sleep(STEP_SLEEP)

        print(f"  Episode {episode_num} complete ({step_id} steps).")
        round_episodes.append(episode)
        controlled_delay(3000)   # 3-second pause between episodes

    return round_episodes


# ──────────────────────────────────────────────────────────────
# Expert reference collection  (run once before round 0)
# ──────────────────────────────────────────────────────────────

def collect_expert_episodes(env, n_episodes, save_path):
    """
    Collect n_episodes of pure human-controlled demonstrations.
    Used as reference trajectories for Decision deviation and Trajectory deviation.

    The human controls the robot for every step (no policy involvement).
    Episodes are saved to save_path and returned as a list of Episode objects.
    """
    print(f"\n{'='*55}")
    print(f"  EXPERT REFERENCE COLLECTION  —  {n_episodes} episodes")
    print(f"{'='*55}")

    img_dir = "jam_state_img/study/expert_reference"
    os.makedirs(img_dir, exist_ok=True)
    env.img_save_dir = "study/expert_reference"

    expert_episodes = []

    for ep_idx in range(n_episodes):
        wait_for_space(
            env.screen,
            [f"Expert Demo {ep_idx + 1} of {n_episodes}", "You control the robot."],
            sub="Press SPACE to start",
        )

        obs, _ = env.reset()
        done = False
        screen_state = "start"
        episode = Episode(ep_idx)
        step_id = 0

        env.help_remaining_time = 0.0
        env.render(ep_idx, step_id, screen_state, n_episodes)
        print(f"\n[Expert ep {ep_idx + 1}/{n_episodes}]")

        while not done:
            if screen_state == "start":
                if env.check_button_click():
                    screen_state = "positioning"
                continue

            # ── positioning: user moves to start position, then presses SPACE ──
            if screen_state == "positioning":
                env.render(ep_idx, step_id, screen_state, n_episodes)
                # _draw_overlay(
                #     env.screen,
                #     ["Move to your start position"],
                #     sub="Press SPACE when ready to begin recording",
                # )
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        screen_state = "help"
                    elif event.type == pygame.QUIT:
                        pygame.quit()
                        raise SystemExit
                continue

            action = env.get_help()
            if action is None:
                env.render(ep_idx, step_id, screen_state, n_episodes)
                time.sleep(STEP_SLEEP)
                continue

            obs, _, done, _, _ = env.step(action)
            img_path = (
                f"jam_state_img/study/expert_reference/"
                f"episode{ep_idx}_step{step_id}.png"
            )
            episode.add_step(TimeStep(
                step_id=step_id,
                action=action,
                state=obs.copy(),
                state_img_path=img_path,
                context="expert",
                robot_prediction=None,
                std_action=None,
                intv_lo=None,
                intv_hi=None,
            ))

            if done:
                screen_state = "done"
            env.render(ep_idx, step_id, screen_state, n_episodes)
            step_id += 1
            time.sleep(STEP_SLEEP)

        print(f"  Expert episode {ep_idx + 1} complete ({step_id} steps).")
        expert_episodes.append(episode)
        controlled_delay(3000)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(expert_episodes, f)
    print(f"\n  Saved expert reference → {save_path}")
    return expert_episodes


# ──────────────────────────────────────────────────────────────
# Final rollout
# ──────────────────────────────────────────────────────────────

def run_rollout(env, n_rollout, ensemble):
    """
    Run n_rollout episodes autonomously with the final policy.
    No human interventions, no IQT help requests — robot acts every step.
    IQT bounds are still tracked and shown on the uncertainty graph.
    """
    print(f"\n{'='*55}")
    print(f"  FINAL ROLLOUT  —  {n_rollout} episodes  (robot only)")
    print(f"{'='*55}")

    total_rollout = n_rollout

    for ep in range(n_rollout):
        obs, _ = env.reset()
        obs_prev1 = obs.copy()
        obs_prev2 = obs.copy()
        done = False

        screen_state = "start"
        step_id = 0
        stuck_count = 0

        # IQT reset (display only — no updates since no expert feedback)
        q_lo = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        q_hi = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        uncertainty_hist = []

        img_dir = f"jam_state_img/study/rollout"
        os.makedirs(img_dir, exist_ok=True)
        env.img_save_dir = "study/rollout"

        env.help_remaining_time = 0.0
        env.render(ep, step_id, screen_state, total_rollout)

        print(f"\n[Rollout ep {ep + 1}/{n_rollout}]")

        while not done:
            if screen_state == "start":
                if env.check_button_click():
                    screen_state = "robot"
                continue

            # stuck detection
            if similar_state(obs, obs_prev1):
                stuck_count += 1
            else:
                stuck_count = 0

            # robot prediction (ensemble mean; std not used in rollout)
            action, _ = get_ensemble_prediction(obs, obs_prev1, obs_prev2, ensemble)

            if stuck_count >= 13:
                sigma = 15.0
            elif stuck_count >= 10:
                sigma = 10.0
            elif stuck_count >= 7:
                sigma = 5.0
            else:
                sigma = 0.0
            if sigma > 0.0:
                noise_xy = np.random.normal(0.0, sigma, 2)
                action[0] += noise_xy[0]
                action[1] += noise_xy[1]

            action[0] = np.clip(action[0], 0.0, 700.0)
            action[1] = np.clip(action[1], 0.0, 700.0)
            valid_gripper = np.array([0.0, 0.5, 1.0])
            action[2] = valid_gripper[np.argmin(np.abs(valid_gripper - action[2]))]

            # uncertainty display (IQT not updated — no expert labels)
            uncertainty = np.linalg.norm(q_hi + q_lo)
            uncertainty_hist.append(uncertainty)
            if len(uncertainty_hist) > 150:
                uncertainty_hist = uncertainty_hist[-150:]
            env.uncertainty_history = uncertainty_hist

            # always robot-controlled; pump events so window stays responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit

            obs_prev2 = obs_prev1.copy()
            obs_prev1 = obs.copy()

            obs, _, done, _, _ = env.step(action)
            if done:
                screen_state = "done"
            env.render(ep, step_id, screen_state, total_rollout)
            step_id += 1
            time.sleep(STEP_SLEEP)

        print(f"  Rollout episode {ep + 1} complete ({step_id} steps).")
        controlled_delay(3000)


# ──────────────────────────────────────────────────────────────
# Main study loop
# ──────────────────────────────────────────────────────────────

def run_study():
    # ── user inputs ───────────────────────────────
    shape_map  = {"1": "square", "2": "triangle", "3": "zigzag", "4": "swirl"}
    shape        = shape_map[input("Shape  1=square  2=triangle  3=zigzag  4=swirl: ").strip()]
    N            = int(input("Episodes per round (N): "))
    M            = int(input("Number of rounds (M): "))
    N_EXPERT     = int(input("Expert reference episodes to collect before study (N_expert): "))
    show_rollout = input("Show final rollout after study? (y/n): ").strip().lower() == "y"

    # ── paths ─────────────────────────────────────
    base_data_file  = f"data/jam_train_data_all_{shape}.pkl"
    study_model_dir = "trained_policy/study"
    # initial ensemble: N_ENSEMBLE pre-trained models (each with its own norm stats)
    init_policy_paths = [
        f"trained_policy/ensemble/cont_policy_{shape}20hz_{i+1}.pth"
        for i in range(N_ENSEMBLE)
    ]
    init_stats_paths = [
        f"trained_policy/ensemble/norm_stats_{shape}20hz_{i+1}.npz"
        for i in range(N_ENSEMBLE)
    ]

    os.makedirs("data/study",    exist_ok=True)
    os.makedirs(study_model_dir, exist_ok=True)

    print(f"\nStudy config: shape={shape}  N={N}  M={M}  buffer_size={BUFFER_SIZE}")
    print(f"Total episodes: {N * M}  ({M} rounds × {N} episodes)\n")

    # ── pygame + env ──────────────────────────────
    pygame.init()
    pygame.display.init()
    env = JamSpreadingEnv()

    # ── initialize buffer with randomly sampled base episodes ────
    base_data_file = f"data/jam_train_data_all_{shape}_20hz.pkl"
    with open(base_data_file, "rb") as f:
        base_episodes = pickle.load(f)
    rng = random.Random(42)
    idxs = list(range(len(base_episodes)))
    rng.shuffle(idxs)
    episode_buffer = [base_episodes[i] for i in idxs[:BUFFER_SIZE]]
    print(f"  Buffer initialized with {len(episode_buffer)} base episodes (BUFFER_SIZE={BUFFER_SIZE})")

    # load initial ensemble
    ensemble = load_ensemble(init_policy_paths, init_stats_paths)
    print(f"  Loaded initial ensemble ({N_ENSEMBLE} members)")

    # ── collect expert reference episodes (once, before any rounds) ────
    expert_ref_path = "data/study/expert_reference.pkl"
    expert_episodes = collect_expert_episodes(env, N_EXPERT, expert_ref_path)

    for round_idx in range(M):
        # per-round image directory
        img_dir = f"jam_state_img/study/round_{round_idx}"
        os.makedirs(img_dir, exist_ok=True)
        env.img_save_dir = f"study/round_{round_idx}"

        print(f"\n{'='*55}")
        print(f"  ROUND {round_idx + 1} / {M}  —  {N} episodes  ({N_ENSEMBLE}-member ensemble)")
        print(f"{'='*55}")

        # wait for experimenter to press SPACE before each round
        wait_for_space(
            env.screen,
            [f"Round {round_idx + 1} of {M}", f"{N} episodes this round"],
        )

        # run N episodes
        round_episodes = run_round(
            env, N, M, round_idx,
            global_ep_offset=round_idx * N,
            ensemble=ensemble,
        )
        # save this round's raw data
        round_data_path = f"data/study/round_{round_idx}.pkl"
        with open(round_data_path, "wb") as f:
            pickle.dump(round_episodes, f)
        print(f"\n  Saved round data → {round_data_path}")

        # add new episodes to end of buffer, push oldest out
        episode_buffer.extend(round_episodes)
        episode_buffer = episode_buffer[-BUFFER_SIZE:]
        print(f"  Buffer updated: {len(episode_buffer)} episodes (kept newest {BUFFER_SIZE})")

        # retrain after every round (always, so rollout uses the latest policy)
        show_message_screen(
            env.screen,
            [f"Round {round_idx + 1} complete", "Retraining policy…"],
            sub="Please wait — this may take a few minutes",
        )
        print(f"\n  Retraining after round {round_idx + 1}…")
        new_policy_paths, new_stats_path = retrain_policy(
            episode_buffer=episode_buffer,
            round_idx=round_idx,
            output_dir=study_model_dir,
        )
        # all retrained members share the same norm stats
        ensemble = load_ensemble(new_policy_paths, [new_stats_path] * N_ENSEMBLE)
        pygame.event.clear()   # flush events queued during retraining (no pump was called)

    # ── study complete ────────────────────────────
    print(f"\nStudy complete: {N * M} total episodes across {M} rounds.")

    if show_rollout:
        wait_for_space(
            env.screen,
            ["Study complete!", "Final rollout coming up…"],
            sub="Press SPACE to begin rollout",
        )
        run_rollout(env, 1, ensemble=ensemble)

    wait_for_space(
        env.screen,
        ["All done!", f"{N * M} study episodes  ·  {M} rounds"],
        sub="Press SPACE to exit",
    )

    env.close()
    if pygame.get_init():
        pygame.quit()


if __name__ == "__main__":
    run_study()
