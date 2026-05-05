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
from policies.conformal.mlp import Continuous_Policy, SafetyClassifier

ACTION_DIM  = 3
STATE_DIM   = 36
BUFFER_SIZE = 30
N_ENSEMBLE  = 3
UNCERTAINTY_THRESHOLD_ENSEMBLE  = 6.73
UNCERTAINTY_THRESHOLD_CONFORMAL = 56.4255
UNCERTAINTY_THRESHOLD_CONFORMAL_ENSEMBLE = 6.73 
UNCERTAINTY_THRESHOLD_ENSEMBLE_CLASSIFIER = 1.0   # normalized composite: >1.0 → need help
STEP_SLEEP  = 1 / 20   # seconds between steps

# DAgger method identifiers
METHOD_CONFORMAL_ENSEMBLE    = "conformal++"   # ensemble mean/std + IQT-scaled interval (current)
METHOD_ENSEMBLE              = "ensemble"      # ensemble disagreement only, no IQT
METHOD_CONFORMAL             = "conformal"     # single policy + IQT absolute interval
METHOD_LAZYDAGGER            = "lazydagger"    # single policy + binary safety classifier
METHOD_ENSEMBLE_CLASSIFIER   = "ensemble_clf"  # N_ENSEMBLE policies + safety classifier; auto exit

# LazyDAgger thresholds
LAZYDAGGER_BETA_H      = 56.4255 # action-loss boundary for classifier labels (train: loss>β_H → unsafe)
LAZYDAGGER_BETA_R      = LAZYDAGGER_BETA_H * 1/10   # true loss below which supervisor releases control
LAZYDAGGER_NOISE_SIGMA = 3.0    # σ of Gaussian noise added to supervisor actions

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
# Per-method prediction + interval  (returns robot_pred, std_action,
#                                    uncertainty, intv_lo, intv_hi)
# ──────────────────────────────────────────────────────────────

def predict_conformal_ensemble(obs, obs_prev1, obs_prev2, model, q_lo, q_hi):
    """
    conformal++: ensemble mean ± 2*std*q  (IQT-calibrated).
    model : list of N_ENSEMBLE members from load_ensemble()
    """
    mean_action, std_action = get_ensemble_prediction(obs, obs_prev1, obs_prev2, model)
    intv_lo    = mean_action - 2 * std_action * q_lo
    intv_hi    = mean_action + 2 * std_action * q_hi
    uncertainty = np.linalg.norm(2 * std_action * q_lo + 2 * std_action * q_hi)
    return mean_action, std_action, uncertainty, intv_lo, intv_hi


def predict_ensemble(obs, obs_prev1, obs_prev2, model, q_lo, q_hi):
    """
    EnsembleDAgger: fixed ±3σ interval, no IQT.
    Uncertainty = ensemble std norm (disagreement).
    q_lo / q_hi are ignored but kept for a uniform signature.
    model : list of N_ENSEMBLE members from load_ensemble()
    """
    mean_action, std_action = get_ensemble_prediction(obs, obs_prev1, obs_prev2, model)
    n_std       = 3.0
    intv_lo     = mean_action - n_std * std_action
    intv_hi     = mean_action + n_std * std_action
    uncertainty = np.linalg.norm(std_action)
    return mean_action, std_action, uncertainty, intv_lo, intv_hi


def predict_conformal(obs, obs_prev1, obs_prev2, model, q_lo, q_hi):
    """
    Pure ConformalDAgger: single policy, interval = pred ± q  (IQT-calibrated).
    q_lo / q_hi are absolute half-widths (not std-scaled).
    Uncertainty = width of the interval.
    model : 1-member list from load_ensemble() (only index 0 is used)
    """
    mean_action, std_action = get_ensemble_prediction(obs, obs_prev1, obs_prev2, model)
    # std_action is 0 for a 1-member ensemble — interval comes from q alone
    intv_lo     = mean_action - q_lo
    intv_hi     = mean_action + q_hi
    uncertainty = np.linalg.norm(q_lo + q_hi)
    return mean_action, std_action, uncertainty, intv_lo, intv_hi


def predict_lazydagger(obs, obs_prev1, obs_prev2, model, q_lo, q_hi):
    """
    LazyDAgger: single policy + binary safety classifier.
    model : (policy_list_1member, (clf_net, min_X, range_X))
    uncertainty = P(need_help | state) in [0, 1]; threshold in METHOD_CONFIG = 0.5
    q_lo / q_hi are unused but kept for uniform signature.
    """
    policy_list, (clf_net, clf_min_X, clf_range_X) = model
    mean_action, std_action = get_ensemble_prediction(obs, obs_prev1, obs_prev2, policy_list)

    input_obs = np.concatenate(
        [np.array(obs_prev2, dtype=np.float32),
         np.array(obs_prev1, dtype=np.float32),
         np.array(obs,       dtype=np.float32)],
        axis=0,
    )
    x_norm = (input_obs - clf_min_X) / clf_range_X
    with torch.no_grad():
        probs = clf_net(torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0))
        prob_unsafe = probs[0, 1].item()  # index 1 = P(unsafe)

    uncertainty = prob_unsafe
    print("prob_unsafe:", prob_unsafe )
    return mean_action, std_action, uncertainty, None, None


def predict_ensemble_clf(obs, obs_prev1, obs_prev2, model, q_lo, q_hi):
    """
    EnsembleDAgger + SafetyClassifier:
    - Ensemble mean/std for action; classifier for unsafe detection.
    - uncertainty = max(ensemble_std_norm / ENSEMBLE_THRESHOLD, prob_unsafe / 0.5)
      → > 1.0 triggers human help (either signal exceeds its own threshold).
      → < 1.0 in help mode automatically returns control to the robot.
    model : (policy_list, (clf_net, clf_min_X, clf_range_X))
    q_lo / q_hi are unused but kept for uniform signature.
    """
    policy_list, (clf_net, clf_min_X, clf_range_X) = model
    mean_action, std_action = get_ensemble_prediction(obs, obs_prev1, obs_prev2, policy_list)

    ensemble_unc = np.linalg.norm(std_action)

    input_obs = np.concatenate(
        [np.array(obs_prev2, dtype=np.float32),
         np.array(obs_prev1, dtype=np.float32),
         np.array(obs,       dtype=np.float32)],
        axis=0,
    )
    x_norm = (input_obs - clf_min_X) / clf_range_X
    with torch.no_grad():
        probs = clf_net(torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0))
        prob_unsafe = probs[0, 1].item()

    # normalized composite: > 1.0 iff either signal exceeds its threshold
    uncertainty = max(ensemble_unc / UNCERTAINTY_THRESHOLD_ENSEMBLE, prob_unsafe / 0.5)

    n_std = 3.0
    intv_lo = mean_action - n_std * std_action
    intv_hi = mean_action + n_std * std_action

    return mean_action, std_action, uncertainty, intv_lo, intv_hi


# ──────────────────────────────────────────────────────────────
# Per-method IQT residual computation  (called only in help state)
# Returns (resid_lo, resid_hi) with negatives replaced by 1.0
# ──────────────────────────────────────────────────────────────

def iqt_residuals_scaled(expert_y, y_pred, std_action):
    """
    conformal++: residuals normalised by 2*std so q stays dimensionless.
    Matches the existing formulation.
    """
    denom = 2 * std_action
    denom[denom == 0] = 1.0          # avoid div-by-zero on degenerate dims
    resid_lo = (y_pred - expert_y) / denom
    resid_hi = (expert_y - y_pred) / denom
    resid_lo[resid_lo < 0] = 1.0
    resid_hi[resid_hi < 0] = 1.0
    return resid_lo, resid_hi


def iqt_residuals_absolute(expert_y, y_pred, std_action):
    """
    Pure conformal: raw absolute residuals (std_action is 0, nothing to scale by).
    q accumulates in the same units as the action.
    """
    resid_lo = y_pred - expert_y
    resid_hi = expert_y - y_pred
    resid_lo[resid_lo < 0] = 1.0
    resid_hi[resid_hi < 0] = 1.0
    return resid_lo, resid_hi


# Map method → (predict_fn, residual_fn, uses_iqt, unc_threshold)
# uses_iqt=False  → q_lo/q_hi are never updated
# unc_threshold   → need_help = uncertainty > unc_threshold
METHOD_CONFIG = {
    METHOD_CONFORMAL_ENSEMBLE:  (predict_conformal_ensemble, iqt_residuals_scaled,    True,  UNCERTAINTY_THRESHOLD_CONFORMAL_ENSEMBLE),
    METHOD_ENSEMBLE:            (predict_ensemble,           None,                    False, UNCERTAINTY_THRESHOLD_ENSEMBLE),
    METHOD_CONFORMAL:           (predict_conformal,          iqt_residuals_absolute,  True,  UNCERTAINTY_THRESHOLD_CONFORMAL),
    METHOD_LAZYDAGGER:          (predict_lazydagger,         None,                    False, 0.5),
    METHOD_ENSEMBLE_CLASSIFIER: (predict_ensemble_clf,       None,                    False, UNCERTAINTY_THRESHOLD_ENSEMBLE_CLASSIFIER),
}


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


def retrain_policy(episode_buffer, round_idx, output_dir, n_members=N_ENSEMBLE):
    """
    Train n_members independent policies on the current episode buffer.
    All members share the same normalisation stats (same dataset).

    Returns
    -------
    policy_paths : list of n_members .pth paths
    stats_path   : single .npz path (shared norm stats)
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"  [retrain] Training {n_members} member(s) on {len(episode_buffer)} buffered episodes")

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

    for member_idx in range(n_members):
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
# LazyDAgger classifier helpers
# ──────────────────────────────────────────────────────────────

def train_lazydagger_classifier(episode_buffer, output_dir, round_idx, stats_path):
    """
    Retrain SafetyClassifier by combining:
      - all original labeled data from data/train_safety/classifier/safety_dataset.pkl
      - new steps from episode_buffer where the human intervened (robot_asked / human_intervened)
    Label convention: 0=safe, 1=unsafe  (||robot_pred - action|| > LAZYDAGGER_BETA_H → unsafe)
    Returns clf_path.
    """
    stats   = np.load(stats_path)
    min_X   = stats["min_X"]
    max_X   = stats["max_X"]
    range_X = max_X - min_X
    range_X[range_X == 0] = 1.0

    # ── original safety dataset (raw, unnormalized) ────────────
    orig_path = "data/train_safety/classifier/safety_dataset.pkl"
    with open(orig_path, "rb") as f:
        orig = pickle.load(f)
    all_X      = list(orig["X"])       # list of (36,) arrays, unnormalized
    all_labels = list(orig["labels"])  # 0=safe, 1=unsafe
    print(f"  [lazydagger_clf] Loaded {len(all_X)} original samples from {orig_path}")

    # ── new human-intervention steps from episode buffer ───────
    n_new = 0
    for ep in episode_buffer:
        steps = ep.steps if hasattr(ep, "steps") else ep["steps"]
        if len(steps) < 4:
            continue
        for t in range(len(steps) - 3):
            def _s(s):  return np.array(s.state  if hasattr(s, "state")  else s["state"],  dtype=np.float32)
            def _a(s):  return np.array(s.action if hasattr(s, "action") else s["action"], dtype=np.float32)
            def _rp(s):
                rp = s.robot_prediction if hasattr(s, "robot_prediction") else s.get("robot_prediction")
                return np.array(rp, dtype=np.float32) if rp is not None else None
            def _ctx(s): return s.context if hasattr(s, "context") else s.get("context", "")

            if _ctx(steps[t + 3]) not in ("robot_asked", "human_intervened"):
                continue
            robot_pred = _rp(steps[t + 3])
            if robot_pred is None:
                continue

            state_input = np.concatenate([_s(steps[t]), _s(steps[t + 1]), _s(steps[t + 2])], axis=0)
            loss        = np.linalg.norm(robot_pred - _a(steps[t + 3]))
            all_X.append(state_input)
            all_labels.append(1 if loss > LAZYDAGGER_BETA_H else 0)
            n_new += 1
    print(f"  [lazydagger_clf] Added {n_new} new human-intervention samples")

    X = np.array(all_X,      dtype=np.float32)
    Y = np.array(all_labels, dtype=np.int64)

    # normalize all data with current round's stats
    X_norm   = (X - min_X) / range_X
    n_unsafe = int((Y == 1).sum())
    n_safe   = int((Y == 0).sum())
    print(f"  [lazydagger_clf] Total: {len(X)} samples  (unsafe={n_unsafe}  safe={n_safe})")

    clf_path = os.path.join(output_dir, f"lazydagger_clf_round_{round_idx}.pth")
    clf      = SafetyClassifier(input_dim=STATE_DIM)

    loader    = DataLoader(TensorDataset(torch.tensor(X_norm), torch.tensor(Y)), batch_size=256, shuffle=True)
    optimizer = optim.Adam(clf.parameters(), lr=1e-4)
    ce        = torch.nn.CrossEntropyLoss()

    for epoch in range(1000):
        clf.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            ce(clf(xb.float()), yb).backward()
            optimizer.step()
        if epoch % 50 == 0:
            clf.eval()
            with torch.no_grad():
                preds = clf(torch.tensor(X_norm)).argmax(dim=1).numpy()
            acc = (preds == Y).mean()
            print(f"    epoch {epoch:3d}  acc={acc:.3f}")

    torch.save(clf.state_dict(), clf_path)
    print(f"  [lazydagger_clf] Saved → {clf_path}")
    return clf_path


def load_lazydagger_classifier(clf_path, stats_path):
    """Load BinaryClassifier; returns (clf_net, min_X, range_X) tuple."""
    stats   = np.load(stats_path)
    min_X   = stats["min_X"]
    max_X   = stats["max_X"]
    range_X = max_X - min_X
    range_X[range_X == 0] = 1.0

    clf = SafetyClassifier(input_dim=STATE_DIM)
    clf.load_state_dict(torch.load(clf_path))
    clf.eval()
    return (clf, min_X, range_X)


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


def wait_for_yes_no(screen, lines):
    """Draw overlay with YES / NO buttons. Returns True for YES, False for NO."""
    font_big = pygame.font.SysFont("Arial", 34, bold=True)
    font_btn = pygame.font.SysFont("Arial", 26, bold=True)
    yes_rect = pygame.Rect(225, 360, 150, 52)
    no_rect  = pygame.Rect(525, 360, 150, 52)

    while True:
        screen.fill((30, 30, 30))
        y = 180
        for line in lines:
            surf = font_big.render(line, True, (255, 255, 255))
            screen.blit(surf, surf.get_rect(center=(450, y)))
            y += 60
        for label, color, rect in [
            ("YES", (60, 150, 60), yes_rect),
            ("NO",  (150, 60, 60), no_rect),
        ]:
            pygame.draw.rect(screen, color, rect, border_radius=8)
            surf = font_btn.render(label, True, (255, 255, 255))
            screen.blit(surf, surf.get_rect(center=rect.center))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if yes_rect.collidepoint(event.pos):
                    return True
                if no_rect.collidepoint(event.pos):
                    return False
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit


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

def run_round(env, N, M, round_idx, global_ep_offset, ensemble, method=METHOD_CONFORMAL_ENSEMBLE):
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
    method           : one of METHOD_CONFORMAL_ENSEMBLE / METHOD_ENSEMBLE / METHOD_CONFORMAL
    """
    predict_fn, residual_fn, uses_iqt, unc_threshold = METHOD_CONFIG[method]
    env.unc_threshold  = unc_threshold
    env.can_intervene  = method not in (METHOD_ENSEMBLE, METHOD_LAZYDAGGER, METHOD_ENSEMBLE_CLASSIFIER)
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

            # ── robot prediction + interval (method-specific) ────
            mean_action, std_action, uncertainty, intv_lo, intv_hi = predict_fn(
                obs, obs_prev1, obs_prev2, ensemble, q_lo, q_hi,
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

            uncertainty_hist.append(uncertainty)
            if len(uncertainty_hist) > 150:
                uncertainty_hist = uncertainty_hist[-150:]
            env.uncertainty_history = uncertainty_hist

            # ── state machine ─────────────────────────────
            human_action = None  # set below if human is controlling

            if screen_state == "robot":
                need_help = uncertainty > unc_threshold
                can_intervene = method not in (METHOD_ENSEMBLE, METHOD_LAZYDAGGER, METHOD_ENSEMBLE_CLASSIFIER)
                if can_intervene and env.check_button_click():
                    screen_state = "ready_intervene"
                    context = "human_intervened"
                elif need_help:
                    screen_state = "ready_help"
                    context = "robot_asked"
                else:
                    action  = robot_prediction
                    context = "robot_independent"

            elif screen_state == "help":
                if method == METHOD_LAZYDAGGER:
                    # Exit when true action loss drops below β_R; noisy supervisor execution
                    action       = env.get_help()
                    human_action = action

                    if action is not None:
                        true_loss = float(np.linalg.norm(robot_prediction - action))
                        if true_loss < LAZYDAGGER_BETA_R:
                            screen_state = "robot"
                            env.prediction_overlay = None
                        noise_xy  = np.random.normal(0.0, LAZYDAGGER_NOISE_SIGMA, 2)
                        action    = action.copy()
                        action[0] = np.clip(action[0] + noise_xy[0], 0.0, 700.0)
                        action[1] = np.clip(action[1] + noise_xy[1], 0.0, 700.0)

                elif method == METHOD_ENSEMBLE_CLASSIFIER:
                    # In human mode: exit when BOTH
                    #   (1) ‖robot_pred − human‖ < 56.4255  (action distance safe)
                    #   (2) ensemble std norm < 6.73          (ensemble not too uncertain)
                    action       = env.get_help()
                    human_action = action

                    if action is not None:
                        dist         = float(np.linalg.norm(robot_prediction - action))
                        ensemble_unc = np.linalg.norm(std_action)
                        if dist < LAZYDAGGER_BETA_H and ensemble_unc < UNCERTAINTY_THRESHOLD_ENSEMBLE:
                            screen_state = "robot"
                            env.prediction_overlay = None
                
                elif method == METHOD_ENSEMBLE:
                    # In human mode: exit when BOTH
                    #   (1) ‖robot_pred − human‖ < 56.4255  (action distance safe)
                    #   (2) ensemble std norm < 6.73          (ensemble not too uncertain)
                    action       = env.get_help()
                    human_action = action

                    if action is not None:
                        ensemble_unc = np.linalg.norm(std_action)
                        if ensemble_unc < UNCERTAINTY_THRESHOLD_ENSEMBLE:
                            screen_state = "robot"
                            env.prediction_overlay = None

                else:
                    # SPACE key exits help mode
                    exit_human = False
                    if pygame.event.peek(pygame.KEYDOWN):
                        for event in pygame.event.get(pygame.KEYDOWN):
                            if event.key == pygame.K_SPACE:
                                exit_human = True
                    if exit_human:
                        screen_state = "robot"
                        env.prediction_overlay = None
                        continue

                    action       = env.get_help()
                    human_action = action
                    expert_y     = action
                    y_pred       = robot_prediction

                    env.prediction_overlay = {
                        "y_pred":  y_pred,
                        "intv_lo": intv_lo,
                        "intv_hi": intv_hi,
                    }

                    err_lo = np.any(expert_y < intv_lo)
                    err_hi = np.any(expert_y > intv_hi)
                    print(f"  [IQT] err_lo={err_lo}  err_hi={err_hi}  q_lo={q_lo}  q_hi={q_hi}")

                    if uses_iqt:
                        resid_lo, resid_hi = residual_fn(expert_y, y_pred, std_action)

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

                        q_hi = q_hi + stepsize * B_hi * (err_hi - alpha_desired)
                        q_lo = q_lo + stepsize * B_lo * (err_lo - alpha_desired)

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
                    intv_lo=intv_lo.copy() if intv_lo is not None else None,
                    intv_hi=intv_hi.copy() if intv_hi is not None else None,
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

def run_rollout(env, n_rollout, ensemble, method=None):
    """
    Run n_rollout episodes autonomously with the final policy.
    No human interventions, no IQT help requests — robot acts every step.
    """
    print(f"\n{'='*55}")
    print(f"  FINAL ROLLOUT  —  {n_rollout} episodes  (robot only)")
    print(f"{'='*55}")

    # classifier-based methods wrap the policy list in a tuple
    policy_list = ensemble[0] if isinstance(ensemble, tuple) else ensemble
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
            action, _ = get_ensemble_prediction(obs, obs_prev1, obs_prev2, policy_list)

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
# Human evaluation stage
# ──────────────────────────────────────────────────────────────

EVAL_STOP_INTERVAL = 40   # steps between forced stops in comparison rollout


def _eval_policy_list(ensemble, method):
    """Extract policy list from ensemble (handles classifier-based methods)."""
    return ensemble[0] if method in (METHOD_LAZYDAGGER, METHOD_ENSEMBLE_CLASSIFIER) else ensemble


def run_rating_rollout(env, n_episodes, ensemble, method):
    """
    Autonomous robot rollout; human rates each episode Yes/No after it finishes.
    Returns list of (episode_idx, True/False) tuples.
    """
    policy_list = _eval_policy_list(ensemble, method)
    ratings = []
    env.can_intervene = False

    for ep in range(n_episodes):
        wait_for_space(
            env.screen,
            [f"Rating  {ep + 1} / {n_episodes}", "Watch the robot. Rate when it finishes."],
            sub="Press SPACE to start",
        )

        obs, _ = env.reset()
        obs_prev1 = obs.copy()
        obs_prev2 = obs.copy()
        done = False
        screen_state = "start"
        step_id = 0
        stuck_count = 0
        uncertainty_hist = []

        env.help_remaining_time = 0.0
        env.render(ep, step_id, screen_state, n_episodes)
        print(f"\n[Eval rating ep {ep + 1}/{n_episodes}]")

        while not done:
            if screen_state == "start":
                if env.check_button_click():
                    screen_state = "robot"
                continue

            if similar_state(obs, obs_prev1):
                stuck_count += 1
            else:
                stuck_count = 0

            action, _ = get_ensemble_prediction(obs, obs_prev1, obs_prev2, policy_list)

            if stuck_count >= 13:   sigma = 15.0
            elif stuck_count >= 10: sigma = 10.0
            elif stuck_count >= 7:  sigma = 5.0
            else:                   sigma = 0.0
            if sigma > 0.0:
                noise = np.random.normal(0.0, sigma, 2)
                action[0] += noise[0]; action[1] += noise[1]

            action[0] = np.clip(action[0], 0.0, 700.0)
            action[1] = np.clip(action[1], 0.0, 700.0)
            valid_gripper = np.array([0.0, 0.5, 1.0])
            action[2] = valid_gripper[np.argmin(np.abs(valid_gripper - action[2]))]

            uncertainty_hist.append(0.0)
            if len(uncertainty_hist) > 150:
                uncertainty_hist = uncertainty_hist[-150:]
            env.uncertainty_history = uncertainty_hist

            pygame.event.pump()
            obs_prev2 = obs_prev1.copy()
            obs_prev1 = obs.copy()
            obs, _, done, _, _ = env.step(action)
            if done:
                screen_state = "done"
            env.render(ep, step_id, screen_state, n_episodes)
            step_id += 1
            time.sleep(STEP_SLEEP)

        controlled_delay(500)
        rating = wait_for_yes_no(env.screen, ["Did the robot complete", "the task as desired?"])
        ratings.append((ep, rating))
        print(f"  Rating: {'YES' if rating else 'NO'}")
        controlled_delay(1000)

    return ratings


def run_comparison_rollout(env, n_episodes, ensemble, method, n_stops=3):
    """
    Robot rollout with n_stops forced human-takeover segments at evenly-spaced steps.
    For each segment, records (robot_pred, human_action) per step.
    Human presses SPACE to end each segment and return control to the robot.
    Returns list of per-episode dicts: {"episode", "segments": [[{step, robot_pred, human_action}...]...]}.
    """
    policy_list = _eval_policy_list(ensemble, method)
    all_comparisons = []
    env.can_intervene = False
    stop_steps = [(s + 1) * EVAL_STOP_INTERVAL for s in range(n_stops)]

    for ep in range(n_episodes):
        wait_for_space(
            env.screen,
            [f"Comparison  {ep + 1} / {n_episodes}", f"Robot pauses {n_stops}x — take over briefly each time."],
            sub="Press SPACE to begin",
        )

        obs, _ = env.reset()
        obs_prev1 = obs.copy()
        obs_prev2 = obs.copy()
        done = False
        screen_state = "start"
        step_id = 0
        stuck_count = 0
        stop_idx = 0
        ep_segments = []
        current_segment = []
        uncertainty_hist = []

        env.help_remaining_time = 0.0
        env.render(ep, step_id, screen_state, n_episodes)
        print(f"\n[Eval comparison ep {ep + 1}/{n_episodes}]  stops at steps {stop_steps}")

        while not done:
            if screen_state == "start":
                if env.check_button_click():
                    screen_state = "robot"
                continue

            if screen_state == "ready_help":
                if env.ready_to_help():
                    screen_state = "help"
                    current_segment = []
                else:
                    env.render(ep, step_id, screen_state, n_episodes)
                    time.sleep(STEP_SLEEP)
                    continue

            if similar_state(obs, obs_prev1):
                stuck_count += 1
            else:
                stuck_count = 0

            action, _ = get_ensemble_prediction(obs, obs_prev1, obs_prev2, policy_list)

            if stuck_count >= 13:   sigma = 15.0
            elif stuck_count >= 10: sigma = 10.0
            elif stuck_count >= 7:  sigma = 5.0
            else:                   sigma = 0.0
            if sigma > 0.0:
                noise = np.random.normal(0.0, sigma, 2)
                action[0] += noise[0]; action[1] += noise[1]

            robot_prediction = action.copy()
            robot_prediction[0] = np.clip(robot_prediction[0], 0.0, 700.0)
            robot_prediction[1] = np.clip(robot_prediction[1], 0.0, 700.0)
            valid_gripper = np.array([0.0, 0.5, 1.0])
            robot_prediction[2] = valid_gripper[np.argmin(np.abs(valid_gripper - robot_prediction[2]))]

            exec_action = robot_prediction.copy()

            uncertainty_hist.append(0.0)
            if len(uncertainty_hist) > 150:
                uncertainty_hist = uncertainty_hist[-150:]
            env.uncertainty_history = uncertainty_hist

            if screen_state == "robot":
                if stop_idx < n_stops and step_id >= stop_steps[stop_idx]:
                    screen_state = "ready_help"
                    stop_idx += 1

            elif screen_state == "help":
                human_action = env.get_help()
                if human_action is not None:
                    exec_action = human_action.copy()
                    current_segment.append({
                        "step":         step_id,
                        "robot_pred":   robot_prediction.copy(),
                        "human_action": human_action.copy(),
                    })

                if pygame.event.peek(pygame.KEYDOWN):
                    for event in pygame.event.get(pygame.KEYDOWN):
                        if event.key == pygame.K_SPACE:
                            ep_segments.append(current_segment)
                            current_segment = []
                            screen_state = "robot"
                            print(f"    Segment {len(ep_segments)} ended: {len(ep_segments[-1])} steps")

            obs_prev2 = obs_prev1.copy()
            obs_prev1 = obs.copy()
            obs, _, done, _, _ = env.step(exec_action)
            if done:
                if current_segment:
                    ep_segments.append(current_segment)
                screen_state = "done"
            env.render(ep, step_id, screen_state, n_episodes)
            step_id += 1
            time.sleep(STEP_SLEEP)

        all_comparisons.append({"episode": ep, "segments": ep_segments})
        print(f"  Comparison ep {ep}: {len(ep_segments)} segments, "
              f"{sum(len(s) for s in ep_segments)} comparison steps")
        controlled_delay(3000)

    return all_comparisons


def run_evaluation(env, ensemble, method, n_rating=3, n_comparison=2):
    """
    Human evaluation stage:
      1. Rating: human watches n_rating autonomous rollouts and clicks Yes/No.
      2. Comparison: robot pauses 3x per episode for n_comparison episodes;
         human takes over each segment (SPACE to end), robot pred vs human action recorded.
    Results saved to data/study/eval_results.pkl.
    """
    print(f"\n{'='*55}")
    print(f"  EVALUATION  —  {n_rating} rating + {n_comparison} comparison episode(s)")
    print(f"{'='*55}")

    wait_for_space(
        env.screen,
        ["Evaluation  —  Part 1", f"Rate {n_rating} robot rollout(s): Yes or No"],
        sub="Press SPACE to begin",
    )
    ratings = run_rating_rollout(env, n_rating, ensemble, method)

    wait_for_space(
        env.screen,
        ["Evaluation  —  Part 2", f"{n_comparison} rollout(s) with {3} pauses each"],
        sub="Press SPACE to begin",
    )
    comparisons = run_comparison_rollout(env, n_comparison, ensemble, method)

    results = {"method": method, "ratings": ratings, "comparisons": comparisons}
    save_path = "data/study/eval_results.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\n  Saved evaluation results → {save_path}")
    return results


# ──────────────────────────────────────────────────────────────
# Main study loop
# ──────────────────────────────────────────────────────────────

def run_study():
    # ── user inputs ───────────────────────────────
    shape_map  = {"1": "square", "2": "triangle", "3": "zigzag", "4": "swirl"}
    method_map = {
        "1": METHOD_CONFORMAL_ENSEMBLE,
        "2": METHOD_CONFORMAL,
        "3": METHOD_ENSEMBLE,
        "4": METHOD_LAZYDAGGER,
        "5": METHOD_ENSEMBLE_CLASSIFIER,
    }
    shape        = shape_map[input("Shape  1=square  2=triangle  3=zigzag  4=swirl: ").strip()]
    method       = method_map[input("Method  1=conformal++  2=conformal  3=ensemble  4=lazydagger  5=ensemble_clf: ").strip()]
    N            = int(input("Episodes per round (N): "))
    M            = int(input("Number of rounds (M): "))

    # ── paths ─────────────────────────────────────
    study_model_dir = "trained_policy/study"
    # conformal and lazydagger use a single policy; conformal++ and ensemble use a full ensemble
    n_init_members = 1 if method in (METHOD_CONFORMAL, METHOD_LAZYDAGGER) else N_ENSEMBLE
    init_policy_paths = [
        f"trained_policy/ensemble/cont_policy_{shape}20hz_{i+1}.pth"
        for i in range(n_init_members)
    ]
    init_stats_paths = [
        f"trained_policy/ensemble/norm_stats_{shape}20hz_{i+1}.npz"
        for i in range(n_init_members)
    ]

    os.makedirs("data/study",    exist_ok=True)
    os.makedirs(study_model_dir, exist_ok=True)

    print(f"\nStudy config: shape={shape}  method={method}  N={N}  M={M}  buffer_size={BUFFER_SIZE}")
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

    # load initial model(s)
    policy_list = load_ensemble(init_policy_paths, init_stats_paths)
    if method in (METHOD_LAZYDAGGER, METHOD_ENSEMBLE_CLASSIFIER):
        clf_init_path   = "trained_safety_classifier/safety_classifier.pth"
        clf_init_stats  = "trained_safety_classifier/norm_stats_safety_classifier.npz"
        ensemble = (policy_list, load_lazydagger_classifier(clf_init_path, clf_init_stats))
        print(f"  Loaded initial safety classifier from {clf_init_path}")
    else:
        ensemble = policy_list
    print(f"  Loaded initial model ({n_init_members} member(s), method={method})")

    for round_idx in range(M):
        # per-round image directory
        img_dir = f"jam_state_img/study/round_{round_idx}"
        os.makedirs(img_dir, exist_ok=True)
        env.img_save_dir = f"study/round_{round_idx}"

        print(f"\n{'='*55}")
        print(f"  ROUND {round_idx + 1} / {M}  —  {N} episodes  (method={method})")
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
            method=method,
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
        n_members = 1 if method in (METHOD_CONFORMAL, METHOD_LAZYDAGGER) else N_ENSEMBLE
        new_policy_paths, new_stats_path = retrain_policy(
            episode_buffer=episode_buffer,
            round_idx=round_idx,
            output_dir=study_model_dir,
            n_members=n_members,
        )
        policy_list = load_ensemble(new_policy_paths[:n_members], [new_stats_path] * n_members)
        if method in (METHOD_LAZYDAGGER, METHOD_ENSEMBLE_CLASSIFIER):
            clf_path = train_lazydagger_classifier(episode_buffer, study_model_dir, round_idx, new_stats_path)
            ensemble = (policy_list, load_lazydagger_classifier(clf_path, new_stats_path))
        else:
            ensemble = policy_list
        pygame.event.clear()   # flush events queued during retraining (no pump was called)

    # ── study complete ────────────────────────────
    print(f"\nStudy complete: {N * M} total episodes across {M} rounds.")

    run_evaluation(env, ensemble=ensemble, method=method, n_rating=1, n_comparison=1)

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
