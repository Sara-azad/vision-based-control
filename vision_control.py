"""
Closed-loop control using ONLY vision-estimated state -- no ground-truth
state is given to the controller at any point. At every timestep:
  1. The true physics simulator advances one step (this is the real world).
  2. We render a camera frame of the new state.
  3. We estimate (x, x_dot, theta, theta_dot) from consecutive frames.
  4. The LQR controller acts on the ESTIMATE, not the true state.

This mirrors a real robot: the controller never has privileged access to
the true state, only to what its sensor (camera + CV pipeline) reports.

Run with: python vision_control.py
Produces: plots/vision_vs_groundtruth.png
"""

import numpy as np
import matplotlib.pyplot as plt

from src.environment import CartPoleEnv
from src.lqr_controller import LQRController
from src.renderer import render_frame
from src.vision_estimator import estimate_state_from_frames

N_TRIALS = 20
MAX_STEPS = 500


def run_episode_ground_truth(env, controller):
    state = env.reset()
    done, steps = False, 0
    while not done and steps < MAX_STEPS:
        action = controller.choose_action(state)
        state, reward, done = env.step(action)
        steps += 1
    return steps


def run_episode_vision(env, controller):
    """Same episode, but the controller only sees vision-estimated state.
    The very first action uses the true initial state (a real system
    would calibrate/localize at startup) -- after that, everything is
    estimated from consecutive rendered frames."""
    state = env.reset()
    prev_frame = render_frame(state[0], state[2])
    done, steps = False, 0

    # Bootstrap: first action from true state (one-time startup calibration)
    action = controller.choose_action(state)
    state, reward, done = env.step(action)
    steps += 1

    while not done and steps < MAX_STEPS:
        curr_frame = render_frame(state[0], state[2])
        estimated_state = estimate_state_from_frames(prev_frame, curr_frame, env.dt)

        if estimated_state is None:
            # Vision pipeline failed to detect cart/pole this frame
            # (e.g. pole angle out of camera's tracked range) -- treat as
            # a control failure, same as losing the pole in a real system.
            break

        action = controller.choose_action(estimated_state)
        prev_frame = curr_frame
        state, reward, done = env.step(action)
        steps += 1

    return steps


def main():
    env = CartPoleEnv()
    controller = LQRController()

    print("Running ground-truth LQR trials...")
    gt_results = [run_episode_ground_truth(env, controller) for _ in range(N_TRIALS)]

    print("Running vision-based LQR trials (this is slower -- rendering + CV every step)...")
    vision_results = [run_episode_vision(env, controller) for _ in range(N_TRIALS)]

    print(f"\nGround-truth state:  mean {np.mean(gt_results):.1f} "
          f"(std {np.std(gt_results):.1f})")
    print(f"Vision-estimated:    mean {np.mean(vision_results):.1f} "
          f"(std {np.std(vision_results):.1f})")

    labels = ["LQR\n(ground-truth state)", "LQR\n(vision-estimated state)"]
    means = [np.mean(gt_results), np.mean(vision_results)]
    stds = [np.std(gt_results), np.std(vision_results)]
    colors = ["#4C72B0", "#C44E52"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, means, yerr=stds, capsize=8, color=colors, alpha=0.85)
    ax.axhline(500, color="gray", linestyle="--", alpha=0.4, label="Max possible (500)")
    ax.set_ylabel("Timesteps balanced (mean \u00b1 std, 20 trials)")
    ax.set_title("LQR Control: Ground-Truth State vs. Vision-Estimated State")
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/vision_vs_groundtruth.png", dpi=150)
    print("Saved plots/vision_vs_groundtruth.png")


if __name__ == "__main__":
    main()
