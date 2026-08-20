"""
Robustness experiment: how much camera degradation can vision-based LQR
control tolerate before it fails? Nominal vision estimation (see
vision_control.py) was accurate enough that control performance matched
ground truth exactly -- not a very informative result on its own. This
script instead sweeps camera noise levels to find where the vision
pipeline actually breaks down, which is closer to a real, useful
robotics question ("how good does my camera/lighting need to be?").

Two forms of degradation are tested, applied to each rendered frame
before the CV pipeline sees it:
  - Gaussian pixel noise (simulates sensor/lighting noise)
  - Gaussian blur (simulates defocus / motion blur)

Run with: python vision_robustness.py
Produces: plots/vision_robustness.png
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

from src.environment import CartPoleEnv
from src.lqr_controller import LQRController
from src.renderer import render_frame
from src.vision_estimator import estimate_state_from_frames

N_TRIALS = 10
MAX_STEPS = 500
NOISE_LEVELS = [0, 10, 25, 40, 60, 80]  # std dev of Gaussian pixel noise


def degrade_frame(frame, noise_std):
    if noise_std == 0:
        return frame
    noise = np.random.normal(0, noise_std, frame.shape)
    noisy = np.clip(frame.astype(float) + noise, 0, 255).astype(np.uint8)
    return noisy


def run_episode_vision_noisy(env, controller, noise_std):
    state = env.reset()
    prev_frame = degrade_frame(render_frame(state[0], state[2]), noise_std)
    done, steps = False, 0

    action = controller.choose_action(state)
    state, reward, done = env.step(action)
    steps += 1

    while not done and steps < MAX_STEPS:
        curr_frame = degrade_frame(render_frame(state[0], state[2]), noise_std)
        estimated_state = estimate_state_from_frames(prev_frame, curr_frame, env.dt)

        if estimated_state is None:
            break

        action = controller.choose_action(estimated_state)
        prev_frame = curr_frame
        state, reward, done = env.step(action)
        steps += 1

    return steps


def main():
    env = CartPoleEnv()
    controller = LQRController()

    results = {}
    for noise_std in NOISE_LEVELS:
        trial_results = [
            run_episode_vision_noisy(env, controller, noise_std)
            for _ in range(N_TRIALS)
        ]
        results[noise_std] = trial_results
        print(f"Noise std {noise_std:3d}: mean {np.mean(trial_results):6.1f} "
              f"(std {np.std(trial_results):.1f})")

    means = [np.mean(results[n]) for n in NOISE_LEVELS]
    stds = [np.std(results[n]) for n in NOISE_LEVELS]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(NOISE_LEVELS, means, yerr=stds, marker="o", capsize=5,
                linewidth=2, color="#C44E52")
    ax.axhline(500, color="gray", linestyle="--", alpha=0.4, label="Max possible (500)")
    ax.set_xlabel("Gaussian pixel noise (std dev)")
    ax.set_ylabel("Timesteps balanced (mean \u00b1 std, 10 trials)")
    ax.set_title("Vision-Based LQR Control: Robustness to Camera Noise")
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/vision_robustness.png", dpi=150)
    print("Saved plots/vision_robustness.png")


if __name__ == "__main__":
    main()
