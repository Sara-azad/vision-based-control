"""
CartPole environment implemented from the physical equations of motion.

Why write this instead of using a library (e.g. gymnasium)?
  1. It proves you understand the *dynamics*, not just the RL API.
  2. It's a direct bridge between control theory and RL: the state vector
     below is exactly the state-space representation x = [x, x_dot, theta, theta_dot]
     you'd use to write this same system as x_dot = f(x, u) for a classical
     controller (PID, LQR, etc). RL and classical control are solving the
     SAME dynamical system -- just choosing the control input u differently.

Physical system:
    A pole is attached by an un-actuated joint to a cart, which moves along
    a frictionless track. A force F (left or right) is applied to the cart.
    The goal is to keep the pole upright by choosing F at every timestep.

State vector (this is your control-theory state vector):
    x        - cart position (m)
    x_dot    - cart velocity (m/s)
    theta    - pole angle from vertical (rad)
    theta_dot- pole angular velocity (rad/s)

Equations of motion (derived from Lagrangian mechanics for this system --
this is the standard formulation used across the control/RL literature,
e.g. Barto, Sutton & Anderson 1983):

    temp      = (F + m_p * l * theta_dot^2 * sin(theta)) / (m_c + m_p)
    theta_acc = (g * sin(theta) - cos(theta) * temp) /
                (l * (4/3 - m_p * cos(theta)^2 / (m_c + m_p)))
    x_acc     = temp - m_p * l * theta_acc * cos(theta) / (m_c + m_p)

We integrate these with simple Euler integration (small timestep dt).
"""

import numpy as np


class CartPoleEnv:
    def __init__(self, mass_pole=0.1, length=0.5):
        # Physical constants. mass_pole and length are exposed as
        # parameters (rather than hardcoded) so we can build a "real"
        # system that differs from the model an LQR controller was
        # designed with -- used in the model-mismatch experiment.
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = mass_pole
        self.total_mass = self.mass_cart + self.mass_pole
        self.length = length  # half the pole's length
        self.polemass_length = self.mass_pole * self.length
        self.force_mag = 10.0
        self.dt = 0.02  # seconds between state updates

        # Episode ends (system considered "failed") if:
        self.theta_threshold = 12 * 2 * np.pi / 360  # 12 degrees, in radians
        self.x_threshold = 2.4  # cart position limit (m)

        self.max_steps = 500
        self.state = None
        self.steps_taken = 0

    def reset(self):
        """Start near the unstable equilibrium (pole upright) with small
        random perturbation -- standard practice so the agent must learn
        to handle slightly different starting conditions."""
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_taken = 0
        return self.state.copy()

    def step(self, action, disturbance=0.0):
        """
        action: 0 -> push cart left, 1 -> push cart right
        disturbance: extra external force (N) added this step, e.g. to
            simulate a sudden bump/gust -- used in the disturbance-
            rejection experiment. Zero by default (no effect on normal use).
        Returns: next_state, reward, done
        """
        x, x_dot, theta, theta_dot = self.state
        force = (self.force_mag if action == 1 else -self.force_mag) + disturbance

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sin_theta) / self.total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.length * (4.0 / 3.0 - self.mass_pole * cos_theta ** 2 / self.total_mass)
        )
        x_acc = temp - self.polemass_length * theta_acc * cos_theta / self.total_mass

        # Euler integration
        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * x_acc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps_taken += 1

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold
            or theta > self.theta_threshold
            or self.steps_taken >= self.max_steps
        )

        # +1 reward for every timestep the pole stays upright -- this
        # directly rewards balancing longer, which is the control objective.
        reward = 1.0

        return self.state.copy(), reward, done
