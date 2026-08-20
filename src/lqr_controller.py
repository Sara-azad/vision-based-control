"""
LQR (Linear-Quadratic Regulator) controller for CartPole.

This is the classical, model-based counterpart to the Q-learning agent --
same physical system, opposite philosophy: instead of learning a policy
from trial and error, we linearize the known dynamics and solve for the
mathematically optimal linear controller.

--- Step 1: Linearize around the upright equilibrium (theta = 0) ---

Starting from the nonlinear equations in environment.py:

    temp      = (F + m_p*l*theta_dot^2*sin(theta)) / (m_c+m_p)
    theta_acc = (g*sin(theta) - cos(theta)*temp) / (l*(4/3 - m_p*cos^2(theta)/(m_c+m_p)))
    x_acc     = temp - m_p*l*theta_acc*cos(theta) / (m_c+m_p)

Near theta = 0: sin(theta) ~ theta, cos(theta) ~ 1, and the theta_dot^2*sin(theta)
term is second-order small (product of two small quantities), so we drop it.
This gives the standard linearized cart-pole state-space model:

    state = [x, x_dot, theta, theta_dot],  input u = F

    x_dot_dot     = (1/M) * F + (-m*g/M) * theta                     [approx, M = m_c]
    theta_dot_dot = -(1/(M*l)) * F + (M+m)*g/(M*l) * theta

which we assemble into the standard state-space matrices A (4x4) and
B (4x1) such that:  d(state)/dt = A @ state + B @ F

--- Step 2: Solve for the optimal gain K ---

LQR finds the control law u = -K @ state that minimizes the cost
    J = integral( state^T Q state + u^T R u ) dt
by solving the continuous-time Algebraic Riccati Equation (ARE):
    A^T P + P A - P B R^-1 B^T P + Q = 0
then K = R^-1 B^T P.

Q and R are design choices: Q penalizes state deviation (we weight theta
and x most heavily -- keeping the pole up and cart centered matters most),
R penalizes control effort (how hard we push).
"""

import numpy as np
from scipy.linalg import solve_continuous_are


class LQRController:
    def __init__(self, mass_cart=1.0, mass_pole=0.1, length=0.5, gravity=9.8,
                 Q=None, R=None):
        self.mass_cart = mass_cart
        self.mass_pole = mass_pole
        self.length = length
        self.gravity = gravity
        self.total_mass = mass_cart + mass_pole

        M = mass_cart
        m = mass_pole
        l = length
        g = gravity

        # Linearized state-space matrices around theta = 0 (upright).
        # State order: [x, x_dot, theta, theta_dot]
        self.A = np.array([
            [0, 1, 0, 0],
            [0, 0, -(m * g) / M, 0],
            [0, 0, 0, 1],
            [0, 0, (M + m) * g / (M * l), 0],
        ])
        self.B = np.array([
            [0],
            [1 / M],
            [0],
            [-1 / (M * l)],
        ])

        # Default cost weights: prioritize keeping the pole upright (theta)
        # and the cart centered (x); moderate penalty on control effort.
        self.Q = Q if Q is not None else np.diag([1.0, 1.0, 10.0, 1.0])
        self.R = R if R is not None else np.array([[0.1]])

        # Solve the continuous-time Algebraic Riccati Equation for P,
        # then compute the optimal gain K.
        P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.inv(self.R) @ self.B.T @ P  # shape (1, 4)

    def compute_force(self, state):
        """Optimal continuous control law: u = -K @ state."""
        state = np.array(state)
        force = -self.K @ state
        return float(force[0])

    def choose_action(self, state):
        """Convert the continuous LQR force into the discrete left/right
        action the CartPoleEnv expects, so both controllers can be run
        through the exact same environment/step() interface."""
        force = self.compute_force(state)
        return 1 if force >= 0 else 0
