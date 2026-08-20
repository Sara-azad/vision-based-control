"""
Classical computer vision pipeline that estimates the CartPole's state
(x, theta) from a single rendered camera frame -- no ground-truth state
is used here at all.

Pipeline (standard classical CV, no learned model):
  1. Color-threshold the image in HSV to isolate the blue cart and red
     pole as separate binary masks. HSV is used instead of RGB because
     it's far more robust to brightness/lighting changes -- a real
     robotics-vision default, even though our synthetic images don't
     actually vary in lighting.
  2. Cart position: find the centroid (center of mass) of the blue mask
     -> pixel x -> convert back to meters using the known camera
     calibration (PIXELS_PER_METER from renderer.py).
  3. Pole angle: find all pixels in the red mask, fit a line through them
     with PCA (principal component analysis) -- the dominant eigenvector
     of the pixel coordinate covariance matrix gives the line's
     direction, which converts directly to theta. Using PCA rather than
     e.g. picking the topmost/bottommost pixel is deliberately more
     robust to a few noisy/missing pixels at the pole's tip.

Velocity (x_dot, theta_dot) can't be recovered from a single frame --
consistent with a real camera, which only gives you position, not
velocity, in one shot. It's estimated by finite-difference between
consecutive frames, exactly as a real vision-based control system would.
"""

import numpy as np
import cv2

from src.renderer import IMG_WIDTH, IMG_HEIGHT, PIXELS_PER_METER, CART_Y_PIXEL

# HSV color ranges tuned to the exact colors used in renderer.py
BLUE_LOWER = np.array([100, 100, 50])
BLUE_UPPER = np.array([140, 255, 255])
RED_LOWER1 = np.array([0, 100, 50])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([170, 100, 50])
RED_UPPER2 = np.array([180, 255, 255])


def estimate_cart_x(frame):
    """Returns estimated cart position in meters, or None if not found."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    centroid_px = xs.mean()
    x_meters = (centroid_px - IMG_WIDTH / 2) / PIXELS_PER_METER
    return x_meters


def estimate_pole_theta(frame):
    """Returns estimated pole angle (radians from vertical), or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    mask = mask1 | mask2
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return None

    # PCA: center the points, find the dominant direction via the
    # covariance matrix's leading eigenvector.
    pts = np.stack([xs, ys], axis=1).astype(float)
    pts -= pts.mean(axis=0)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, np.argmax(eigvals)]  # (dx, dy) of the pole's long axis

    dx, dy = direction
    # Ensure the direction points "upward" in image coords (negative dy),
    # matching the pole tip being above the pivot.
    if dy > 0:
        dx, dy = -dx, -dy

    # In renderer.py: tip_x = pivot_x + L*sin(theta), tip_y = pivot_y - L*cos(theta)
    # so (dx, dy) = (sin(theta), -cos(theta))  =>  theta = atan2(dx, -dy)
    theta = np.arctan2(dx, -dy)
    return theta


def estimate_state_from_frames(prev_frame, curr_frame, dt):
    """
    Full state estimate (x, x_dot, theta, theta_dot) from two consecutive
    frames -- velocities via finite difference, matching how a real
    vision-based controller would have to work (no direct velocity sensor).
    Returns None if the cart/pole isn't detected in either frame.
    """
    x_prev = estimate_cart_x(prev_frame)
    theta_prev = estimate_pole_theta(prev_frame)
    x_curr = estimate_cart_x(curr_frame)
    theta_curr = estimate_pole_theta(curr_frame)

    if None in (x_prev, theta_prev, x_curr, theta_curr):
        return None

    x_dot = (x_curr - x_prev) / dt
    theta_dot = (theta_curr - theta_prev) / dt

    return np.array([x_curr, x_dot, theta_curr, theta_dot])
