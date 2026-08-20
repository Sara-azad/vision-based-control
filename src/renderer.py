"""
Renders the CartPole system's physical state as a synthetic camera image
-- simulating what an overhead camera looking at the real system would
see. This is the "sensor" for the vision-based control experiment: the
controller will only get to see this image, never the true (x, theta)
state directly.

Design choices, and why:
  - Cart is drawn as a solid BLUE rectangle, pole as a solid RED line.
    Using distinct, saturated colors is deliberate -- it mirrors a very
    common real robotics technique (color-based fiducial markers) that
    makes the perception problem tractable with classical CV rather than
    needing a trained detector.
  - Image coordinates: pixel (0,0) is top-left, x increases rightward,
    y increases downward (standard image convention) -- the vision
    estimator has to convert back to physical, sign-consistent
    coordinates. Getting this conversion right/wrong is a real, common
    source of bugs in robotics vision pipelines, so it's called out
    explicitly in vision_estimator.py.
"""

import numpy as np
from PIL import Image, ImageDraw

IMG_WIDTH = 400
IMG_HEIGHT = 300
TRACK_HALF_WIDTH_M = 2.4  # matches CartPoleEnv.x_threshold
PIXELS_PER_METER = (IMG_WIDTH / 2) / TRACK_HALF_WIDTH_M
CART_Y_PIXEL = IMG_HEIGHT - 60  # cart sits near the bottom of the frame
CART_WIDTH_PX, CART_HEIGHT_PX = 40, 20
POLE_LENGTH_PX = 80  # visual length; doesn't need to exactly equal the physical length in pixels


def world_x_to_pixel(x):
    """Convert physical cart position (meters, 0 = center) to pixel x."""
    return int(IMG_WIDTH / 2 + x * PIXELS_PER_METER)


def render_frame(x, theta):
    """
    x: cart position (m)
    theta: pole angle from vertical (rad)
    Returns: a numpy array (H, W, 3) uint8 image.
    """
    img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Track reference line
    draw.line([(0, CART_Y_PIXEL + CART_HEIGHT_PX // 2),
               (IMG_WIDTH, CART_Y_PIXEL + CART_HEIGHT_PX // 2)],
              fill=(200, 200, 200), width=2)

    cart_px = world_x_to_pixel(x)

    # Cart: solid blue rectangle
    draw.rectangle([
        cart_px - CART_WIDTH_PX // 2, CART_Y_PIXEL - CART_HEIGHT_PX // 2,
        cart_px + CART_WIDTH_PX // 2, CART_Y_PIXEL + CART_HEIGHT_PX // 2,
    ], fill=(30, 60, 220))

    # Pole: solid red line, pivoting from the top of the cart.
    # theta = 0 is straight up; positive theta tips it (matches the
    # physics convention in environment.py: sin(theta) contributes to x).
    pivot = (cart_px, CART_Y_PIXEL - CART_HEIGHT_PX // 2)
    tip_x = pivot[0] + POLE_LENGTH_PX * np.sin(theta)
    tip_y = pivot[1] - POLE_LENGTH_PX * np.cos(theta)
    draw.line([pivot, (tip_x, tip_y)], fill=(220, 30, 30), width=5)

    return np.array(img)
