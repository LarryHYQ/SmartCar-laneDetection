"""
from:
https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathTracking/move_to_pose/move_to_pose.py
"""


import numpy as np
from math import sqrt
from Config import *


dt = 0.01


def curvatureSolve(x0, y0, t0, x1, y1, t1):
    dx = x1 - x0
    dy = y1 - y0
    alpha = (np.arctan2(dy, dx) - t0 + np.pi) % (2 * np.pi) - np.pi
    beta = (t1 - t0 - alpha + np.pi) % (2 * np.pi) - np.pi
    rho = sqrt(dx * dx + dy * dy)
    v = K_RHO * rho
    w = K_ALPHA * alpha + K_BETA * beta
    return w / v


def move_to_pose(x_start, y_start, theta_start, x_goal, y_goal, theta_goal):
    """
    rho is the distance between the robot and the goal position
    alpha is the angle to the goal relative to the heading of the robot
    beta is the angle between the robot's position and the goal position plus the goal angle

    K_RHO*rho and K_ALPHA*alpha drive the robot along a line towards the goal
    K_BETA*beta rotates the line so that it is parallel to the goal angle
    """
    x = x_start
    y = y_start
    theta = theta_start

    x_diff = x_goal - x
    y_diff = y_goal - y

    x_traj, y_traj = [], []

    rho = sqrt(x_diff * x_diff + y_diff * y_diff)
    while rho > 0.001:
        x_traj.append(x)
        y_traj.append(y)

        x_diff = x_goal - x
        y_diff = y_goal - y

        # Restrict alpha and beta (angle differences) to the range
        # [-pi, pi] to prevent unstable behavior e.g. difference going
        # from 0 rad to 2*pi rad with slight turn

        rho = sqrt(x_diff * x_diff + y_diff * y_diff)
        alpha = (np.arctan2(y_diff, x_diff) - theta + np.pi) % (2 * np.pi) - np.pi
        beta = (theta_goal - theta - alpha + np.pi) % (2 * np.pi) - np.pi

        v = K_RHO * rho
        w = K_ALPHA * alpha + K_BETA * beta

        if alpha > np.pi / 2 or alpha < -np.pi / 2:
            v = -v

        theta = theta + w * dt
        x = x + v * np.cos(theta) * dt
        y = y + v * np.sin(theta) * dt

    return x_traj, y_traj

