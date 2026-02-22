# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 23:31:24 2026

@author: HIMANSHU
"""

import numpy as np
from forward_kinematics import forward_kinematics


def compute_jacobian(theta):
    """
    Compute the Jacobian matrix for UR5 using numerical differentiation

    Input:
        theta : list or numpy array of 6 joint angles

    Output:
        J : 3x6 Jacobian matrix
    """

    # Convert to numpy array
    theta = np.array(theta, dtype=float)

    # Initialize Jacobian matrix
    J = np.zeros((3, 6))

    # Small change for numerical differentiation
    delta = 1e-6

    # Current end effector position
    T = forward_kinematics(theta)
    current_position = T[0:3, 3]

    # Compute partial derivatives
    for i in range(6):

        theta_new = theta.copy()
        theta_new[i] += delta

        T_new = forward_kinematics(theta_new)
        new_position = T_new[0:3, 3]

        # Numerical derivative
        J[:, i] = (new_position - current_position) / delta

    return J


# Test the Jacobian
if __name__ == "__main__":

    theta = [0, 0, 0, 0, 0, 0]

    J = compute_jacobian(theta)

    print("Jacobian Matrix:")
    print(J)