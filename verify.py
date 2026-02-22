# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 23:37:22 2026

@author: HIMANSHU
"""

import numpy as np
from forward_kinematics import forward_kinematics
from inverse_kinematics import inverse_kinematics


print("UR5 IK Verification Tool")
print("------------------------")

# Target position
target = np.array([-0.5, -0.2, 0.3])

print("Target Position:")
print(target)

# Solve inverse kinematics
theta_solution = inverse_kinematics(target)

print("\nJoint Angles Solution:")
print(theta_solution)

# Verify using forward kinematics
T = forward_kinematics(theta_solution)

verified_position = T[0:3, 3]

print("\nVerified Position:")
print(verified_position)

# Compute error
error = np.linalg.norm(target - verified_position)

print("\nPosition Error:")
print(error)