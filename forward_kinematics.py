import numpy as np

# UR5 DH parameters
d = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]
a = [0, -0.425, -0.39225, 0, 0, 0]
alpha = [1.5708, 0, 0, 1.5708, -1.5708, 0]


def dh_matrix(a, alpha, d, theta):
    
    T = np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])
    
    return T


def forward_kinematics(theta):
    
    T = np.eye(4)
    
    for i in range(6):
        T = np.dot(T, dh_matrix(a[i], alpha[i], d[i], theta[i]))
    
    return T


# Test run
if __name__ == "__main__":
    
    theta = [0, 0, 0, 0, 0, 0]
    
    T = forward_kinematics(theta)
    
    print("Forward Kinematics Result:")
    print(T)