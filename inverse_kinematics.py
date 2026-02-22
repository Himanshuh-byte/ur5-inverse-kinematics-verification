import numpy as np
from forward_kinematics import forward_kinematics
from jacobian import compute_jacobian


def inverse_kinematics(target_position,
                       max_iterations=1000,
                       tolerance=1e-4):

    theta = np.zeros(6)

    target_position = np.array(target_position)

    for iteration in range(max_iterations):

        T = forward_kinematics(theta)

        current_position = T[0:3, 3]

        error = target_position - current_position

        if np.linalg.norm(error) < tolerance:

            print("Converged in", iteration, "iterations")

            return theta

        J = compute_jacobian(theta)

        J_pinv = np.linalg.pinv(J)

        theta = theta + np.dot(J_pinv, error)

    print("Did not fully converge")

    return theta


# Test
if __name__ == "__main__":

    target = [-0.5, -0.2, 0.3]

    theta_solution = inverse_kinematics(target)

    print("Joint Angles Solution:")
    print(theta_solution)

    T = forward_kinematics(theta_solution)

    print("Verified Position:")
    print(T[0:3, 3])