# 331396
# A - 6, B - 9, C - 3

import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

class GradientDescent:
    def __init__(self, point: list, learning_rate: float, num_iterations: int, function_grad: classmethod) -> None:
        self.point = point
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.function_grad = function_grad

    def grad_descent(self, isInDomain: classmethod) -> np.array:
        desc_point = np.array(self.point.copy())
        desc_trajectory = [desc_point.copy()]
    
        for _ in range(self.num_iterations):
            desc_point = desc_point - self.learning_rate * self.function_grad(desc_point)
            if not isInDomain(desc_point):
                return np.array(desc_trajectory)
            desc_trajectory.append(desc_point.copy())
            
        return np.array(desc_trajectory)


class Function_f:
    def __init__(self) -> None:
        pass

    def formula(self, x: np.array) -> float:
        return 6*x + 9*np.sin(x)

    def gradient(self, x: np.array) -> np.array:
        return 6 + 9*np.cos(x)

    def isInDomain(self, x: np.array) -> bool:
        return -4*np.pi <= x <= 4*np.pi

class Function_g:
    def __init__(self) -> None:
        pass

    def formula(self, point: np.array) -> float:
        x, y = point
        return 3*x*y / (np.exp(x**2 + y**2))

    def gradient(self, point: np.array) -> np.array:
        x, y = point
        return np.array([
            (3*y - 6*x**2*y) / (np.exp(x**2 + y**2)),
            (3*x - 6*x*y**2) / (np.exp(x**2 + y**2))
        ])
    def isInDomain(self, point: np.array) -> bool:
        x, y = point
        return -2 <= x <= 2 and -2 <= y <= 2
    
class DataVisualizer:
    def __init__(self):
        pass
    
    def Visualize_2D(self, X, Y, trajectory_args, trajectory_values):
        plt.figure(figsize=(6, 10))
        
        plt.title('F(x) plot')
        plt.xlabel('X')
        plt.ylabel('Y')
        
        plt.plot(X, Y)

        desc_trajectory_args, asc_trajectory_args = trajectory_args
        desc_trajectory_values, asc_trajectory_values = trajectory_values

        plt.plot(
            desc_trajectory_args, desc_trajectory_values,
            color='red', linestyle='-', marker='o', label='Trajectory descending'
        )
        plt.plot(
            asc_trajectory_args, asc_trajectory_values,
            color='orange', linestyle='-', marker='*', label='Trajectory ascending'
        )

        plt.legend()
        plt.grid(True)
        
        plt.show()

    def Visualize_3D(self, X, Y, Z, trajectory_args, trajectory_values):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(X, Y, Z, cmap='inferno')

        desc_trajectory_args, asc_trajectory_args = trajectory_args
        desc_trajectory_values, asc_trajectory_values = trajectory_values

        ax.plot(
            desc_trajectory_args[:, 0], desc_trajectory_args[:, 1], desc_trajectory_values,
            color='green', linestyle='--', marker='*', label='Trajectory descending'
        )
        ax.plot(
            asc_trajectory_args[:, 0], asc_trajectory_args[:, 1], asc_trajectory_values,
            color='blue', linestyle='-', marker='o', label='Trajectory ascending'
        )

        ax.set_title('G(x) plot')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('G(X, Y)')

        plt.legend()
        plt.show()

    
    def Visualize_3D_subplots(self, X, Y, Z, trajectory_args, trajectory_values):
        fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(10, 10))
        ax1.plot_surface(X, Y, Z, cmap='inferno'), ax2.plot_surface(X, Y, Z, cmap='inferno')

        desc_trajectory_args, asc_trajectory_args = trajectory_args
        desc_trajectory_values, asc_trajectory_values = trajectory_values

        ax1.plot(
            desc_trajectory_args[:, 0], desc_trajectory_args[:, 1], desc_trajectory_values,
            color='green', linestyle='--', marker='*', label='Trajectory descending'
        )
        ax2.plot(
            asc_trajectory_args[:, 0], asc_trajectory_args[:, 1], asc_trajectory_values,
            color='blue', linestyle='-', marker='o', label='Trajectory ascending'
        )

        ax1.set_title('G(x) plot'), ax2.set_title('G(x) plot')
        ax1.set_xlabel('X'), ax2.set_xlabel('X')
        ax1.set_ylabel('Y'), ax2.set_ylabel('Y')
        ax1.set_zlabel('G(X, Y)'), ax2.set_zlabel('G(X, Y)')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--function', type=str, choices=['f', 'g'], default='f', help='Choose function f or g')
    parser.add_argument('-a', '--alpha', type=float, default=0.1, help='Insert learning rate')
    parser.add_argument('-l', '--limit', type=int, default=1000, help='Insert iterations limit')
    args = parser.parse_args()

    eps = 1e-10
    learning_rate = args.alpha
    iterations_limit = args.limit

    if args.function == 'f':
        point = [random.uniform(-4*np.pi + eps, 4*np.pi - eps)]
        func = Function_f()
    
    elif args.function == 'g':
        point = [
            random.uniform(-2 + eps, 2 - eps),
            random.uniform(-2 + eps, 2 - eps),
        ]
        func = Function_g()

    print("Starting point: ", point)

    desc_trajectory_args = GradientDescent(
        point, learning_rate, iterations_limit, func.gradient).grad_descent(func.isInDomain)
    asc_trajectory_args = GradientDescent(
        point, -learning_rate, iterations_limit, func.gradient).grad_descent(func.isInDomain)

    desc_trajectory_values = np.array([func.formula(coords) for coords in desc_trajectory_args])
    asc_trajectory_values = np.array([func.formula(coords) for coords in asc_trajectory_args])

    condition = 'g' if args.function == 'g' else 'f'

    print(
        f"Function {condition} has minimum ({desc_trajectory_args[-1]}, {desc_trajectory_values[-1]})"
    )
    
    print(
        f"Function {condition} has maximum ({asc_trajectory_args[-1]}, {asc_trajectory_values[-1]})"
    )

    trajectory_args = (desc_trajectory_args, asc_trajectory_args)
    trajectory_values = (desc_trajectory_values, asc_trajectory_values)

    plotter = DataVisualizer()

    if args.function == 'f':
        X = np.linspace(-4*np.pi, 4*np.pi)
        Y = func.formula(X)
        plotter.Visualize_2D(X, Y, trajectory_args, trajectory_values)

    elif args.function == 'g':
        X = np.linspace(-2, 2)
        Y = np.linspace(-2, 2)
        X, Y = np.meshgrid(X, Y)
        Z = func.formula([X, Y])
        plotter.Visualize_3D(X, Y, Z, trajectory_args, trajectory_values)
        # plotter.Visualize_3D_subplots(X, Y, Z, trajectory_args, trajectory_values)
