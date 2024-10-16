# 331396
# A - 6, B - 9, C - 3

import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

class GradientDescent:
    def __init__(self, points, learning_rate, num_iterations, function_grad):
        self.points = np.array(points)
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.function_grad = function_grad

    def grad_descent(self):
        trajectory = [self.points.copy()]
        for _ in range(self.num_iterations):
            self.points = self.points - self.learning_rate * self.function_grad(self.points)
            trajectory.append(self.points.copy())
        return np.array(trajectory)

class Function_f:
    def __init__(self):
        pass

    def formula(self, x):
        return 6*x + 9*np.sin(x)

    def gradient(self, x):
        # x is already a NumPy array so is the return value
        return 6 - 9*np.cos(x)

class Function_g:
    def __init__(self):
        pass

    def formula(self, points):
        x, y = points
        return 3*x*y / (np.exp(x**2 + y**2))

    def gradient(self, points):
        x, y = points
        return np.array([
            (3*y - 6*x**2*y) / (np.exp(x**2 + y**2)),
            (3*x - 6*x*y**2) / (np.exp(x**2 + y**2))
        ])
    
class DataVisualizer:
    def __init__(self):
        pass
    
    def Visualize_2D(self, X, Y, trajectory_args, trajectory_values):
        plt.figure(figsize=(6, 10))
        
        plt.title('F(x) plot')
        plt.xlabel('X')
        plt.ylabel('Y')
        
        plt.plot(X, Y)
        plt.plot(
            trajectory_args, trajectory_values,
            color='red', linestyle='-', marker='o', label='Trajectory'
        )

        plt.legend()
        plt.grid(True)
        
        plt.show()
    
    def Visualize_3D(self, X, Y, Z, trajectory_args, trajectory_values):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')

        ax.set_title('G(x) plot')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('G(X, Y)')

        ax.plot(
            trajectory_args[:, 0], trajectory_args[:, 1], trajectory_values,
            color='green', linestyle='-', marker='o', label='Trajectory'
        )
        ax.plot_surface(X, Y, Z, cmap='inferno')

        plt.legend()
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--function', type=str, choices=['f', 'g'], default='f', help='Choose function f or g')
    args = parser.parse_args()

    eps = 1e-10

    if args.function == 'f':
        points = [random.uniform(-4*np.pi + eps, 4*np.pi - eps)]
        func = Function_f()
    
    elif args.function == 'g':
        points = [
            random.uniform(-2 + eps, 2 - eps),
            random.uniform(-2 + eps, 2 - eps)
        ]
        func = Function_g()

    trajectory_args = GradientDescent(points, 0.1, 10000, func.gradient).grad_descent()
    trajectory_values = np.array([func.formula(coords) for coords in trajectory_args])

    plotter = DataVisualizer()

    if args.function == 'f':
        X = np.linspace(-4*np.pi, 4*np.pi)
        Y = func.formula(X)
        plotter.Visualize_2D(X, Y, trajectory_args, trajectory_values)

    elif args.function == 'g':
        g_x = np.linspace(-2, 2)
        g_y = np.linspace(-2, 2)
        X, Y = np.meshgrid(g_x, g_y)
        Z = func.formula([X, Y])
        plotter.Visualize_3D(X, Y, Z, trajectory_args, trajectory_values)
