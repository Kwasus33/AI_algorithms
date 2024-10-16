# 331396
# A - 6, B - 9, C - 3

import random
import numpy as np
import matplotlib.pyplot as plt

class GradientDescent:
    def __init__(self, points, learning_rate, num_iterations, function_grad):
        self.points = points
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.function_grad = function_grad

    def grad_descent(self):
        trajectory = [self.points.copy()]
        for _ in range(self.num_iterations):
            self.points = self.points - self.learning_rate * self.function_grad(self.points)
            trajectory.append(self.points.copy())
        return trajectory

class Function_f:
    def __init__(self):
        pass
    def function_f(self, points):
        x = points
        return 6*x + 9*np.sin(x)

    def gradient_f(self, points):
        x = points
        return [6 - 9*np.cos(x)]

class Function_g:
    def __init__(self):
        pass

    def function_g(self, points):
        x, y = points
        return 3*x*y / (np.exp(x**2 + y**2))

    def gradient_g(self, points):
        x, y = points
        return [(3*y - 6*x**2*y) / (np.exp(x**2 + y**2)),
                (3*x - 6*x*y**2) / (np.exp(x**2 + y**2))]
    
class DataVisualizer:
    def __init__(self):
        pass
    
    def Visualize_2D(self, X, Y):
        plt.figure(figsize=(6, 10))
        plt.plot(X, Y)
        plt.title('F(x) plot')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
    
    def Visualize_3D(self, X, Y, Z):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(X, Y, Z, cmap='inferno')
        ax.set_title('G(x) plot')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('G(X, Y)')
        plt.show()


if __name__ == "__main__":

    eps = 1e-10
    points_f = [random.uniform(-4*np.pi + eps, 4*np.pi - eps)]
    points_g = [
        random.uniform(-2 + eps, 2 - eps), random.uniform(-2 + eps, 2 - eps)
        ]   
    
    func_f = Function_f()
    func_g = Function_g()

    trajectory_f = GradientDescent(points_f, 0.1, 1000, func_f.gradient_f)
    trajectory_g = GradientDescent(points_g, 0.1, 1000, func_g.gradient_g)
   
    plotter = DataVisualizer()
    
    # g_x = np.linspace(-2, 2)
    # g_y = np.linspace(-2, 2)
    # X, Y = np.meshgrid(g_x, g_y)
    # Z = func_g.function_g([X, Y])
    # plotter.Visualize_3D(X, Y, Z)

    X = np.linspace(-4*np.pi, 4*np.pi)
    Y = func_f.function_f(X)
    plotter.Visualize_2D(X, Y)

