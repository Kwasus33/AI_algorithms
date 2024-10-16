# 331396
# A - 6, B - 9, C - 3


import math
import random
# import numpy as np


def function_f(points):
    x = points
    return 6*x + 9*math.sin(x)


def gradient_f(points):
    x = points
    return [6 - 9*math.cos(x)]


def function_g(points):
    x, y = points
    return 3*x*y / (math.exp(x**2 + y**2))


def gradient_g(points):
    x, y = points
    return [(3*y - 6*x**2*y) / (math.exp(x**2 + y**2)),
            (3*x - 6*x*y**2) / (math.exp(x**2 + y**2))]


def grad_descent(points, learning_rate, num_iterations, function_grad):
    trajectory = [points.copy()]
    for _ in range(num_iterations):
        points = points - learning_rate * function_grad(points)
        trajectory.append(points.copy())
    return trajectory


if __name__ == "__main__":

    f_section = [-4*math.pi, 4*math.pi]
    g_section = [(-2, 2), (-2, 2)]

    points_f = [random.uniform(f_section[0], f_section[1])]
    points_g = [
        random.uniform(g_section[0][0], g_section[0][1]),
        random.uniform(g_section[1][0], g_section[1][1])
        ]   
    
    func_f = function_f
    func_g = function_g

    grad_descent(points_f, 0.1, 1000, func_f)
    grad_descent(points_g, 0.1, 1000, func_g)
   