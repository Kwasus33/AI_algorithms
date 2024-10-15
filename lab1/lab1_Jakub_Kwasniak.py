# 331396
# A - 6, B - 9, C - 3


import math


def function_f(x):
    return 6*x + 9*math.sin(x)


def gradient_f(x):
    return 6 - 9*math.cos(x)


def function_g(x, y):
    return 3*x*y / (math.exp(x**2 + y**2))


def gradient_g(x, y):
    return 