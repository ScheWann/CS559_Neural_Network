# 01-IDNUMBER-YOURLASTNAME.py

import numpy as np
import matplotlib.pyplot as plt

def step_function(v):
    return 1 if v >= 0 else 0

def neural_network(x, y):
    v1 = x - y + 1
    h1 = step_function(v1)
    v2 = -x - y + 1
    h2 = step_function(v2)
    v3 = -x
    h3 = step_function(v3)
    v_out = h1 + h2 - h3 - 1.5
    z = step_function(v_out)
    return z

def main():
    num_points = 1000
    points = np.random.uniform(-2, 2, size=(num_points, 2))

    blue_points = np.array([p for p in points if neural_network(p[0], p[1]) == 0])
    red_points = np.array([p for p in points if neural_network(p[0], p[1]) == 1])

    plt.figure(figsize=(8, 8))
    
    if blue_points.size > 0:
        plt.scatter(blue_points[:, 0], blue_points[:, 1], c='blue', label='Output = 0', s=10, alpha=0.7)
    if red_points.size > 0:
        plt.scatter(red_points[:, 0], red_points[:, 1], c='red', label='Output = 1', s=10, alpha=0.7)

    x_range = np.linspace(-2.2, 2.2, 100)
    
    # H1: y = x + 1
    plt.plot(x_range, x_range + 1, 'g--', linewidth=2, label='H1 (y=x+1)')

    # H2: y = -x + 1
    plt.plot(x_range, -x_range + 1, 'k-', linewidth=2.5, label='H2 (y=-x+1)')

    # H3: x = 0
    plt.axvline(0, color='k', linestyle='-', linewidth=2.5, label='H3 (x=0)')

    plt.title('Q2: Decision Region')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend(loc='upper left')
    plt.axis('equal')
    plt.xlim(-2.2, 2.2)
    plt.ylim(-2.2, 2.2)
    plt.show()

if __name__ == '__main__':
    main()