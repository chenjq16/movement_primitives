import matplotlib.pyplot as plt
import numpy as np
import pytransform3d.plot_utils as ppu
from movement_primitives.dmp import DMP
from movement_primitives.data import generate_minimum_jerk

import pandas as pd

data_right = pd.read_csv('examples/data/right.csv')
data_len_right,_ = data_right.shape
y_demo_right = data_right.values
T_right = np.arange(0, data_len_right, 1)

data_left = pd.read_csv('examples/data/left.csv')
data_len_left,_ = data_left.shape
y_demo_left = data_left.values
T_left = np.arange(0, data_len_left, 1)

execution_time_right = data_len_right
start_y_right = y_demo_right[0,:]
goal_y_right = y_demo_right[-1,:]
random_state = np.random.RandomState(42)

execution_time_left = data_len_left
start_y_left = y_demo_left[0,:]
goal_y_left = y_demo_left[-1,:]

dmp_right = DMP(n_dims=y_demo_right.shape[1], execution_time=execution_time_right, dt=1, n_weights_per_dim=100)
dmp_right.imitate(T_right, y_demo_right)

dmp_left = DMP(n_dims=y_demo_left.shape[1], execution_time=execution_time_left, dt=1, n_weights_per_dim=100)
dmp_left.imitate(T_left, y_demo_left)

dmp_right.configure(start_y=start_y_right, goal_y=goal_y_right)
dmp_left.configure(start_y=start_y_left, goal_y=goal_y_left)

ax = ppu.make_3d_axis(1000)
T_right, Y_right = dmp_right.open_loop()
ax.plot(Y_right[:, 0], Y_right[:, 1], Y_right[:, 2], c="g")
ax.scatter(start_y_right[0], start_y_right[1], start_y_right[2], c="r")
ax.scatter(goal_y_right[0], goal_y_right[1], goal_y_right[2], c="g")

T_left, Y_left = dmp_left.open_loop()
ax.plot(Y_left[:, 0], Y_left[:, 1], Y_left[:, 2], c="purple")
ax.scatter(start_y_left[0], start_y_left[1], start_y_left[2], c="y")
ax.scatter(goal_y_left[0], goal_y_left[1], goal_y_left[2], c="purple")

X, Xd, Xdd = generate_minimum_jerk(goal_y_right, start_y_left, execution_time=30, dt=1)

T = np.arange(0, data_len_right+data_len_left+2+31, 1)
Y = np.concatenate((Y_right, X, Y_left), axis=0)

plt.figure(figsize=(15, 10))
ax1 = plt.subplot(231)
ax1.set_title("Dimension 1")
ax1.set_xlabel("Time")
ax1.set_ylabel("Position")
ax2 = plt.subplot(232)
ax2.set_title("Dimension 2")
ax2.set_xlabel("Time")
ax2.set_ylabel("Position")
ax3 = plt.subplot(233)
ax3.set_title("Dimension 3")
ax3.set_xlabel("Time")
ax3.set_ylabel("Position")
ax4 = plt.subplot(234)
ax4.set_title("Dimension 1")
ax4.set_xlabel("Time")
ax4.set_ylabel("Velocity")
ax5 = plt.subplot(235)
ax5.set_title("Dimension 2")
ax5.set_xlabel("Time")
ax5.set_ylabel("Velocity")
ax6 = plt.subplot(236)
ax6.set_title("Dimension 3")
ax6.set_xlabel("Time")
ax6.set_ylabel("Velocity")
# ax1.plot(T_right, Y_right[:, 0], label="Demo")
# ax2.plot(T_right, Y_right[:, 1], label="Demo")
# ax3.plot(T_right, Y_right[:, 2], label="Demo")
# ax4.plot(T_right, np.gradient(Y_right[:, 0]) / dmp_right.dt_)
# ax5.plot(T_right, np.gradient(Y_right[:, 1]) / dmp_right.dt_)
# ax6.plot(T_right, np.gradient(Y_right[:, 2]) / dmp_right.dt_)
ax1.plot(T, Y[:, 0], label="Demo")
ax2.plot(T, Y[:, 1], label="Demo")
ax3.plot(T, Y[:, 2], label="Demo")
ax4.plot(T, np.gradient(Y[:, 0]) / dmp_right.dt_)
ax5.plot(T, np.gradient(Y[:, 1]) / dmp_right.dt_)
ax6.plot(T, np.gradient(Y[:, 2]) / dmp_right.dt_)

for _ in range(5):
    start_y_random_right = start_y_right + 40 * np.sin(_) * (-1) ** _
    goal_y_random_left = goal_y_left + 50 * np.cos(_) * (-1) ** _

    offset_random = ((goal_y_random_left - goal_y_left) + (start_y_random_right - start_y_right)) / 2

    goal_y_random_right = goal_y_right + offset_random
    start_y_random_left = start_y_left + offset_random

    dmp_right.configure(start_y=start_y_random_right, goal_y=goal_y_random_right)
    T_right, Y_right = dmp_right.open_loop()
    ax.plot(Y_right[:, 0], Y_right[:, 1], Y_right[:, 2], c="b")
    ax.scatter(start_y_random_right[0], start_y_random_right[1], start_y_random_right[2], c="b")

    dmp_left.configure(start_y=start_y_random_left, goal_y=goal_y_random_left)
    T_left, Y_left = dmp_left.open_loop()
    ax.plot(Y_left[:, 0], Y_left[:, 1], Y_left[:, 2], c="orange")
    ax.scatter(start_y_random_left[0], start_y_random_left[1], start_y_random_left[2], c="orange")

    X, Xd, Xdd = generate_minimum_jerk(goal_y_random_right, start_y_random_left, execution_time=30, dt=1)

    T = np.arange(0, data_len_right+data_len_left+2+31, 1)
    Y = np.concatenate((Y_right, X, Y_left), axis=0)

    ax1.plot(T, Y[:, 0], label="Random" + str(_))
    ax2.plot(T, Y[:, 1], label="Random" + str(_))
    ax3.plot(T, Y[:, 2], label="Random" + str(_))
    ax4.plot(T, np.gradient(Y[:, 0]) / dmp_right.dt_)
    ax5.plot(T, np.gradient(Y[:, 1]) / dmp_right.dt_)
    ax6.plot(T, np.gradient(Y[:, 2]) / dmp_right.dt_)

ax1.legend()
plt.tight_layout()
plt.show()
