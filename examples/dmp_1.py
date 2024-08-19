"""
========================
Obstacle Avoidance in 3D
========================

Plots a DMP in 3D that goes through a point obstacle without obstacle
avoidance. Then we start use the same DMP to start from multiple random
start positions with an activated coupling term for obstacle avoidance.
"""
print(__doc__)


import matplotlib.pyplot as plt
import numpy as np
import pytransform3d.plot_utils as ppu
from movement_primitives.dmp import DMP, CouplingTermObstacleAvoidance3D

import pandas as pd

data = pd.read_csv('examples/data/combined.csv')
data_len,_ = data.shape
y_demo = data.values
T = np.arange(0, data_len, 1)

execution_time = data_len
start_y = y_demo[0,:]
goal_y = y_demo[-1,:]
random_state = np.random.RandomState(42)

dmp = DMP(n_dims=y_demo.shape[1], execution_time=execution_time, dt=1, n_weights_per_dim=100)
dmp.imitate(T, y_demo)

dmp.configure(start_y=start_y, goal_y=goal_y)

ax = ppu.make_3d_axis(1000)
T, Y = dmp.open_loop()
ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], c="g")
ax.scatter(start_y[0], start_y[1], start_y[2], c="r")
ax.scatter(goal_y[0], goal_y[1], goal_y[2], c="g")

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
ax1.plot(T, Y[:, 0], label="Demo")
ax2.plot(T, Y[:, 1], label="Demo")
ax3.plot(T, Y[:, 2], label="Demo")
ax4.plot(T, np.gradient(Y[:, 0]) / dmp.dt_)
ax5.plot(T, np.gradient(Y[:, 1]) / dmp.dt_)
ax6.plot(T, np.gradient(Y[:, 2]) / dmp.dt_)

for _ in range(5):
    start_y_random = start_y + 40 * np.sin(_) * (-1) ** _
    goal_y_random = goal_y + 50 * np.cos(_) * (-1) ** _
    dmp.configure(start_y=start_y_random, goal_y=goal_y_random)
    T, Y = dmp.open_loop()
    ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], c="b")
    ax.scatter(start_y_random[0], start_y_random[1], start_y_random[2], c="b")
    ax1.plot(T, Y[:, 0], label="Random" + str(_))
    ax2.plot(T, Y[:, 1], label="Random" + str(_))
    ax3.plot(T, Y[:, 2], label="Random" + str(_))
    ax4.plot(T, np.gradient(Y[:, 0]) / dmp.dt_)
    ax5.plot(T, np.gradient(Y[:, 1]) / dmp.dt_)
    ax6.plot(T, np.gradient(Y[:, 2]) / dmp.dt_)
ax1.legend()
plt.tight_layout()
plt.show()
