import matplotlib.pyplot as plt
import numpy as np
import pytransform3d.plot_utils as ppu
from movement_primitives.promp import ProMP

import pandas as pd

demons = np.load('examples/data/aligned_dtw_data_list.npy')
n_demonstrations, n_steps, n_task_dims = demons.shape
data = pd.read_csv('examples/data/combined.csv')
data_len,_ = data.shape
y_demo = data.values
T = np.arange(0, n_steps, 1)

execution_time = 1.0
start_y = y_demo[0,:]
goal_y = y_demo[-1,:]
random_state = np.random.RandomState(42)

promp = ProMP(n_dims=n_task_dims)
promp.imitate([T] * n_demonstrations, demons)

ax = ppu.make_3d_axis(1000)
T, Y = promp.open_loop()
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
ax4.plot(T, np.gradient(Y[:, 0]) / promp.dt_)
ax5.plot(T, np.gradient(Y[:, 1]) / promp.dt_)
ax6.plot(T, np.gradient(Y[:, 2]) / promp.dt_)

for _ in range(3):
    start_y_random = start_y + 100 * random_state.randn(3)
    goal_y_random = goal_y + 100 * random_state.randn(3)
    promp.configure(start_y=start_y_random, goal_y=goal_y_random)
    T, Y = promp.open_loop()
    ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], c="b")
    ax.scatter(start_y_random[0], start_y_random[1], start_y_random[2], c="b")
    ax1.plot(T, Y[:, 0], label="Random")
    ax2.plot(T, Y[:, 1], label="Random")
    ax3.plot(T, Y[:, 2], label="Random")
    ax4.plot(T, np.gradient(Y[:, 0]) / promp.dt_)
    ax5.plot(T, np.gradient(Y[:, 1]) / promp.dt_)
    ax6.plot(T, np.gradient(Y[:, 2]) / promp.dt_)
ax1.legend()
plt.tight_layout()
plt.show()
