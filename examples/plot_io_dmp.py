import matplotlib.pyplot as plt
from pytransform3d import plot_utils
import numpy as np
from movement_primitives.dmp import DMP, CouplingTermCartesianDistance
from movement_primitives.io import write_yaml, read_yaml

dt = 0.01

dmp = DMP(n_dims=6, execution_time=1.0, dt=dt, n_weights_per_dim=10, int_dt=0.0001)

T = np.linspace(0.0, 1.0, 101)
Y = np.empty((len(T), 6))
Y[:, 0] = np.cos(np.pi * T)
Y[:, 1] = np.sin(np.pi * T)
Y[:, 2] = np.sin(2 * np.pi * T)
Y[:, 3] = np.cos(np.pi * T)
Y[:, 4] = np.sin(np.pi * T)
Y[:, 5] = 0.5 + np.sin(2 * np.pi * T)
dmp.imitate(T, Y)
write_yaml("dmp.yaml", dmp, verbose=0)
dmp2 = read_yaml("dmp.yaml")

plot_utils.make_3d_axis(2.0, 111)

plt.plot(Y[:, 0], Y[:, 1], Y[:, 2], label="Demo 1")
plt.plot(Y[:, 3], Y[:, 4], Y[:, 5], label="Demo 2")

dmp2.configure(start_y=Y[0], goal_y=Y[-1])
T, Y = dmp2.open_loop()
plt.plot(Y[:, 0], Y[:, 1], Y[:, 2], label="Reproduction 1")
plt.plot(Y[:, 3], Y[:, 4], Y[:, 5], label="Reproduction 2")

dmp2.configure(start_y=Y[0], goal_y=Y[-1])
ct = CouplingTermCartesianDistance(desired_distance=1.0, lf=(1.0, 0.0), k=0.1)
T, Y = dmp2.open_loop(coupling_term=ct)
plt.plot(Y[:, 0], Y[:, 1], Y[:, 2], label="Coupled 1")
plt.plot(Y[:, 3], Y[:, 4], Y[:, 5], label="Coupled 2")

plt.legend(loc="best")
plt.show()
