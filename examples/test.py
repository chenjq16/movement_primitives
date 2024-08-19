import matplotlib.pyplot as plt
from movement_primitives.data import generate_minimum_jerk
import pytransform3d.plot_utils as ppu

start_point = [-4.829480483183144770e+02,8.760403679196624580e+02,2.744144167104050211e+02]
end_point = [1.225381952479193330e+02,-2.345317071149563048e+02,1.093845527134714985e+03]
X, Xd, Xdd = generate_minimum_jerk(start_point, end_point)
plt.figure()
plt.subplot(311)
plt.ylabel("$x$")
plt.plot(X[:, 0])
plt.subplot(312)
plt.ylabel("$\dot{x}$")
plt.plot(Xd[:, 0])
plt.subplot(313)
plt.xlabel("$t$")
plt.ylabel("$\ddot{x}$")
plt.plot(Xdd[:, 0])

plt.figure()
plt.subplot(311)
plt.ylabel("$y$")
plt.plot(X[:, 1])
plt.subplot(312)
plt.ylabel("$\dot{y}$")
plt.plot(Xd[:, 1])
plt.subplot(313)
plt.xlabel("$t$")
plt.ylabel("$\ddot{y}$")
plt.plot(Xdd[:, 1])

plt.figure()
ax = ppu.make_3d_axis(1000)
ax.plot(X[:, 0], X[:, 1], X[:, 2], c="g")
ax.scatter(start_point[0], start_point[1], start_point[2], c="r")
ax.scatter(end_point[0], end_point[1], end_point[2], c="g")

plt.show()
