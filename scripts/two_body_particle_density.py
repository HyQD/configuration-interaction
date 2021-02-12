import numpy as np
import matplotlib.pyplot as plt

from quantum_systems import GeneralOrbitalSystem, ODQD
from configuration_interaction import CISD


n = 2
l = 12
grid_length = 10
num_grid_points = 2001


system = GeneralOrbitalSystem(n, ODQD(l, grid_length, num_grid_points))


cisd = CISD(system, verbose=True).compute_ground_state()


plt.plot(system.grid, cisd.compute_particle_density().real)


X, Y = np.meshgrid(system.grid, system.grid)

plt.figure()
plt.contourf(X, Y, cisd.compute_two_body_particle_density().real)
plt.show()
