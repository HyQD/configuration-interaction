import numpy as np
import matplotlib.pyplot as plt

from quantum_systems import TwoDimensionalHarmonicOscillator
from configuration_interaction import get_ci_class

n = 6
l = 12
omega = 0.28

tdho = TwoDimensionalHarmonicOscillator(
    n, l, radius_length=10, num_grid_points=201, omega=omega
)
# tdho.setup_system(verbose=True, cast_to_complex=False)
tdho.setup_system(verbose=True, cast_to_complex=True)

CISDTQ56 = get_ci_class("CISDTQ56")

cisdtq56 = CISDTQ56(tdho, verbose=True)
cisdtq56.spin_reduce_states()
cisdtq56.compute_ground_state()
