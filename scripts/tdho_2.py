import numpy as np

from configuration_interaction import CISD
from quantum_systems import (
    TwoDimensionalHarmonicOscillator,
    GeneralOrbitalSystem,
)


n = 2
l = 10

tdho = GeneralOrbitalSystem(n, TwoDimensionalHarmonicOscillator(l, 5, 201))

fci = CISD(tdho, verbose=True)
fci.compute_ground_state()
