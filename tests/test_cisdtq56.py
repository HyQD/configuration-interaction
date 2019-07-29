import pytest

from quantum_systems import TwoDimensionalHarmonicOscillator
from configuration_interaction import get_ci_class


def get_tdho(omega):
    n = 6
    l = 12

    _tdho = TwoDimensionalHarmonicOscillator(
        n, l, radius_length=10, num_grid_points=101, omega=omega
    )
    _tdho.setup_system(verbose=True, cast_to_complex=False)

    return _tdho


def test_cisdtq56():

    for omega, energy in [
        (0.01, 1.009487),
        (0.1, 4.149560),
        (0.5, 12.897229),
        (1.0, 21.420589),
    ]:
        tdho = get_tdho(omega)
        cisdtq56 = get_ci_class("CISDTQ56")(tdho, verbose=True)
        cisdtq56.compute_ground_state()

        assert abs(energy - cisdtq56.energies[0]) < 1e-5
