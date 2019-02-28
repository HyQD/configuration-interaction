import pytest

from quantum_systems import OneDimensionalHarmonicOscillator


@pytest.fixture
def odho_ti_small():
    n = 20
    l = 40
    grid = 10
    num_grid_points = 400

    return OneDimensionalHarmonicOscillator(n, l, grid, num_grid_points)
