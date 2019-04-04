import pytest

from quantum_systems import OneDimensionalHarmonicOscillator


@pytest.fixture
def odho_ti_small():
    n = 2
    l = 12
    grid = 10
    num_grid_points = 400

    odho = OneDimensionalHarmonicOscillator(n, l, grid, num_grid_points)
    odho.setup_system()

    return odho
