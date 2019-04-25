import pytest

from configuration_interaction import CIS, CID, CISD
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


@pytest.fixture(params=[2, 3, 4])
def odho_ti(request):
    n = request.param
    l = 12

    grid = 10
    num_grid_points = 400

    odho = OneDimensionalHarmonicOscillator(n, l, grid, num_grid_points)
    odho.setup_system()

    return odho


@pytest.fixture(params=[CIS, CID, CISD])
def CI(request):
    return request.param
