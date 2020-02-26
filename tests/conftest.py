import pytest

from configuration_interaction import CIS, CID, CISD
from quantum_systems import ODQD, GeneralOrbitalSystem


@pytest.fixture
def odho_ti_small():
    n = 2
    l = 6
    grid = 10
    num_grid_points = 400

    odho = GeneralOrbitalSystem(n, ODQD(l, grid, num_grid_points))

    return odho


@pytest.fixture(params=[2, 3, 4])
def odho_ti(request):
    n = request.param
    l = 6

    grid = 10
    num_grid_points = 400

    odho = GeneralOrbitalSystem(n, ODQD(l, grid, num_grid_points))

    return odho


@pytest.fixture(params=[CIS, CID, CISD])
def CI(request):
    return request.param
