import pytest
import numpy as np

from configuration_interaction import CISD
from configuration_interaction.ci_helper import (
    state_printer,
    create_excited_states,
    create_reference_state,
)
from tests.helper import (
    create_singles_states,
    create_doubles_states,
    construct_one_body_density_matrix_brute_force,
    setup_hamiltonian_brute_force,
)

from quantum_systems import (
    GeneralOrbitalSystem,
    TwoDimensionalHarmonicOscillator,
    ODQD,
    construct_pyscf_system_ao,
)
from quantum_systems.system_helper import compute_particle_density


def test_setup(odho_ti_small):
    cisd = CISD(odho_ti_small, verbose=True)

    assert cisd.num_states == 66
    assert len(cisd.states) == cisd.num_states

    counter = 0
    for i in range(len(cisd.states)):
        if cisd.states[i, 0] > 0:
            counter += 1

    assert counter == cisd.num_states


def test_states_setup(odho_ti_small):
    cisd = CISD(odho_ti_small, verbose=True)

    n, l = cisd.n, cisd.l
    states_c = np.zeros_like(cisd.states)
    create_reference_state(n, l, states_c)
    index = create_singles_states(n, l, states_c, index=1)
    create_doubles_states(n, l, states_c, index=index)
    states_c = np.sort(states_c, axis=0)

    for cisd_state, state in zip(cisd.states, states_c):
        print(f"{state_printer(cisd_state)}\n{state_printer(state)}\n")

    np.testing.assert_allclose(cisd.states, states_c)


def test_slater_condon_hamiltonian(odho_ti_small):
    cisd = CISD(odho_ti_small, verbose=True)
    cisd.compute_ground_state()

    hamiltonian_b = np.zeros_like(cisd.hamiltonian)
    setup_hamiltonian_brute_force(
        hamiltonian_b,
        cisd.states,
        odho_ti_small.h,
        odho_ti_small.u,
        odho_ti_small.n,
        odho_ti_small.l,
    )

    np.testing.assert_allclose(hamiltonian_b, cisd.hamiltonian, atol=1e-7)


def test_slater_condon_density_matrix(odho_ti_small):
    cisd = CISD(odho_ti_small, verbose=True)
    cisd.compute_ground_state()

    for K in range(cisd.num_states):
        # Compare particle densities in order to implicitly compare one-body
        # density matrices.
        rho = cisd.compute_particle_density(K=K)
        rho_qp_b = np.zeros(
            (odho_ti_small.l, odho_ti_small.l), dtype=np.complex128
        )
        construct_one_body_density_matrix_brute_force(
            rho_qp_b, cisd.states, cisd.C[:, K]
        )
        rho_b = compute_particle_density(
            rho_qp_b, odho_ti_small.spf, odho_ti_small.spf.conj(), np
        )

        # Normalize particle densities
        rho_b = cisd.n * rho_b / np.trapz(rho_b, x=odho_ti_small.grid)
        rho = cisd.n * rho / np.trapz(rho, x=odho_ti_small.grid)

        np.testing.assert_allclose(rho_b, rho)


def test_large_basis():

    n = 2
    l = 33  # Force the use of two uint64 in a determinant

    tdho = GeneralOrbitalSystem(n, TwoDimensionalHarmonicOscillator(l, 10, 101))

    cisd = CISD(tdho, verbose=True).compute_ground_state()

    assert abs(cisd.energies[0] - 3.0094342497034936) < 1e-8


def test_one_body_expectation_value():
    n = 2
    l = 10

    odho = GeneralOrbitalSystem(n, ODQD(l, 11, 201))

    cisd = CISD(odho, verbose=True).compute_ground_state()

    assert abs(n - cisd.compute_one_body_expectation_value(odho.s)) < 1e-8
    assert (
        abs(cisd.compute_one_body_expectation_value(odho.dipole_moment[0]))
        < 1e-8
    )


def test_spin_projection():
    n = 2
    l = 20  # Force the use of two uint64 in a determinant

    tdho = GeneralOrbitalSystem(n, TwoDimensionalHarmonicOscillator(l, 10, 101))
    cisd = CISD(tdho, verbose=True, s=0).compute_ground_state()

    for K in range(cisd.num_states):
        # Test if the expectation value of the S_z-operator is zero for all
        # states. This should be the case when we've removed all the
        # determinants with a spin-projection number different from 0.
        np.testing.assert_allclose(
            cisd.compute_one_body_expectation_value(tdho.spin_z),
            0,
            atol=1e-12,
            rtol=1e-12,
        )


@pytest.mark.skip
def test_spin_squared():
    n = 2
    l = 20

    # tdho = GeneralOrbitalSystem(n, TwoDimensionalHarmonicOscillator(l, 10, 101))
    # cisd = CISD(tdho, verbose=True, s=None).compute_ground_state()

    # for K in range(10):
    #     # Test if the expectation value of the S^2-operator is zero for all
    #     # states. That is, check that we only include singlet states. This
    #     # should be the case when we've removed all the determinants with a
    #     # spin-projection number different from 0.
    #     np.testing.assert_allclose(
    #         cisd.compute_two_body_expectation_value(tdho.spin_2),
    #         0,
    #         atol=1e-12,
    #         rtol=1e-12,
    #     )

    he = construct_pyscf_system_ao("he")
    cisd = CISD(he, verbose=True, s=1).compute_ground_state()

    for K in range(cisd.num_states):
        print(cisd.allowed_dipole_transition(0, K))

    for K in range(cisd.num_states):
        print(
            (
                K,
                cisd.energies[K],
                cisd.compute_two_body_expectation_value(he.spin_2),
            )
        )
        # np.testing.assert_allclose(
        #     cisd.compute_two_body_expectation_value(he.spin_2),
        #     0,
        #     atol=1e-12,
        #     rtol=1e-12,
        # )
    assert False


def test_energy_expectation_values():
    n = 2
    l = 20

    tdho = GeneralOrbitalSystem(n, TwoDimensionalHarmonicOscillator(l, 10, 101))
    cisd = CISD(tdho, verbose=True, s=None).compute_ground_state()

    for K in range(10):
        E_K = cisd.compute_one_body_expectation_value(
            tdho.h, K=K
        ) + 0.5 * cisd.compute_two_body_expectation_value(tdho.u, K=K)
        np.testing.assert_allclose(
            cisd.energies[K],
            E_K,
        )

    he = construct_pyscf_system_ao("he")
    cisd = CISD(he, verbose=True, s=None).compute_ground_state()

    for K in range(10):
        E_K = cisd.compute_one_body_expectation_value(
            he.h, K=K
        ) + 0.5 * cisd.compute_two_body_expectation_value(he.u, K=K)
        np.testing.assert_allclose(
            cisd.energies[K],
            E_K,
        )


def test_hellman_feynman():
    from quantum_systems import construct_pyscf_system_rhf

    def compute_cisd_with_static_field(e_str, num_states=5):

        system = construct_pyscf_system_rhf(
            molecule="h 0.0 0.0 -0.7; h 0.0 0.0 0.7",
            basis="cc-pvdz",
            add_spin=False,
            anti_symmetrize=False,
        )

        # Add an electric field that is uniformly polarized in all spatial directions, i.e.,
        # the polarization vector is given by n=(1,1,1)/sqrt(3).
        system._basis_set.h = system.h + e_str * (
            system.position[0] + system.position[1] + system.position[2]
        ) / np.sqrt(3)

        system_gos = system.construct_general_orbital_system()

        cisd = CISD(system_gos, verbose=True).compute_ground_state()

        expec_r_cisd = np.zeros((3, num_states), dtype=np.complex128)
        for i in range(3):
            for I in range(num_states):
                expec_r_cisd[i, I] = cisd.compute_one_body_expectation_value(
                    system_gos.position[i], K=I
                )

        return cisd.energies, expec_r_cisd

    e_str = 0.01
    de = 0.001
    num_states = 5

    e_cisd, expec_r_cisd = compute_cisd_with_static_field(e_str, num_states)
    e_cisd_p_de, expec_r_cisd_p_de = compute_cisd_with_static_field(
        e_str + de, num_states
    )
    e_cisd_m_de, expec_r_cisd_m_de = compute_cisd_with_static_field(
        e_str - de, num_states
    )

    expec_r_fdm_cisd = np.zeros(num_states, dtype=np.complex128)
    diff_expec_r = np.zeros(num_states, dtype=np.complex128)

    for I in range(num_states):
        expec_r_fdm_cisd[I] = (e_cisd_p_de[I] - e_cisd_m_de[I]) / (2 * de)
        isotropic_expec_r = np.sum(expec_r_cisd[:, I]) / np.sqrt(3)
        diff_expec_r[I] = expec_r_fdm_cisd[I] - isotropic_expec_r

    assert abs(diff_expec_r[0]) < 1e-7
    assert abs(diff_expec_r[1]) < 1e-6
    assert abs(diff_expec_r[2]) < 1e-6
    assert abs(diff_expec_r[3]) < 1e-6
    assert abs(diff_expec_r[4]) < 1e-6
