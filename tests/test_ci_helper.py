import pytest
import numpy as np

from configuration_interaction.ci_helper import (
    BITTYPE,
    BITSTRING_SIZE,
    num_states,
    popcount_64,
    count_state,
    occupied_index,
    get_index,
    state_diff,
    state_equality,
    compute_sign,
    create_particle,
    annihilate_particle,
    evaluate_one_body_overlap,
    evaluate_two_body_overlap,
    construct_one_body_density_matrix,
    setup_one_body_hamiltonian,
    setup_two_body_hamiltonian,
)
from tests.helper import (
    setup_hamiltonian,
    construct_one_body_density_matrix_brute_force,
    get_double_index,
)
from quantum_systems import CustomSystem, RandomSystem


NUM_SINGLES_STATES = lambda n, m: n * m
NUM_DOUBLES_STATES = (
    lambda n, m: NUM_SINGLES_STATES(n, m) * (n - 1) // 2 * (m - 1) // 2
)
NUM_TRIPLES_STATES = (
    lambda n, m: NUM_DOUBLES_STATES(n, m) * (n - 2) // 3 * (m - 2) // 3
)
NUM_QUADRUPLES_STATES = (
    lambda n, m: NUM_TRIPLES_STATES(n, m) * (n - 3) // 4 * (m - 3) // 4
)


@pytest.fixture
def nl_num_states():
    n = 4
    l = 20

    return n, l


def test_num_states(nl_num_states):
    assert NUM_SINGLES_STATES(*nl_num_states) == num_states(
        *nl_num_states, order=1
    )
    assert NUM_DOUBLES_STATES(*nl_num_states) == num_states(
        *nl_num_states, order=2
    )
    assert NUM_TRIPLES_STATES(*nl_num_states) == num_states(
        *nl_num_states, order=3
    )
    assert NUM_QUADRUPLES_STATES(*nl_num_states) == num_states(
        *nl_num_states, order=4
    )


def test_num_singles_states(nl_num_states):
    n, l = nl_num_states
    m = l - n

    num = 0
    for i in range(n):
        for a in range(n, l):
            num += 1

    assert num == NUM_SINGLES_STATES(n, m)


def test_num_doubles_states(nl_num_states):
    n, l = nl_num_states
    m = l - n

    num = 0
    for i in range(n):
        for j in range(i + 1, n):
            for a in range(n, l):
                for b in range(a + 1, l):
                    num += 1

    assert num == NUM_DOUBLES_STATES(n, m)


def test_num_triples_states(nl_num_states):
    n, l = nl_num_states
    m = l - n

    num = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for a in range(n, l):
                    for b in range(a + 1, l):
                        for c in range(b + 1, l):
                            num += 1

    assert num == NUM_TRIPLES_STATES(n, m)


def test_num_quadruples_states(nl_num_states):
    n, l = nl_num_states
    m = l - n

    num = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for _l in range(k + 1, n):
                    for a in range(n, l):
                        for b in range(a + 1, l):
                            for c in range(b + 1, l):
                                for d in range(c + 1, l):
                                    num += 1

    assert num == NUM_QUADRUPLES_STATES(n, m)


def test_popcount_64():
    zero = BITTYPE(0b0)
    assert popcount_64(zero) == 0

    full = BITTYPE(~0b0)
    assert popcount_64(full) == BITSTRING_SIZE

    num = BITTYPE(0)
    prev_num = BITTYPE(0)
    set_bits = 0

    for i in np.random.randint(BITSTRING_SIZE, size=10000, dtype=BITTYPE):
        num |= BITTYPE(1 << np.uint32(i))

        if num != prev_num:
            set_bits += 1

        assert popcount_64(num) == set_bits

        prev_num = num


def test_count_state():
    state = np.array([0, 0, 0]).astype(BITTYPE)

    num_set_indices = np.random.randint(BITSTRING_SIZE * len(state))
    indices = np.random.choice(
        np.arange(BITSTRING_SIZE * len(state)),
        size=num_set_indices,
        replace=False,
    )

    for p in indices:
        elem_p = p // BITSTRING_SIZE

        state[elem_p] |= BITTYPE(1 << (p - elem_p * BITSTRING_SIZE))

    counter = 0
    for elem in state:
        counter += popcount_64(elem)

    assert counter == num_set_indices
    assert counter == count_state(state)
    assert count_state(state) == num_set_indices


def test_occupied_index():
    state = np.array([0, 0, 0]).astype(BITTYPE)

    num_set_indices = np.random.randint(BITSTRING_SIZE * len(state))
    indices = np.random.choice(
        np.arange(BITSTRING_SIZE * len(state)),
        size=num_set_indices,
        replace=False,
    )

    for p in indices:
        elem_p = p // BITSTRING_SIZE

        state[elem_p] |= BITTYPE(1 << (p - elem_p * BITSTRING_SIZE))

    counter = 0
    for elem in state:
        counter += popcount_64(elem)

    assert counter == num_set_indices

    for p in range(BITSTRING_SIZE * len(state)):
        occ = occupied_index(state, p)

        assert occ if p in indices else not occ


def test_get_index():
    zero = np.array([0, 0, 0, 0]).astype(BITTYPE)
    assert get_index(zero) == -1

    one = np.array([1, 0, 0]).astype(BITTYPE)
    assert get_index(one) == 0

    one[0] |= np.uint32(0b10)
    assert get_index(one) == 0

    one[0] ^= np.uint32(0b1)
    assert get_index(one) == 1


def test_get_double_index():
    zero = np.array([0, 0, 0, 0]).astype(BITTYPE)
    assert get_double_index(zero) == (-1, -1)
    p = get_index(zero, index_num=0)
    q = get_index(zero, index_num=1)
    assert get_double_index(zero) == (p, q)

    one = np.array([1, 0, 0]).astype(BITTYPE)
    assert get_double_index(one) == (-1, -1)
    p = get_index(one, index_num=0)
    q = get_index(one, index_num=1)
    assert get_double_index(one)[0] != p
    assert get_double_index(one)[1] == q

    one[0] |= np.uint32(0b10)
    assert get_double_index(one) == (0, 1)
    p = get_index(one, index_num=0)
    q = get_index(one, index_num=1)
    assert get_double_index(one) == (p, q)

    one[0] ^= np.uint32(0b1)
    assert get_double_index(one) == (-1, -1)
    p = get_index(one, index_num=0)
    q = get_index(one, index_num=1)
    assert get_double_index(one)[0] != p
    assert get_double_index(one)[1] == q

    one[0] |= np.uint32(0b10000)
    assert get_double_index(one) == (1, 4)
    p = get_index(one, index_num=0)
    q = get_index(one, index_num=1)
    assert get_double_index(one) == (p, q)

    one[0] |= np.uint32(0b1000000)
    assert get_double_index(one) == (1, 4)
    p = get_index(one, index_num=0)
    q = get_index(one, index_num=1)
    assert get_double_index(one) == (p, q)

    state = np.array([6]).astype(BITTYPE)
    assert get_double_index(state) == (1, 2)
    p = get_index(state, index_num=0)
    q = get_index(state, index_num=1)
    assert get_double_index(state) == (p, q)

    state = np.array([1, 1]).astype(BITTYPE)
    assert get_double_index(state) == (0, 64)
    p = get_index(state, index_num=0)
    q = get_index(state, index_num=1)
    assert get_double_index(state) == (p, q)

    state = np.array([1, 2]).astype(BITTYPE)
    assert get_double_index(state) == (0, 65)
    p = get_index(state, index_num=0)
    q = get_index(state, index_num=1)
    assert get_double_index(state) == (p, q)


def test_state_diff():
    zero = np.array([0, 0]).astype(BITTYPE)
    assert state_diff(zero, zero) == 0

    one = np.array([0b1, 0]).astype(BITTYPE)
    assert state_diff(zero, one) == 1

    two = np.array([0b1, 0b1]).astype(BITTYPE)
    assert state_diff(one, two) == 1
    assert state_diff(two, one) == 1
    assert state_diff(zero, two) == 2


def test_state_equality():
    zero = np.array([0, 0]).astype(BITTYPE)
    assert state_equality(zero, zero)

    one = np.array([0b1, 0]).astype(BITTYPE)
    assert state_equality(one, one)
    assert not state_equality(zero, one)

    two = np.array([0b1, 0b1]).astype(BITTYPE)
    assert state_equality(two, two)
    assert not state_equality(one, two)
    assert not state_equality(two, one)
    assert not state_equality(zero, two)


def test_sign():
    zero = np.array([0]).astype(BITTYPE)
    assert compute_sign(zero, 0) == 1

    one = np.array([1, 0]).astype(BITTYPE)
    assert compute_sign(one, 0) == 1
    assert compute_sign(one, 10) == -1

    two = np.array([5]).astype(BITTYPE)
    assert compute_sign(two, 0) == 1
    assert compute_sign(two, 1) == -1
    assert compute_sign(two, 2) == -1
    assert compute_sign(two, 3) == 1
    assert compute_sign(two, 4) == 1

    three = np.array([0b1001010]).astype(BITTYPE)
    assert compute_sign(three, 0) == 1
    assert compute_sign(three, 1) == 1
    assert compute_sign(three, 2) == -1
    assert compute_sign(three, 3) == -1
    assert compute_sign(three, 4) == 1
    assert compute_sign(three, 5) == 1
    assert compute_sign(three, 6) == 1
    assert compute_sign(three, 7) == -1

    extended = np.array([0b11, 0b101]).astype(BITTYPE)
    assert compute_sign(extended, 0) == 1
    assert compute_sign(extended, 1) == -1
    assert compute_sign(extended, 2) == 1
    assert compute_sign(extended, 10) == 1
    assert compute_sign(extended, BITSTRING_SIZE) == 1
    assert compute_sign(extended, BITSTRING_SIZE + 1) == -1
    assert compute_sign(extended, BITSTRING_SIZE + 2) == -1
    assert compute_sign(extended, BITSTRING_SIZE + 3) == 1


def test_create_particle():
    det = np.array([0, 0]).astype(BITTYPE)

    new_det, new_sign = create_particle(det, 0)
    assert new_sign == 1
    assert new_det[0] == 1

    new_det, new_sign = create_particle(new_det, 1)
    assert new_sign == -1
    assert new_det[0] == 0b11

    new_det, new_sign = create_particle(new_det, BITSTRING_SIZE)
    assert new_sign == 1
    assert new_det[0] == 0b11
    assert new_det[1] == 0b1

    new_det, new_sign = create_particle(new_det, BITSTRING_SIZE)
    assert new_sign == 0
    assert new_det[0] == 0b11
    assert new_det[1] == 0


def test_annihilate_particle():
    det = np.array([0b11, 0b101]).astype(BITTYPE)

    new_det, new_sign = annihilate_particle(det, 0)
    assert new_sign == 1
    assert new_det[0] == 0b10

    new_det, new_sign = annihilate_particle(new_det, 0)
    assert new_sign == 0
    assert new_det[0] == 0b11

    new_det, new_sign = annihilate_particle(new_det, 1)
    assert new_sign == -1
    assert new_det[0] == 0b1

    new_det, new_sign = annihilate_particle(new_det, BITSTRING_SIZE)
    assert new_sign == -1
    assert new_det[0] == 0b1
    assert new_det[1] == 0b100

    new_det, new_sign = annihilate_particle(new_det, BITSTRING_SIZE + 2)
    assert new_sign == -1
    assert new_det[0] == 0b1
    assert new_det[1] == 0b0


def test_one_body_overlap():
    phi_i = np.array([0b11, 0b101]).astype(BITTYPE)
    phi_j = np.array([0b101, 0b101]).astype(BITTYPE)
    assert evaluate_one_body_overlap(phi_i, phi_j, p=1, q=2) == 1
    assert evaluate_one_body_overlap(phi_i, phi_j, p=1, q=0) == 0

    phi_j = np.array([0b11, 0b11]).astype(BITTYPE)
    assert (
        evaluate_one_body_overlap(
            phi_i, phi_j, p=BITSTRING_SIZE + 2, q=BITSTRING_SIZE + 1
        )
        == 1
    )

    phi_j = np.array([0b11, 0b101]).astype(BITTYPE)
    assert evaluate_one_body_overlap(phi_i, phi_j, p=0, q=0) == 1
    assert evaluate_one_body_overlap(phi_i, phi_j, p=1, q=1) == 1
    assert evaluate_one_body_overlap(phi_i, phi_j, p=2, q=2) == 0


def test_two_body_overlap():
    phi_i = np.array([0b11, 0b101]).astype(BITTYPE)
    phi_j = np.array([0b101, 0b101]).astype(BITTYPE)

    assert evaluate_two_body_overlap(phi_i, phi_j, p=1, q=2, r=2, s=2) == 0
    assert evaluate_two_body_overlap(phi_i, phi_j, p=2, q=2, r=2, s=0) == 0
    assert evaluate_two_body_overlap(phi_i, phi_j, p=1, q=0, r=2, s=0) == 1
    assert evaluate_two_body_overlap(phi_i, phi_j, p=0, q=1, r=2, s=0) == -1
    assert evaluate_two_body_overlap(phi_i, phi_j, p=0, q=1, r=0, s=2) == 1
    assert evaluate_two_body_overlap(phi_i, phi_j, p=1, q=0, r=0, s=2) == -1


def test_hamiltonian_setup():
    n = 2
    l = 12

    rs = RandomSystem(n, l)
    rs.setup_system(add_spin=True, anti_symmetrize=True)

    from configuration_interaction import CISD

    cisd = CISD(rs)

    orig_hamiltonian = np.zeros(
        (cisd.num_states, cisd.num_states), dtype=np.complex128
    )
    new_hamiltonian = np.zeros_like(orig_hamiltonian)

    setup_hamiltonian(orig_hamiltonian, cisd.states, rs.h, rs.u, rs.n, rs.l)
    setup_one_body_hamiltonian(new_hamiltonian, cisd.states, rs.h, rs.n, rs.l)
    setup_two_body_hamiltonian(new_hamiltonian, cisd.states, rs.u, rs.n, rs.l)

    np.testing.assert_allclose(orig_hamiltonian, new_hamiltonian)


def test_construct_one_body_density_matrices(odho_ti_small, CI):
    ci = CI(odho_ti_small, verbose=True)

    ci.compute_ground_state()

    rho_b = np.zeros((odho_ti_small.l, odho_ti_small.l), dtype=np.complex128)
    rho = np.zeros((odho_ti_small.l, odho_ti_small.l), dtype=np.complex128)

    for K in range(ci.num_states):
        construct_one_body_density_matrix_brute_force(
            rho_b, ci.states, ci.C[:, K]
        )
        construct_one_body_density_matrix(rho, ci.states, ci.C[:, K])

        np.testing.assert_allclose(rho_b, rho, atol=1e-7)


def test_construct_one_body_density_matrices_random(CI):
    n = 2
    l = 12

    cs = CustomSystem(n, l)

    ci = CI(cs, verbose=True)
    ci._C = np.random.random(
        (ci.num_states, ci.num_states)
    ) + 1j * np.random.random((ci.num_states, ci.num_states))

    rho_b = np.zeros((cs.l, cs.l), dtype=np.complex128)
    rho = np.zeros((cs.l, cs.l), dtype=np.complex128)

    for K in range(ci.num_states):
        construct_one_body_density_matrix_brute_force(
            rho_b, ci.states, ci.C[:, K]
        )
        construct_one_body_density_matrix(rho, ci.states, ci.C[:, K])

        np.testing.assert_allclose(rho_b, rho)
