import numpy as np

from configuration_interaction.ci_helper import (
    BITTYPE,
    BITSTRING_SIZE,
    popcount_64,
    count_state,
    occupied_index,
    get_index,
    get_double_index,
    state_diff,
    state_equality,
    compute_sign,
    create_particle,
    annihilate_particle,
    evaluate_one_body_overlap,
    evaluate_two_body_overlap,
)


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

    one = np.array([1, 0, 0]).astype(BITTYPE)
    assert get_double_index(one) == (-1, -1)

    one[0] |= np.uint32(0b10)
    assert get_double_index(one) == (0, 1)

    one[0] ^= np.uint32(0b1)
    assert get_double_index(one) == (-1, -1)

    one[0] |= np.uint32(0b10000)
    assert get_double_index(one) == (1, 4)

    one[0] |= np.uint32(0b1000000)
    assert get_double_index(one) == (1, 4)

    state = np.array([6]).astype(BITTYPE)
    assert get_double_index(state) == (1, 2)


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
