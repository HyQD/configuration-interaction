import numpy as np

from configuration_interaction.ci_helper import (
    popcount_64,
    state_diff,
    state_equality,
    compute_sign,
    create_particle,
    annihilate_particle,
    evaluate_one_body_overlap,
    evaluate_two_body_overlap,
)


def test_popcount_64():
    zero = np.uint32(0b0)
    assert popcount_64(zero) == 0

    full = np.uint32(~0b0)
    assert popcount_64(full) == 32

    num = 0
    prev_num = 0
    set_bits = 0

    for i in np.random.randint(32, size=10000, dtype=np.uint32):
        num |= 1 << i

        if num != prev_num:
            set_bits += 1

        assert popcount_64(num) == set_bits

        prev_num = num


def test_state_diff():
    zero = np.array([0, 0]).astype(np.uint32)
    assert state_diff(zero, zero) == 0

    one = np.array([0b1, 0]).astype(np.uint32)
    assert state_diff(zero, one) == 1

    two = np.array([0b1, 0b1]).astype(np.uint32)
    assert state_diff(one, two) == 1
    assert state_diff(two, one) == 1
    assert state_diff(zero, two) == 2


def test_state_equality():
    zero = np.array([0, 0]).astype(np.uint32)
    assert state_equality(zero, zero)

    one = np.array([0b1, 0]).astype(np.uint32)
    assert state_equality(one, one)
    assert not state_equality(zero, one)

    two = np.array([0b1, 0b1]).astype(np.uint32)
    assert state_equality(two, two)
    assert not state_equality(one, two)
    assert not state_equality(two, one)
    assert not state_equality(zero, two)


def test_sign():
    zero = np.array([0]).astype(np.uint32)
    assert compute_sign(zero, 0) == 1

    one = np.array([1, 0]).astype(np.uint32)
    assert compute_sign(one, 0) == 1
    assert compute_sign(one, 10) == -1

    two = np.array([5]).astype(np.uint32)
    assert compute_sign(two, 0) == 1
    assert compute_sign(two, 1) == -1
    assert compute_sign(two, 2) == -1
    assert compute_sign(two, 3) == 1
    assert compute_sign(two, 4) == 1

    three = np.array([0b1001010]).astype(np.uint32)
    assert compute_sign(three, 0) == 1
    assert compute_sign(three, 1) == 1
    assert compute_sign(three, 2) == -1
    assert compute_sign(three, 3) == -1
    assert compute_sign(three, 4) == 1
    assert compute_sign(three, 5) == 1
    assert compute_sign(three, 6) == 1
    assert compute_sign(three, 7) == -1

    extended = np.array([0b11, 0b101]).astype(np.uint32)
    assert compute_sign(extended, 0) == 1
    assert compute_sign(extended, 1) == -1
    assert compute_sign(extended, 2) == 1
    assert compute_sign(extended, 10) == 1
    assert compute_sign(extended, 32) == 1
    assert compute_sign(extended, 33) == -1
    assert compute_sign(extended, 34) == -1
    assert compute_sign(extended, 35) == 1


def test_create_particle():
    det = np.array([0, 0]).astype(np.uint32)

    new_det, new_sign = create_particle(det, 0)
    assert new_sign == 1
    assert new_det[0] == 1

    new_det, new_sign = create_particle(new_det, 1)
    assert new_sign == -1
    assert new_det[0] == 0b11

    new_det, new_sign = create_particle(new_det, 32)
    assert new_sign == 1
    assert new_det[0] == 0b11
    assert new_det[1] == 0b1

    new_det, new_sign = create_particle(new_det, 32)
    assert new_sign == 0
    assert new_det[0] == 0b11
    assert new_det[1] == 0


def test_annihilate_particle():
    det = np.array([0b11, 0b101]).astype(np.uint32)

    new_det, new_sign = annihilate_particle(det, 0)
    assert new_sign == 1
    assert new_det[0] == 0b10

    new_det, new_sign = annihilate_particle(new_det, 0)
    assert new_sign == 0
    assert new_det[0] == 0b11

    new_det, new_sign = annihilate_particle(new_det, 1)
    assert new_sign == -1
    assert new_det[0] == 0b1

    new_det, new_sign = annihilate_particle(new_det, 32)
    assert new_sign == -1
    assert new_det[0] == 0b1
    assert new_det[1] == 0b100

    new_det, new_sign = annihilate_particle(new_det, 34)
    assert new_sign == -1
    assert new_det[0] == 0b1
    assert new_det[1] == 0b0


def test_one_body_overlap():
    phi_i = np.array([0b11, 0b101]).astype(np.uint32)
    phi_j = np.array([0b101, 0b101]).astype(np.uint32)
    assert evaluate_one_body_overlap(phi_i, phi_j, p=1, q=2) == 1
    assert evaluate_one_body_overlap(phi_i, phi_j, p=1, q=0) == 0

    phi_j = np.array([0b11, 0b11]).astype(np.uint32)
    assert evaluate_one_body_overlap(phi_i, phi_j, p=34, q=33) == 1

    phi_j = np.array([0b11, 0b101]).astype(np.uint32)
    assert evaluate_one_body_overlap(phi_i, phi_j, p=0, q=0) == 1
    assert evaluate_one_body_overlap(phi_i, phi_j, p=1, q=1) == 1
    assert evaluate_one_body_overlap(phi_i, phi_j, p=2, q=2) == 0


def test_two_body_overlap():
    phi_i = np.array([0b11, 0b101]).astype(np.uint32)
    phi_j = np.array([0b101, 0b101]).astype(np.uint32)

    assert evaluate_two_body_overlap(phi_i, phi_j, p=1, q=2, r=2, s=2) == 0
    assert evaluate_two_body_overlap(phi_i, phi_j, p=2, q=2, r=2, s=0) == 0
    assert evaluate_two_body_overlap(phi_i, phi_j, p=1, q=0, r=2, s=0) == 1
    assert evaluate_two_body_overlap(phi_i, phi_j, p=0, q=1, r=2, s=0) == -1
    assert evaluate_two_body_overlap(phi_i, phi_j, p=0, q=1, r=0, s=2) == 1
    assert evaluate_two_body_overlap(phi_i, phi_j, p=1, q=0, r=0, s=2) == -1
