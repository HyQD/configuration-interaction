import numpy as np

from configuration_interaction.ci_helper import (
    popcount_64,
    compute_sign,
    create_particle,
    annihilate_particle,
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
