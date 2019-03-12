import numba
import numpy as np

BITTYPE = np.uint32
BITSTRING_SIZE = np.dtype(BITTYPE).itemsize * 8

# Const used by the Hamming weight algorithm
m_1 = 0x5555_5555_5555_5555
m_2 = 0x3333_3333_3333_3333
m_4 = 0x0F0F_0F0F_0F0F_0F0F
h_01 = 0x0101_0101_0101_0101


@numba.njit(cache=True)
def popcount_64(num):
    # Implementation of the Hamming weight algorithm shown here:
    # https://en.wikipedia.org/wiki/Hamming_weight#Efficient_implementation

    num -= (num >> 1) & m_1
    num = (num & m_2) + ((num >> 2) & m_2)
    num = (num + (num >> 4)) & m_4

    return (num * h_01) >> 56


@numba.njit(cache=True)
def get_diff_lists(states):
    diff_by_one_list = []
    diff_by_two_list = []

    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            diff = states[i] ^ states[j]

            num_bits = 0

            for elem in diff:
                num_bits += popcount_64(elem)

            if num_bits == 2:
                diff_by_one_list.append((i, j))
            elif num_bits == 4:
                diff_by_two_list.append((i, j))

    return diff_by_one_list, diff_by_two_list


@numba.njit(cache=True)
def create_reference_state(n, l, states):
    ref_index = 0

    for i in range(n):
        elem = i // BITSTRING_SIZE
        states[ref_index, elem] |= 1 << (i - elem * BITSTRING_SIZE)

    for i in range(ref_index + 1, len(states)):
        states[i] += states[ref_index]


@numba.njit(cache=True)
def create_singles_states(n, l, states):
    index = 1

    for i in range(n):
        elem_i = i // BITSTRING_SIZE
        for a in range(n, l):
            elem_a = a // BITSTRING_SIZE

            states[index, elem_i] ^= 1 << (i - elem_i * BITSTRING_SIZE)
            states[index, elem_a] |= 1 << (a - elem_a * BITSTRING_SIZE)

            index += 1


@numba.njit(cache=True)
def create_doubles_states(n, l, states):
    index = 1

    for i in range(n):
        elem_i = i // BITSTRING_SIZE
        for j in range(i + 1, n):
            elem_j = j // BITSTRING_SIZE
            for a in range(n, l):
                elem_a = a // BITSTRING_SIZE
                for b in range(a + 1, l):
                    elem_b = b // BITSTRING_SIZE

                    states[index, elem_i] ^= 1 << (i - elem_i * BITSTRING_SIZE)
                    states[index, elem_j] ^= 1 << (j - elem_j * BITSTRING_SIZE)
                    states[index, elem_a] |= 1 << (a - elem_a * BITSTRING_SIZE)
                    states[index, elem_b] |= 1 << (b - elem_b * BITSTRING_SIZE)

                    index += 1


@numba.njit(cache=True)
def compute_sign(state, p):
    elem_i = 0
    k = 0

    for i in range(p):

        if (i - elem_i * BITSTRING_SIZE) >= BITSTRING_SIZE:
            elem_i += 1

        k += (state[elem_i] >> (i - elem_i * BITSTRING_SIZE)) & 1

    return (-1) ** k


@numba.njit(cache=True)
def create_particle(state, p):
    elem_p = p // BITSTRING_SIZE

    sign = compute_sign(state, p)
    new_state = state.copy()

    new_state[elem_p] ^= 1 << (p - elem_p * BITSTRING_SIZE)

    if new_state[elem_p] & (1 << (p - elem_p * BITSTRING_SIZE)) == 0:
        sign = 0

    return new_state, sign


@numba.njit(cache=True)
def annihilate_particle(state, p):
    elem_p = p // BITSTRING_SIZE

    sign = compute_sign(state, p)
    new_state = state.copy()

    new_state[elem_p] ^= 1 << (p - elem_p * BITSTRING_SIZE)

    if new_state[elem_p] & (1 << (p - elem_p * BITSTRING_SIZE)) != 0:
        sign = 0

    return new_state, sign


@numba.njit(cache=True)
def evaluate_one_body_overlap(state_i, state_j, p, q):
    """Function evaluating the overlap

        O_{IJ} = <\Phi_I| c_{p}^{\dagger} c_{q} |\Phi_J>,

    that is, the overlap between two Slater determinants acted upon by a
    creation and an annihilation operator."""

    q_state, sign_q = annihilate_particle(state_j, q)

    if sign_q == 0:
        return 0

    p_state, sign_p = create_particle(q_state, p)

    if sign_p == 0:
        return 0

    return sign_p * sign_q
