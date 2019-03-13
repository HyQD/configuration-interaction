import numba
import numpy as np

BITTYPE = np.uint32
BITSTRING_SIZE = np.dtype(BITTYPE).itemsize * 8

# Const used by the Hamming weight algorithm
m_1 = 0x5555_5555_5555_5555
m_2 = 0x3333_3333_3333_3333
m_4 = 0x0F0F_0F0F_0F0F_0F0F
h_01 = 0x0101_0101_0101_0101


@numba.njit(cache=True, nogil=True, fastmath=True)
def popcount_64(num):
    # Implementation of the Hamming weight algorithm shown here:
    # https://en.wikipedia.org/wiki/Hamming_weight#Efficient_implementation

    num -= (num >> 1) & m_1
    num = (num & m_2) + ((num >> 2) & m_2)
    num = (num + (num >> 4)) & m_4

    return (num * h_01) >> 56


@numba.njit(cache=True, nogil=True, fastmath=True)
def state_diff(state_i, state_j):
    """Function computing the difference between state_i and state_j. This is
    done by computing

        diff := state_i XOR state_j,

    and counting all non-zero bits. As the states are arrays with bits, we
    iterate through each element in diff and perform popcount_64 to count the
    bits, which we accumulate."""

    diff = state_i ^ state_j

    num_bits = 0
    for elem in diff:
        num_bits += popcount_64(elem)

    return num_bits


@numba.njit(cache=True, nogil=True, fastmath=True)
def state_equality(state_i, state_j):
    """Function checking if state_i == state_j."""
    return state_diff(state_i, state_j) == 0


@numba.njit(cache=True, nogil=True, fastmath=True)
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


@numba.njit(cache=True, nogil=True, fastmath=True)
def create_reference_state(n, l, states):
    ref_index = 0

    for i in range(n):
        elem = i // BITSTRING_SIZE
        states[ref_index, elem] |= 1 << (i - elem * BITSTRING_SIZE)

    for i in range(ref_index + 1, len(states)):
        states[i] += states[ref_index]


@numba.njit(cache=True, nogil=True, fastmath=True)
def create_singles_states(n, l, states, index):
    for i in range(n):
        elem_i = i // BITSTRING_SIZE
        for a in range(n, l):
            elem_a = a // BITSTRING_SIZE

            states[index, elem_i] ^= 1 << (i - elem_i * BITSTRING_SIZE)
            states[index, elem_a] |= 1 << (a - elem_a * BITSTRING_SIZE)

            index += 1

    return index


@numba.njit(cache=True, nogil=True, fastmath=True)
def create_doubles_states(n, l, states, index):
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

    return index


@numba.njit(cache=True, nogil=True, fastmath=True)
def compute_sign(state, p):
    elem_i = 0
    k = 0

    for i in range(p):

        if (i - elem_i * BITSTRING_SIZE) >= BITSTRING_SIZE:
            elem_i += 1

        k += (state[elem_i] >> (i - elem_i * BITSTRING_SIZE)) & 1

    return (-1) ** k


@numba.njit(cache=True, nogil=True, fastmath=True)
def create_particle(state, p):
    elem_p = p // BITSTRING_SIZE

    sign = compute_sign(state, p)
    new_state = state.copy()

    new_state[elem_p] ^= 1 << (p - elem_p * BITSTRING_SIZE)

    if new_state[elem_p] & (1 << (p - elem_p * BITSTRING_SIZE)) == 0:
        sign = 0

    return new_state, sign


@numba.njit(cache=True, nogil=True, fastmath=True)
def annihilate_particle(state, p):
    elem_p = p // BITSTRING_SIZE

    sign = compute_sign(state, p)
    new_state = state.copy()

    new_state[elem_p] ^= 1 << (p - elem_p * BITSTRING_SIZE)

    if new_state[elem_p] & (1 << (p - elem_p * BITSTRING_SIZE)) != 0:
        sign = 0

    return new_state, sign


@numba.njit(cache=True, nogil=True, fastmath=True)
def evaluate_one_body_overlap(state_i, state_j, p, q):
    r"""Function evaluating the overlap

        O_{IJ} = <\Phi_I| c_{p}^{\dagger} c_{q} |\Phi_J>,

    that is, the overlap between two Slater determinants acted upon by a
    creation and an annihilation operator."""

    state_q, sign_q = annihilate_particle(state_j, q)

    if sign_q == 0:
        return 0

    state_p, sign_p = create_particle(state_q, p)

    if sign_p == 0:
        return 0

    # Check if state_i == state_p
    if not state_equality(state_i, state_p):
        return 0

    return sign_p * sign_q


@numba.njit(cache=True, nogil=True, fastmath=True)
def evaluate_two_body_overlap(state_i, state_j, p, q, r, s):
    r"""Fnction evaluating the overlap

        O_{IJ} = <\Phi_I| c_{p}^{\dagger} c_{q}^{\dagger} c_{s} c_{r} |\Phi_J>,

    that is, the overlap between two Slater determinants acted upon by a pair of
    creation and annihilation operators from the second quantized form of a two
    body operator.

    Note especially the ordering of the ordering of the annihilation operators.
    """

    state_r, sign_r = annihilate_particle(state_j, r)

    if sign_r == 0:
        return 0

    state_s, sign_s = annihilate_particle(state_r, s)

    if sign_s == 0:
        return 0

    state_q, sign_q = create_particle(state_s, q)

    if sign_q == 0:
        return 0

    state_p, sign_p = create_particle(state_q, p)

    if sign_p == 0:
        return 0

    # Check if state_i == state_p
    if not state_equality(state_i, state_p):
        return 0

    return sign_p * sign_q * sign_s * sign_r


@numba.njit(parallel=True, nogil=True, fastmath=True)
# @numba.njit(cache=True, fastmath=True)
def setup_hamiltonian_brute_force(hamiltonian, states, h, u, n, l):
    num_states = len(states)

    # for I in range(num_states):
    for I in numba.prange(num_states):
        state_I = states[I]
        for J in range(I, num_states):
            state_J = states[J]

            val = complex(0)

            for p in range(l):
                for q in range(l):
                    sign = evaluate_one_body_overlap(state_I, state_J, p, q)
                    val += sign * h[p, q]

                    for r in range(l):
                        for s in range(l):
                            sign = evaluate_two_body_overlap(
                                state_I, state_J, p, q, r, s
                            )

                            if sign == 0:
                                continue

                            val += 0.25 * sign * u[p, q, r, s]

            hamiltonian[I, J] = val

            if I != J:
                hamiltonian[J, I] = np.conj(val)


@numba.njit(parallel=True, nogil=True, fastmath=True)
def construct_one_body_density_matrix(rho_qp, states, c):
    num_states = len(states)

    for I in numba.prange(num_states):
        state_I = states[I]
        for J in range(num_states):
            state_J = states[J]

            for p in range(l):
                for q in range(l):
                    rho_qp[q, p] = np.dot(
                        c, np.conj(c)
                    ) * evaluate_one_body_overlap(state_I, state_J, p=p, q=q)
