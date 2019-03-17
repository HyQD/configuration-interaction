import numba
import numpy as np

BITTYPE = np.uint64
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
def count_state(state):
    counter = 0

    for elem in state:
        counter += popcount_64(elem)

    return counter


@numba.njit(cache=True, nogil=True, fastmath=True)
def occupied_index(state, p):
    elem_p = p // BITSTRING_SIZE

    return (state[elem_p] & (1 << (p - elem_p * BITSTRING_SIZE))) != 0


@numba.njit(cache=True, nogil=True, fastmath=True)
def get_index(state):
    """Computes the index of the first set bit in state. That is, if state is

        state = 0b100 = 4,

    then get_index(state) returns

        get_index(state) = 2.

    This is done by checking if the lowermost bit is set and then rolling the
    bits one position to the right and counting places until a set bit is
    encountered. Returns -1 if there are no set bits."""

    index = 0

    for elem_p in range(len(state)):
        for p in range(BITSTRING_SIZE):
            if (state[elem_p] >> p) & 0b1 != 0:
                return index

            index += 1

    return -1


@numba.njit(cache=True, nogil=True, fastmath=True)
def get_double_index(state):
    """Computes the indices of the two first set bits in state. That is, if
    state is given by

        state = 0b110 = 6,

    then get_double_index(state) returns

        get_double_index(state) = (1, 2)."""
    first_index = 0
    second_index = 0

    first_add = 1
    second_add = 1

    for elem_p in range(len(state)):
        for p in range(BITSTRING_SIZE):
            check = (state[elem_p] >> p) & 0b1 != 0

            if check and first_add == 1:
                first_add = 0
            elif check and second_add == 1:
                return first_index, second_index

            first_index += first_add
            second_index += second_add

    return -1, -1


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
def setup_hamiltonian_brute_force(hamiltonian, states, h, u, n, l):
    num_states = len(states)

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
                hamiltonian[J, I] = val.conjugate()


@numba.njit(parallel=True, nogil=True, fastmath=True)
def setup_hamiltonian(hamiltonian, states, h, u, n, l):
    
    """
    The Hamiltonian should be split in a one- and two-body component 

        H = H1 + H2 
        H1 = <psi|h_{pq} c_p^\dagger c_q|psi>
        H2 = <psi|u_{pq,rs} c_p^\dagger c_q^\dagger c_s c_r|psi> 

    This is advantageous when doing TD-CI since we do not have to re-compute H2 
    (assuming static orbitals) if the two-body is time-independent (which it usually is). 
    """
    num_states = len(states)

    for I in range(num_states):
        state_I = states[I]

        val = complex(0)

        for i in range(l):
            if not occupied_index(state_I, i):
                continue

            val += h[i, i]
            for j in range(i + 1, l):
                if not occupied_index(state_I, j):
                    continue

                val += u[i, j, i, j]

        hamiltonian[I, I] += val

    for I in numba.prange(num_states):
        state_I = states[I]

        for J in range(I + 1, num_states):
            state_J = states[J]
            diff = state_diff(state_I, state_J)

            if diff > 4:
                continue

            val = complex(0)

            if diff == 2:
                val += diff_by_one_hamiltonian(state_I, state_J, h, u, n, l)
            elif diff == 4:
                val += diff_by_two_hamiltonian(state_I, state_J, h, u, n, l)

            hamiltonian[I, J] += val
            hamiltonian[J, I] += val.conjugate()


@numba.njit(cache=True, nogil=True, fastmath=True)
def diff_by_one_hamiltonian(state_I, state_J, h, u, n, l):
    diff = state_I ^ state_J

    # Index m in state_I, removed from state_J
    m = get_index(state_I & diff)
    sign_m = compute_sign(state_I, m)

    # Index p in state_J, not in state_I
    p = get_index(state_J & diff)
    sign_p = compute_sign(state_J, p)

    sign = sign_m * sign_p

    val = sign * h[m, p]

    for i in range(l):
        if not occupied_index(state_I, i):
            continue

        val += sign * u[m, i, p, i]

    return val


@numba.njit(cache=True, nogil=True, fastmath=True)
def diff_by_two_hamiltonian(state_I, state_J, h, u, n, l):
    diff = state_I ^ state_J

    # Index m, n in state_I, removed from state_J
    m, n = get_double_index(state_I & diff)
    sign_m = compute_sign(state_I, m)
    sign_n = compute_sign(state_I, n)

    # Index p, q in state_J, not in state_I
    p, q = get_double_index(state_J & diff)
    sign_p = compute_sign(state_J, p)
    sign_q = compute_sign(state_J, q)

    sign = sign_m * sign_n * sign_p * sign_q

    return sign * u[m, n, p, q]


@numba.njit(cache=True)
def construct_one_body_density_matrix(rho_qp, states, c):
    num_states = len(states)
    l = len(rho_qp)

    for p in range(l):
        for q in range(l):
            val = 0

            for I in range(num_states):
                state_I = states[I]
                for J in range(num_states):
                    state_J = states[J]

                    val += (
                        c[I].conjugate()
                        * c[J]
                        * evaluate_one_body_overlap(state_I, state_J, p=p, q=q)
                    )

            rho_qp[q, p] = val


def compute_particle_density(rho_qp, spf, np):
    rho = np.zeros(spf.shape[1:], dtype=spf.dtype)
    spf_slice = slice(0, spf.shape[0])

    for _i in np.ndindex(rho.shape):
        i = (spf_slice, *_i)
        rho[_i] += np.dot(spf[i].conj(), np.dot(rho_qp, spf[i]))

    return rho
