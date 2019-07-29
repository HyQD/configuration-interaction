import numba
import numpy as np
import warnings

BITTYPE = np.uint64
BITSTRING_SIZE = np.dtype(BITTYPE).itemsize * 8


ORDER = {"S": 1, "D": 2, "T": 3, "Q": 4, "5": 5, "6": 6}


def num_states(n, m, order):
    if order <= 0:
        return 1

    num = (
        num_states(n, m, order - 1)
        * (n - (order - 1))
        * (m - (order - 1))
        // order ** 2
    )

    return num


def state_printer(state):
    """Function creating a string of the bit-representation of a state"""

    s = ""
    for elem in state:
        s += bin(elem)[2:].zfill(BITSTRING_SIZE)

    return s


# Const used by the Hamming weight algorithm
m_1 = 0x5555_5555_5555_5555
m_2 = 0x3333_3333_3333_3333
m_4 = 0x0F0F_0F0F_0F0F_0F0F
h_01 = 0x0101_0101_0101_0101


@numba.njit(cache=True, nogil=True, fastmath=True)
def popcount_64(num):
    """Implementation of the Hamming weight algorithm shown here:
    https://en.wikipedia.org/wiki/Hamming_weight#Efficient_implementation
    """
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
    encountered. Returns -1 if there are no set bits.
    """
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

        get_double_index(state) = (1, 2).
    """
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
    bits, which we accumulate.
    """
    diff = state_i ^ state_j

    num_bits = 0
    for elem in diff:
        num_bits += popcount_64(elem)

    return num_bits


@numba.njit(cache=True, nogil=True, fastmath=True)
def state_equality(state_i, state_j):
    """Function checking if state_i == state_j."""
    return state_diff(state_i, state_j) == 0


# Create mask selecting only spin-up or spin-down. We treat all even indices as
# spin-up whereas odd yields spin-down. The byte 0xaa = 0b10101010 (odd indices
# starting from index zero) repeated at all byte positions is used for the down
# mask. For up we have 0x55 = 0b01010101 yielding the even indices.
DOWN_MASK = sum((0xAA << i * 8) for i in range(np.dtype(BITTYPE).itemsize))
UP_MASK = sum((0x55 << i * 8) for i in range(np.dtype(BITTYPE).itemsize))


def compute_spin_projection_eigenvalue(state):
    """Function computing the eigenvalue of the spin-projection operator S_z.
    See "Molecular Electronic-Structure Theory" by T. Helgaker et al, equation
    2.4.25. Note that we treat even indices as spin-up and odd indices as
    spin-down.

    Parameters
    ----------
    state : np.ndarray
        A state array with each bit representing a set spin-orbital in a Slater
        determinant.

    Returns
    -------
    int
        The eigenvalue of the spin-projection operator.
    """
    num_spin_up = count_state(state & UP_MASK)
    num_spin_down = count_state(state & DOWN_MASK)

    return 0.5 * (num_spin_up - num_spin_down)


@numba.njit(cache=True, nogil=True, fastmath=True)
def create_reference_state(n, l, states):
    ref_index = 0

    for i in range(n):
        elem = i // BITSTRING_SIZE
        states[ref_index, elem] |= 1 << (i - elem * BITSTRING_SIZE)

    for i in range(ref_index + 1, len(states)):
        states[i] += states[ref_index]


def create_excited_states(n, l, states, index, order):
    """Driver function to create arbitrary excited states from reference
    states. The caller is responsible for pointing to the correct start index
    of the first excited state, and allocating enough reference states in the
    states array.

    Parameters
    ----------
    n : int
        Number of occupied particles, i.e., set bits in the states arrays.
    l : int
        Number of basis states, i.e., the number of bits used in the states
        arrays.
    states : np.ndarray
        Two-dimensional array acting as the number representation of the CI
        Slater determinant basis.
    index : int
        The index to the desired first excited state of the given order.
    order : int
        The order of the excitation, e.g.,
            order == 1 -> Singles excitations,
            order == 2 -> Doubles excitations,
            order == 3 -> Triples excitations.

    Returns
    -------
    index : int
        The number of the last excited state of the given order. For
        combinations of excited states, e.g., CISD, this will be the index of
        the first excited state in the next order.
    """
    if order > n:
        warning = (
            f"Order ({order}) is greater than the number of occupied "
            + f"particles ({n}). No excitation is done."
        )
        warnings.warn(warning)

        return index

    o_remove = np.zeros(order, dtype=int)
    v_insert = np.zeros(order, dtype=int)

    return _create_excited_states(
        n, l, states, index, order, o_remove, v_insert
    )


@numba.njit(cache=True, nogil=True, fastmath=True)
def _create_excited_states(n, l, states, index, order, o_remove, v_insert):
    if order == 0:
        _excite_state(states[index], o_remove, v_insert)
        return index + 1

    i_start = 0 if len(o_remove) == order else o_remove[order] + 1
    a_start = n if len(v_insert) == order else v_insert[order] + 1

    for i in range(i_start, n):
        o_remove[order - 1] = i
        for a in range(a_start, l):
            v_insert[order - 1] = a

            index = _create_excited_states(
                n, l, states, index, order - 1, o_remove, v_insert
            )

    return index


@numba.njit(cache=True, nogil=True, fastmath=True)
def _excite_state(state, o_remove, v_insert):
    for i, a in zip(o_remove, v_insert):
        elem_i = i // BITSTRING_SIZE
        elem_a = a // BITSTRING_SIZE

        state[elem_i] ^= 1 << (i - elem_i * BITSTRING_SIZE)
        state[elem_a] |= 1 << (a - elem_a * BITSTRING_SIZE)


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
    creation and an annihilation operator.
    """
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
    r"""Function evaluating the overlap

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
def setup_one_body_hamiltonian(hamiltonian, states, h, n, l):
    num_states = len(states)

    for I in range(num_states):
        state_I = states[I]

        val = complex(0)

        for i in range(l):
            if not occupied_index(state_I, i):
                continue

            val += h[i, i]

        hamiltonian[I, I] += val

    for I in numba.prange(num_states):
        state_I = states[I]

        for J in range(I + 1, num_states):
            state_J = states[J]
            diff = state_diff(state_I, state_J)

            if diff != 2:
                continue

            val = diff_by_one_slater_condon_one_body(state_I, state_J, h, n, l)

            hamiltonian[I, J] += val
            hamiltonian[J, I] += val.conjugate()


@numba.njit(cache=True, nogil=True, fastmath=True)
def diff_by_one_slater_condon_one_body(state_I, state_J, h, n, l):
    diff = state_I ^ state_J

    # Index m in state_I, removed from state_J
    m = get_index(state_I & diff)
    sign_m = compute_sign(state_I, m)

    # Index p in state_J, not in state_I
    p = get_index(state_J & diff)
    sign_p = compute_sign(state_J, p)

    sign = sign_m * sign_p

    return sign * h[m, p]


@numba.njit(parallel=True, nogil=True, fastmath=True)
def setup_two_body_hamiltonian(hamiltonian, states, u, n, l):
    num_states = len(states)

    for I in range(num_states):
        state_I = states[I]

        val = complex(0)

        for i in range(l):
            if not occupied_index(state_I, i):
                continue

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
                val += diff_by_one_slater_condon_two_body(
                    state_I, state_J, u, n, l
                )
            elif diff == 4:
                val += diff_by_two_slater_condon_two_body(
                    state_I, state_J, u, n, l
                )

            hamiltonian[I, J] += val
            hamiltonian[J, I] += val.conjugate()


@numba.njit(cache=True, nogil=True, fastmath=True)
def diff_by_one_slater_condon_two_body(state_I, state_J, u, n, l):
    diff = state_I ^ state_J

    # Index m in state_I, removed from state_J
    m = get_index(state_I & diff)
    sign_m = compute_sign(state_I, m)

    # Index p in state_J, not in state_I
    p = get_index(state_J & diff)
    sign_p = compute_sign(state_J, p)

    sign = sign_m * sign_p

    val = complex(0)

    for i in range(l):
        if not occupied_index(state_I, i):
            continue

        val += sign * u[m, i, p, i]

    return val


@numba.njit(cache=True, nogil=True, fastmath=True)
def diff_by_two_slater_condon_two_body(state_I, state_J, u, n, l):
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

    for I in range(num_states):
        state_I = states[I]

        for p in range(l):
            if not occupied_index(state_I, p):
                continue

            rho_qp[p, p] += c[I] * c[I].conjugate()

    for I in range(num_states):
        state_I = states[I]

        for J in range(num_states):
            if I == J:
                continue

            state_J = states[J]
            diff = state_diff(state_I, state_J)

            if diff != 2:
                continue

            diff_state = state_I ^ state_J

            # Index m in state_I, removed from state_J
            m = get_index(state_I & diff_state)
            sign_m = compute_sign(state_I, m)

            # Index p in state_J, not in state_I
            p = get_index(state_J & diff_state)
            sign_p = compute_sign(state_J, p)

            sign = sign_m * sign_p

            rho_qp[p, m] += sign * c[I].conjugate() * c[J]


def compute_particle_density(rho_qp, spf, np):
    rho = np.zeros(spf.shape[1:], dtype=spf.dtype)
    spf_slice = slice(0, spf.shape[0])

    for _i in np.ndindex(rho.shape):
        i = (spf_slice, *_i)
        rho[_i] += np.dot(spf[i].conj(), np.dot(rho_qp, spf[i]))

    return rho
