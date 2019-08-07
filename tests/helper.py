import numba
import numpy

from configuration_interaction.ci_helper import (
    BITSTRING_SIZE,
    evaluate_one_body_overlap,
    evaluate_two_body_overlap,
    occupied_index,
    state_diff,
    get_index,
    compute_sign,
)


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
def construct_one_body_density_matrix_brute_force(rho_qp, states, c):
    num_states = len(states)
    l = len(rho_qp)

    for I in range(num_states):
        state_I = states[I]
        for J in range(num_states):
            state_J = states[J]
            diff = state_diff(state_I, state_J)

            if diff > 2:
                continue

            for p in range(l):
                for q in range(l):
                    rho_qp[q, p] += (
                        c[I].conjugate()
                        * c[J]
                        * evaluate_one_body_overlap(state_I, state_J, p=p, q=q)
                    )


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
def create_triples_states(n, l, states, index):
    for i in range(n):
        elem_i = i // BITSTRING_SIZE
        for j in range(i + 1, n):
            elem_j = j // BITSTRING_SIZE
            for k in range(j + 1, n):
                elem_k = k // BITSTRING_SIZE
                for a in range(n, l):
                    elem_a = a // BITSTRING_SIZE
                    for b in range(a + 1, l):
                        elem_b = b // BITSTRING_SIZE
                        for c in range(b + 1, l):
                            elem_c = c // BITSTRING_SIZE

                            states[index, elem_i] ^= 1 << (
                                i - elem_i * BITSTRING_SIZE
                            )
                            states[index, elem_j] ^= 1 << (
                                j - elem_j * BITSTRING_SIZE
                            )
                            states[index, elem_k] ^= 1 << (
                                k - elem_k * BITSTRING_SIZE
                            )
                            states[index, elem_a] |= 1 << (
                                a - elem_a * BITSTRING_SIZE
                            )
                            states[index, elem_b] |= 1 << (
                                b - elem_b * BITSTRING_SIZE
                            )
                            states[index, elem_c] |= 1 << (
                                c - elem_c * BITSTRING_SIZE
                            )

                            index += 1

    return index


@numba.njit(cache=True, nogil=True, fastmath=True)
def create_quadruples_states(n, l, states, index):
    for i in range(n):
        elem_i = i // BITSTRING_SIZE
        for j in range(i + 1, n):
            elem_j = j // BITSTRING_SIZE
            for k in range(j + 1, n):
                elem_k = k // BITSTRING_SIZE
                for _l in range(k + 1, n):
                    elem__l = _l // BITSTRING_SIZE
                    for a in range(n, l):
                        elem_a = a // BITSTRING_SIZE
                        for b in range(a + 1, l):
                            elem_b = b // BITSTRING_SIZE
                            for c in range(b + 1, l):
                                elem_c = c // BITSTRING_SIZE
                                for d in range(c + 1, l):
                                    elem_d = d // BITSTRING_SIZE

                                    states[index, elem_i] ^= 1 << (
                                        i - elem_i * BITSTRING_SIZE
                                    )
                                    states[index, elem_j] ^= 1 << (
                                        j - elem_j * BITSTRING_SIZE
                                    )
                                    states[index, elem_k] ^= 1 << (
                                        k - elem_k * BITSTRING_SIZE
                                    )
                                    states[index, elem__l] ^= 1 << (
                                        _l - elem__l * BITSTRING_SIZE
                                    )
                                    states[index, elem_a] |= 1 << (
                                        a - elem_a * BITSTRING_SIZE
                                    )
                                    states[index, elem_b] |= 1 << (
                                        b - elem_b * BITSTRING_SIZE
                                    )
                                    states[index, elem_c] |= 1 << (
                                        c - elem_c * BITSTRING_SIZE
                                    )
                                    states[index, elem_d] |= 1 << (
                                        d - elem_d * BITSTRING_SIZE
                                    )

                                    index += 1

    return index


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
