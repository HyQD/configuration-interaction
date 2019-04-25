import numba
import numpy

from configuration_interaction.ci_helper import BITSTRING_SIZE


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
