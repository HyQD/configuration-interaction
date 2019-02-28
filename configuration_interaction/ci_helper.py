import numba

# Const used by the Hamming weight algorithm
m_1 = 0x5555_5555_5555_5555
m_2 = 0x3333_3333_3333_3333
m_4 = 0x0F0F_0F0F_0F0F_0F0F
h_01 = 0x0101_0101_0101_0101


@numba.njit(cache=True)
def popcount_32(num):
    # Implementation of the Hamming weight algorithm shown here:
    # https://en.wikipedia.org/wiki/Hamming_weight#Efficient_implementation

    num -= (num >> 1) & m_1
    num = (num & m_2) + ((num >> 2) & m_2)
    num = (num + (num >> 4)) & m_4

    return (num * h_01) >> 56

    while num > 0:
        num &= num - 1
        count += 1

    return count


@numba.njit(cache=True)
def create_reference_state(n, l, states):
    ref_index = 0

    for i in range(n):
        elem = i // 32
        states[ref_index, elem] |= 1 << (i - elem * 32)

    for i in range(ref_index + 1, len(states)):
        states[i] += states[ref_index]


@numba.njit(cache=True)
def create_doubles_states(n, l, states, index):
    for i in range(n):
        elem_i = i // 32
        for j in range(i + 1, n):
            elem_j = j // 32
            for a in range(n, l):
                elem_a = a // 32
                for b in range(a + 1, l):
                    elem_b = b // 32

                    states[index, elem_i] ^= 1 << (i - elem_i * 32)
                    states[index, elem_j] ^= 1 << (j - elem_j * 32)
                    states[index, elem_a] |= 1 << (a - elem_a * 32)
                    states[index, elem_b] |= 1 << (b - elem_b * 32)

                    index += 1
