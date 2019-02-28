import numba


@numba.njit(cache=True)
def create_reference_state(n, l, states, ref_index=0):
    for i in range(n):
        elem = i // 32
        states[ref_index, elem] |= 1 << (i - elem * 32)

    for i in range(1, len(states)):
        states[i] += states[ref_index]


# @numba.njit(cache=True)
def create_doubles_states(n, l, states, index):
    for a in range(n, l):
        elem_a = a // 32
        for b in range(a + 1, l):
            elem_b = b // 32

            states[index, elem_a] |= 1 << (a - elem_a * 32)
            states[index, elem_b] |= 1 << (b - elem_b * 32)

            index += 1

    print(f"Index = {index}")
