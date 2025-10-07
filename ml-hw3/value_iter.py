import numpy as np
import numpy.typing as npt

def value_iter(P: npt.NDArray[np.float32],
               r: npt.NDArray[np.float32],
               convergence_criteria: np.float32,
               discount: np.float32):
    n = P.shape[0]
    iter_count = np.int32(0)
    action_count = P.shape[2]
    j_curr: npt.NDArray[np.float32] = np.copy(r)
    j_next: npt.NDArray[np.float32] = np.empty(n, dtype=np.float32)
    max_diff = np.float32(np.inf)

    # Calculate J* = j_curr
    while max_diff > convergence_criteria:
        curr_max_diff = np.float32(-1 * np.inf)
        for i in range(n):
            # choose action
            possible_action_values: npt.NDArray[np.float32] = np.zeros(action_count, dtype=np.float32)
            for action in range(action_count):
                for j in range(n):
                    possible_action_values[action] += P[i][j][action] * j_curr[j]
            j_next[i] = r[i] + discount * possible_action_values.max()
            curr_max_diff = np.maximum(curr_max_diff, np.abs(j_curr[i] - j_next[i]))
        max_diff = curr_max_diff
        j_curr, j_next = j_next, j_curr
        iter_count += 1
    
    # From J* choose best policy pi*
    policy: npt.NDArray[np.int32] = np.zeros(n, dtype=np.int32)
    for i in range(n):
        possible_action_values: npt.NDArray[np.float32] = np.zeros(action_count, dtype=np.float32)
        for j in range(n):
            for action in range(action_count):
                possible_action_values[action] += P[i][j][action] * j_curr[j]
        policy[i] = np.argmax(possible_action_values)

    return policy, j_curr, iter_count

