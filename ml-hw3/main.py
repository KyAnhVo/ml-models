from policy_iter import policy_iter
from value_iter import value_iter
import numpy as np
import numpy.typing as npt

r = np.array([0, 1, 0, 10], dtype=np.float32)
transitions = np.array([
    (0, 0, 0, 1),
    (1, 1, 0, 1),
    (1, 1, 1, 0),
    (0, 0, 0, 1),
    (1, 1, 1, 0),
    (0, 0, 1, 2)
], dtype=np.int32)

def q_learning(transitions: npt.NDArray[np.int32], 
               r: npt.NDArray[np.float32], 
               action_count: int = 2,
               state_count: int = 4,
               alpha: np.float32 = np.float32(0.4),
               gamma: np.float32 = np.float32(0.5)
               ):

    n, k = state_count, action_count
    q: npt.NDArray[np.float32] = np.zeros((n, k), dtype=np.float32)
    for from_state, reward, action, to_state in transitions:
        q[from_state][action] = (1 - alpha) * q[from_state][action]
        q[from_state][action] += alpha * (reward + gamma * np.max(q[to_state]))
        print(q.flatten())
    return q

q_learning(transitions, r)

