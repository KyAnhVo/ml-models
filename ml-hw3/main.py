from policy_iter import policy_iter
import numpy as np
import numpy.typing as npt

P = np.array([
    [[1, 0], [0, 1], [0, 0], [0, 0]],
    [[0, 0], [1, 0], [0, 1], [0, 0]],
    [[0.5, 0], [0, 0], [0.5, 0], [0, 1]],
    [[1, 0], [0, 0], [0, 0], [0, 1]]
], dtype=np.float32)
r = np.array([10, 0, 0, -5], dtype=np.float32)

print(policy_iter(P, r, np.float32(0.5)))
