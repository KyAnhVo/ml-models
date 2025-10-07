import numpy as np
import numpy.typing as npt

def value_iter(P: npt.NDArray[np.float32],
               r: npt.NDArray[np.float32],
               convergence_criteria: np.float32):
    n = P.shape[0]
    action_count = P.shape[2]
    j_curr: npt.NDArray[np.float32] = np.copy(r)
    j_next: npt.NDArray[np.float32] = np.zeros(n, dtype=np.float32)
    max_diff = np.float32(np.inf)
    while 

