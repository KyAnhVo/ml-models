import numpy as np
import numpy.typing as npt

def choose_J(P: npt.NDArray[np.float32], 
             r: npt.NDArray[np.float32],
             actions: npt.NDArray[np.int32],
             discount: np.float32):
    # Note: P is (from_state, to_state, action) denoting
    # P(from_state->to_state due to action)
    n = P.shape[0]
    action_count = P.shape[2]
    assert actions.shape == (n,)
    assert P.shape ==  (n, n, action_count)
    assert r.shape == (n,)
    assert 0.0 <= discount < 1

    P_2d = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        action = actions[i]
        for j in range(n):
            P_2d[i][j] = P[i][j][action]
    A = np.eye(n, dtype=np.float32) - discount * P_2d
    return np.linalg.solve(A, r)

def choose_pi(P: npt.NDArray[np.float32],
              J: npt.NDArray[np.float32]):
    action_count = P.shape[2]
    n = P.shape[0]
    assert P.shape == (n, n, action_count)
    assert J.shape == (n,)
    
    pi = np.zeros(n, dtype=np.int32)

    for i in range(n):
        possible_js = np.zeros(action_count, dtype=np.float32)
        for j in range(n):
            for action in range(action_count):
                possible_js[action] += P[i][j][action] * J[j]
        pi[i] = np.argmax(possible_js)

    return pi

def policy_iter(P: npt.NDArray[np.float32],
                r: npt.NDArray[np.float32],
                discount: np.float32,
                iter_count: int = 0):
    policy = np.zeros(P.shape[0], dtype=np.int32)
    if iter_count <= 0: # repeat until convergence
        while True:
            J = choose_J(P, r, policy, discount)
            next_policy = choose_pi(P, J)
            if np.array_equal(policy, next_policy):
                return policy
            policy = next_policy
    else: # repeat iter_count iterations
        for _ in range(iter_count):
            J = choose_J(P, r, policy, discount)
            policy = choose_pi(P, J)
        return policy

