import math
import random

import numpy as np

# greedy initial value
GREEDY_INITIAL = 0

FORWARD_ACCEL = 1
BACKWARD_ACCEL = 0

"""
Notes on boundaries for velocity

We know that if thing falls fast enough, there is no recovery.
Similarly, if cart goes fast enough, there is no recovery before it wee-woo out of track.
Hence, just set an upper bound for both.

Also 1: these values can be tested out, so feel free to test things out.

Also 2: don't cook too hard or we get 100GB of Q-table and I don't have
the RAM for that.

Also 3: PLEASE make sure range(value_bound) is divisible by its intervals.
""" 

# Intervals and velocity bounds (cook but don't overcook)

# Intervals
X_INTERVAL: float = 1.2
X_DOT_INTERVAL: float = 10
THETA_INTERVAL: float = 2
THETA_DOT_INTERVAL: float = 10

# velocity bounds
X_DOT_MAX = 50
THETA_DOT_MAX = 50



class QLearningAgent:
    def __init__(self, lr, gamma, track_length, epsilon, policy='greedy'):
        """
        A function for initializing your agent
        :param lr: learning rate
        :param gamma: discount factor
        :param track_length: how far the ends of the track are from the origin.
            e.g., while track_length is 2.4,
            the x-coordinate of the left end of the track is -2.4,
            the x-coordinate of the right end of the track is 2.4,
            and x-coordinate of the the cart is 0 initially.
        :param epsilon: epsilon for the mixed policy
        :param policy: can be 'greedy' or 'mixed'
        """
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.track_length = track_length
        self.policy = policy
        random.seed(11)
        np.random.seed(11)

        # q-table size
        theta_size      = int(24 / THETA_INTERVAL) + 1
        theta_dot_size  = int((THETA_DOT_MAX * 2) / THETA_INTERVAL) + 1
        x_size          = int(track_length * 2 / X_INTERVAL) + 1
        x_dot_size      = int((X_DOT_MAX * 2) / X_DOT_INTERVAL) + 1
        action_size     = 2
        self.qtable = np.zeros((x_size, x_dot_size, theta_size, theta_dot_size, action_size), 
                               dtype=np.float32)
        self.qtable.fill(GREEDY_INITIAL)

        # intervals

        self.x_interval         = X_INTERVAL
        self.x_dot_interval     = X_DOT_INTERVAL
        self.theta_interval     = THETA_INTERVAL
        self.theta_dot_interval = THETA_DOT_INTERVAL

        # ranges and intervals
        # all maxes decreased by a very small amount to index with floor accordingly

        very_small_num = 1e-3
        self.x_min          = -track_length
        self.x_max          = track_length - very_small_num * self.x_interval
        self.theta_min      = -12.0
        self.theta_max      = 12.0 - very_small_num * self.theta_interval
        self.x_dot_min      = -X_DOT_MAX
        self.x_dot_max      = X_DOT_MAX - very_small_num * self.x_dot_interval
        self.theta_dot_min  = -THETA_DOT_MAX
        self.theta_dot_max  = THETA_DOT_MAX - very_small_num * self.theta_dot_interval



    def reset(self):
        """
        you may add code here to re-initialize your agent before each trial
        :return:
        """
        pass

    def get_action(self, x, x_dot, theta, theta_dot):
        """
        main.py calls this method to get an action from your agent
        :param x: the position of the cart
        :param x_dot: the velocity of the cart
        :param theta: the angle between the cart and the pole
        :param theta_dot: the angular velocity of the pole
        :return:
        """

        # NOTE: Reminder to switch rad to degree first

        if self.policy == 'mixed' and random.random() < self.epsilon:
            action = random.sample([FORWARD_ACCEL, BACKWARD_ACCEL], 1)[0]
        else:
            # fill your code here to get an action from your agent
            theta = theta * 180 / math.pi
            theta_dot = theta_dot * 180 / math.pi
            x, x_dot, theta, theta_dot = self.to_index((x, x_dot, theta, theta_dot))
            q_vals = self.qtable[x, x_dot, theta, theta_dot]
            # alright, since greedy is failing, let's cook a little bit with random when q's are equal
            if q_vals[0] == q_vals[1]:
                # Prefer actions that keep cart centered or pole upright
                if abs(x) > 1.0:
                    return FORWARD_ACCEL if x < 0 else BACKWARD_ACCEL
                if abs(theta) > 0.05:
                    return FORWARD_ACCEL if theta < 0 else BACKWARD_ACCEL
                return FORWARD_ACCEL  # default
            action = int(np.argmax(q_vals))
        return action

    def update_Q(self, prev_state, prev_action, cur_state, reward):
        """
        main.py calls this method so that you can update your Q-table
        :param prev_state: previous state, a tuple of (x, x_dot, theta, theta_dot)
        :param prev_action: previous action, FORWARD_ACCEL or BACKWARD_ACCEL
        :param cur_state: current state, a tuple of (x, x_dot, theta, theta_dot)
        :param reward: reward, 0.0 or -1.0
        e.g., if we have S_i ---(action a, reward)---> S_j, then
            prev_state is S_i,
            prev_action is a,
            cur_state is S_j,
            rewards is reward.
        :return:
        """
        # NOTE: Reminder to switch rad to degree first
        x_prev, x_dot_prev, theta_prev, theta_dot_prev = prev_state
        x_curr, x_dot_curr, theta_curr, theta_dot_curr = cur_state
        
        # swap rad to degree
        theta_prev = theta_prev * 180 / math.pi
        theta_dot_prev = theta_dot_prev * 180 / math.pi
        theta_curr = theta_curr * 180 / math.pi
        theta_dot_curr = theta_dot_curr * 180 / math.pi

        x_prev, x_dot_prev, theta_prev, theta_dot_prev = self.to_index(
                (x_prev, x_dot_prev, theta_prev, theta_dot_prev)
                )
        x_curr, x_dot_curr, theta_curr, theta_dot_curr = self.to_index(
                (x_curr, x_dot_curr, theta_curr, theta_dot_curr)
                )

        j_val = self.qtable[x_curr, x_dot_curr, theta_curr, theta_dot_curr].max(axis=-1)
        self.qtable[x_prev, x_dot_prev, theta_prev, theta_dot_prev, prev_action] = (
            self.lr * (reward + self.gamma * j_val) + 
            (1 - self.lr) * self.qtable[x_prev, x_dot_prev, theta_prev, theta_dot_prev, prev_action]
        )


    # you may add more methods here for your needs. E.g., methods for discretizing the variables.

    def to_index(self, state):
        """
        Convert state to to index
        :param state: 4-tuple (x, x_dot, theta, theta_dot)
        :return: 4-tuple of indices representing the state 4-tuple
        """
        x, x_dot, theta, theta_dot = state

        def indexize(var, var_min, var_max, var_interval):
            var = max(var, var_min)
            var = min(var, var_max)
            return math.floor((var - var_min) / var_interval) # move head to 0, choose interval

        x = indexize(x, self.x_min, self.x_max, self.x_interval)
        x_dot = indexize(x_dot, self.x_dot_min, self.x_dot_max, self.x_dot_interval)
        theta = indexize(theta, self.theta_min, self.theta_max, self.theta_interval)
        theta_dot = indexize(theta_dot, self.theta_dot_min, self.theta_dot_max, self.theta_dot_interval)

        return x, x_dot, theta, theta_dot




if __name__ == '__main__':
    pass
