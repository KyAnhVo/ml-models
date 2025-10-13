import math
import random

import numpy as np

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

# Intervals and velocity bounds

# Intervals
X_INTERVAL: float = 0.1
X_DOT_INTERVAL: float = 0.1
THETA_INTERVAL: float = 0.5
THETA_DOT_INTERVAL: float = 1

# velocity bounds
X_DOT_MAX = 30
THETA_DOT_MAX = 12
# Just negate the 2 max bounds for min bounds
X_DOT_MIN = -X_DOT_MAX
THETA_DOT_MIN = -THETA_DOT_MAX


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
        theta_size      = int(24 / THETA_INTERVAL)
        theta_dot_size  = int((THETA_DOT_MAX - THETA_DOT_MIN) / THETA_INTERVAL)
        x_size          = int(track_length * 2 / X_INTERVAL)
        x_dot_size      = int((X_DOT_MAX - X_DOT_MIN) / X_DOT_INTERVAL)
        action_size     = 2
        self.qtable     = np.zeros((x_size, x_dot_size, theta_size, theta_dot_size, action_size), 
                                   dtype=np.float32)

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
            action = None
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
        raise NotImplementedError

    # you may add more methods here for your needs. E.g., methods for discretizing the variables.


if __name__ == '__main__':
    pass
