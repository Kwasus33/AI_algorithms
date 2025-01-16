import numpy as np


class Qlearning:
    def __init__(self, states_size, actions_size, discount_factor, learning_rate):
        self.states_size = states_size
        self.actions_size = actions_size
        self.discount_factor = discount_factor  # gamma
        self.lr = learning_rate
        self.reset_qtable()

    def update(self, state, action, reward, new_state):
        """
        Bellman’s Equation:
        Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        *** in simple cases like cliff walking mostly max Q(s',a') == Q(s,a) ***
        delta = [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        Q(s,a):= Q(s,a) + lr * delta
        """
        delta = (
            reward
            + self.discount_factor * np.max(self.qtable[new_state, :])
            - self.qtable[state, action]
        )
        self.qtable[state, action] += self.lr * delta

    def reset_qtable(self):
        self.qtable = np.zeros((self.states_size, self.actions_size))


class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        # epislorn is probability of exploration
        # typically low to exploit best part of env but allows to occasionally explore env (tests possible benefits of new actions)

    def choose_action(self, actions_space, state, qtable):
        explor_exploit_tradeoff = np.random.rand()

        # Exploration
        if explor_exploit_tradeoff < self.epsilon:
            action = actions_space.sample()

        # Exploitation (choosing best move - taking the biggest Q-value for this state)
        else:
            # action = np.argmax(qtable[state])

            max_ids = np.where(qtable[state, :] == max(qtable[state, :]))[0]
            action = np.random.choice(max_ids)

        return action
