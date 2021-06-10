import numpy as np


class Agent:

    def __init__(self, lr, gamma, n_actions, n_states, epsilon_max, epsilon_min, epsilon_decrease):
        self.q_table = {}
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decrease = epsilon_decrease

        self.initialize_q_table()

    def initialize_q_table(self):
        for s in range(self.n_states):
            for a in range(self.n_actions):
                self.q_table[(s, a)] = 0.0

    def choose_action(self, state):
        # choose action based on epsilon-greedy
        if np.random.random() < self.epsilon:
            # print("exploratory action!")
            action = np.random.choice([a for a in range(self.n_actions)])
        else:
            # print(f"greedy action! epsilon:{self.epsilon}")
            action = np.argmax(
                np.array([self.__q_val(state, a) for a in range(self.n_actions)])
            )

        return action

    def decrease_epsilon(self):
        self.epsilon = max(
            self.epsilon * self.epsilon_decrease,
            self.epsilon_min
        )

    def learn(self, state, action, reward, state_prime):
        a_max = np.argmax(
            np.array([self.__q_val(state_prime, a) for a in range(self.n_actions)])
        )
        self.q_table[(state, action)] += self.lr * \
            (reward + self.gamma * self.__q_val(state_prime, a_max) - self.__q_val(state, action))

        self.decrease_epsilon()

    def __q_val(self, state, action):
        return self.q_table[(state, action)]
