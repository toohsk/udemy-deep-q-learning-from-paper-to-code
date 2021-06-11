import gym
import matplotlib.pyplot as plt

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules import MSELoss

from utils import plot_learning_curve


class LinearDeepQNet(nn.Module):

    def __init__(self, lr, input_dims, output_dims):

        super(LinearDeepQNet, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, output_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        # in my environment, I havn't set up cuda.
        # so just use cpu, instead cuda stuffs.
        #self.device = T.device('cuda:0' if T.cuda.is_available else 'cpu')
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        x_ = F.relu(self.fc1(state))
        outputs = self.fc2(x_)
        return outputs


class Agent():

    def __init__(self, lr, input_dims, n_actions, gamma, epsilon_max, epsilon_min, epsilon_decrease):

        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions

        self.gamma = gamma
        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decrease = epsilon_decrease

        self.Q = LinearDeepQNet(
            lr=self.lr,
            input_dims=self.input_dims,
            output_dims=self.n_actions
        )

    def choose_action(self, state):
        # choose action based on epsilon-greedy
        if np.random.random() < self.epsilon:
            # print("exploratory action!")
            action = np.random.choice([a for a in range(self.n_actions)])
        else:
            # print(f"greedy action! epsilon:{self.epsilon}")
            state_ = T.tensor(state, dtype=T.float).to(self.Q.device)
            actions = self.Q.forward(state_)
            action = T.argmax(actions).item()

        return action

    def decrease_epsilon(self):
        self.epsilon = max(
            self.epsilon - self.epsilon_decrease,
            self.epsilon_min
        )

    def learn(self, state, action, reward, state_prime):
        self.Q.optimizer.zero_grad()

        # convert to cuda tensor
        states = T.tensor(state, dtype=T.float).to(self.Q.device)
        actions = T.tensor(action).to(self.Q.device)
        rewards = T.tensor(reward).to(self.Q.device)
        states_prime = T.tensor(state_prime, dtype=T.float).to(self.Q.device)

        # use action as a index and take the action's reward.
        q_prediction = self.Q.forward(states)[actions]
        # take max state_prime reward
        q_next = self.Q.forward(states_prime).max()
        q_target = rewards + self.gamma * q_next

        loss = self.Q.loss(q_target, q_prediction).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()

        self.decrease_epsilon()


if __name__ == "__main__":

    env = gym.make('CartPole-v1')

    n_games = 10000
    scores = []
    epsilon_history = []

    agent = Agent(
        lr=0.0001,
        input_dims=env.observation_space.shape,
        n_actions=env.action_space.n,
        gamma=0.99,
        epsilon_max=1.0,
        epsilon_min=0.1,
        epsilon_decrease=1e-5,
    )

    for i in range(n_games):

        score = 0
        is_finish = False
        obs = env.reset()

        while not is_finish:

            action = agent.choose_action(obs)
            obs_prime, reward, is_finish, info = env.step(action)
            score += reward
            agent.learn(
                state=obs,
                action=action,
                reward=reward,
                state_prime=obs_prime,
            )
            obs = obs_prime

        scores.append(score)
        epsilon_history.append(agent.epsilon)

        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print('episode ', i, 'score %.1f avg score %.1f epsilon %.2f' %
                  (score, avg_score, agent.epsilon))

    filename = 'cartpole_naive_dqn.png'
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(
        x=x,
        scores=scores,
        epsilons=epsilon_history,
        filename=filename,
    )