import gym
import numpy as np
import matplotlib.pyplot as plt


env = gym.make('FrozenLake-v0')

n_games = 1000
game_scores = []
summary = []

for game_index in range(n_games):

    # initialize variables for each games.
    is_finish = False
    obs = env.reset()
    score = 0

    while not is_finish:
        action = env.action_space.sample()
        obs, reward, is_finish, info = env.step(action)
        score += reward

    game_scores.append(score)

    if len(game_scores) != 0 and len(game_scores) % 10 == 0:
        summary.append(np.mean(game_scores))
        game_scores = []

plt.plot(summary)
plt.show()
