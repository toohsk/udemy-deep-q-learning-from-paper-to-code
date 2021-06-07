import gym
import numpy as np
import matplotlib.pyplot as plt


env = gym.make('FrozenLake-v0')

# to render environment, run `env.render()` in the terminal.
# SFFF
# FHFH
# FFFH
# HFFG
# agent starts from `S`, and aims for the goal `G`.
# each actions that agent could take are represented as below.
# 0: LEFT, 1: DOWN, 2: RIGHT, 3: UP

# in policy_map, key represents the position in the environment,
# and the value represents the action.
policy_map = {
    0: 2,  # START
    1: 2,
    2: 1,
    3: 0,
    4: 1,
    5: 1,  # HOLE
    6: 1,
    7: 1,  # HOLE
    8: 2,
    9: 2,
    10: 1,
    11: 0,  # HOLE
    12: 1,  # HOLE
    13: 2,
    14: 2,
    15: 3,  # GOAL
}

n_games = 1000
game_scores = []
summary = []

for game_index in range(n_games):

    # initialize variables for each games.
    is_finish = False
    obs = env.reset()
    score = 0

    while not is_finish:
        action = policy_map[obs]
        obs, reward, is_finish, info = env.step(action)
        score += reward

    game_scores.append(score)

    if len(game_scores) != 0 and len(game_scores) % 10 == 0:
        summary.append(np.mean(game_scores))
        game_scores = []

plt.plot(summary)
plt.show()
