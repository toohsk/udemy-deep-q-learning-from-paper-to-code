import gym
import numpy as np
import matplotlib.pyplot as plt

from q_learning_agent import Agent


if __name__ == '__main__':

    # to render environment, run `env.render()` in the terminal.
    # SFFF
    # FHFH
    # FFFH
    # HFFG
    # agent starts from `S`, and aims for the goal `G`.
    # each actions that agent could take are represented as below.
    # 0: LEFT, 1: DOWN, 2: RIGHT, 3: UP
    env = gym.make('FrozenLake-v0')

    agent = Agent(
        lr = 0.001,
        gamma = 0.9,
        n_actions = env.action_space.n,
        n_states = env.observation_space.n,
        epsilon_max = 1.0,
        epsilon_min = 0.01,
        epsilon_decrease = 0.9999995,
    )

    n_games = 500000
    game_scores = []
    summary = []
    plot_interval = 100
    logging_interval = 1000

    for game_index in range(n_games):

        # initialize variables for each games.
        is_finish = False
        obs = env.reset()
        score = 0

        while not is_finish:
            action = agent.choose_action(obs)
            # print(action)
            obs_prime, reward, is_finish, info = env.step(action)
            agent.learn(obs, action, reward, obs_prime)
            obs = obs_prime
            score += reward

        game_scores.append(score)

        if len(game_scores) != 0 and len(game_scores) % plot_interval == 0:
            ave_score = np.mean(game_scores[-plot_interval:])
            summary.append(ave_score)
            if len(game_scores) % logging_interval == 0:
                print(f"game: {game_index+1}", "win percentage: %.2f" % ave_score,
                      "epsilon %.2f" % agent.epsilon)

    plt.plot(summary)
    plt.show()
