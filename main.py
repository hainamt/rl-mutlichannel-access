from agent import QLearnAgent
from environment import env

if __name__ == '__main__':
    # state (t, c)
    # action (a)
    initial_epsilon = 0.5
    min_epsilon = 0.01
    epsilon_decay = 0.99

    initial_learning_rate = 0.1
    min_learning_rate = 0.01
    learning_rate_decay = 0.99

    gamma = 0.99
    num_episodes = 1000
    checkpoints = [int(num_episodes * 0.25), int(num_episodes * 0.5), num_episodes - 1]

    # time_length = len(env)
    # num_channels = len(env[0])
    # q_table = QTable(time_length, num_channels)
    #
    # q_history, delta_q, episode_rewards = run_q_learning(num_episodes, q_table,
    #                                                      gamma,
    #                                                      initial_epsilon, min_epsilon, epsilon_decay,
    #                                                      initial_learning_rate, min_learning_rate, learning_rate_decay,
    #                                                      checkpoints)
    # best_combo = q_table.get_best_route(7)
    # env_quality = np.array([[item.channel_quality for item in timestep] for timestep in env]).T
    #
    # draw_best_route(best_combo, env_quality)

    q_agent = QLearnAgent(env, episode_length=num_episodes, power_consumption_weight=1)
    q_agent.train(
        gamma=gamma,
        initial_epsilon=initial_epsilon,
        min_epsilon=min_epsilon,
        epsilon_decay=epsilon_decay,
        initial_learning_rate=initial_learning_rate,
        min_learning_rate=min_learning_rate,
        learning_rate_decay=learning_rate_decay,
        checkpoints=checkpoints
    )

    q_agent.draw_all(7)

