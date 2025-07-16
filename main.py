from agent import QLearnAgent
from environment import env, ActionType
from pprint import pprint

if __name__ == '__main__':

    seed = 404
    initial_epsilon = 0.7
    min_epsilon = 0.01
    epsilon_decay = 0.99

    initial_learning_rate = 0.3
    min_learning_rate = 0.01
    learning_rate_decay = 0.99

    gamma = 0.99
    num_episodes = 5000
    checkpoints = [int(num_episodes * 0.25), int(num_episodes * 0.5), num_episodes - 1]

    q_agent = QLearnAgent(env, episode_length=num_episodes, power_consumption_weight=1, seed=seed)
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

    q_agent.draw(7)
    print(f"Seed: {seed}")
    route = q_agent.walk(7)
    print("Route:")
    for step in route:
        timestep, current_channel, action = step
        print(f"Timestep: {timestep}, Current Channel: {current_channel}")
        if action.type == ActionType.SWITCH:
            print(f"Chosen Action: Switching to channel {action.channel_index}")
        elif action.type == ActionType.STAY:
            print(f"Chosen Action: Staying on the same channel {current_channel}")
        else:
            print("Unknown action")

        print('-' * 50)

