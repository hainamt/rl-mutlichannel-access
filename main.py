from environment import *
import matplotlib.pyplot as plt
from q_table import QTable
from tqdm import tqdm
import random
import copy


# policy
def select_action(timestep: int, current_channel_index: int, epsilon: float = 0.1):
    random_float = random.random()
    if random_float <= epsilon:
        is_random = True
        action = random.choice(q_table.get_all_actions_of_state(timestep, current_channel_index))
    else:
        action = q_table.get_best_action(timestep, current_channel_index)
        is_random = False
    # print(f"action: {action}")
    return action, is_random


def calculate_reward(action: Action, next_channel: Channel):
    return next_channel.channel_quality.value - energy_consumption_weight * action.type.value


def step(timestep: int, current_channel: Channel, action: Action):
    next_channel = current_channel
    next_timestep = timestep + 1
    if next_timestep < len(env):  # Check if not at the end
        for channel in env[next_timestep]:
            if channel.channel_index == action.channel_index:
                next_channel = channel
    return next_channel, calculate_reward(action, next_channel)


def run_q_learning(num_episodes: int,
                   q_table: QTable,
                   gamma: float,
                   initial_epsilon: float,
                   min_epsilon: float,
                   epsilon_decay: float,
                   initial_learning_rate: float,
                   min_learning_rate: float,
                   learning_rate_decay: float,
                   checkpoints: list[int]):
    q_history = []
    delta_q = dict()
    max_timestep = len(env) - 1
    episode_rewards = []

    for i in tqdm(range(num_episodes)):
        rewards = 0
        current_channel = Channel(7)
        epsilon = max(min_epsilon, initial_epsilon * (epsilon_decay ** i))
        learning_rate = max(min_learning_rate, initial_learning_rate * (learning_rate_decay ** i))

        for timestep in range(len(env)):
            chosen_action, is_random = select_action(timestep, current_channel.channel_index, epsilon=epsilon)
            if is_random:
                print(f"Random action selected at timestep {timestep}, episode {i}")

            next_channel, reward = step(timestep, current_channel, chosen_action)
            rewards += reward

            old_q = q_table.get_q_value(timestep=timestep,
                                        channel_index=current_channel.channel_index,
                                        action=chosen_action)

            if timestep == max_timestep:
                next_state_best_q = 0
            else:
                next_timestep = timestep + 1
                next_state_best_q = q_table.get_best_q_value(next_timestep, next_channel.channel_index)

            new_q = old_q + learning_rate * (reward + gamma * next_state_best_q - old_q)
            q_table.set_q_value(timestep, current_channel.channel_index, chosen_action, new_q)

            if (timestep, current_channel, chosen_action) not in delta_q:
                delta_q[(timestep, current_channel, chosen_action)] = []
            delta_q[(timestep, current_channel, chosen_action)].append(abs(new_q - old_q))
            current_channel = next_channel

        episode_rewards.append(rewards)

        if i in checkpoints:
            q_history.append(copy.deepcopy(q_table))

    print(f"Latest epsilon: {epsilon}")
    print(f"Latest learning rate: {learning_rate}")

    return q_history, delta_q, episode_rewards


if __name__ == '__main__':
    # state (t, c)
    # action (a)
    initial_epsilon = 0.5
    min_epsilon = 0.01
    epsilon_decay = 0.997

    initial_learning_rate = 0.1
    min_learning_rate = 0.01
    learning_rate_decay = 0.99

    gamma = 0.99
    num_episodes = 10000
    checkpoints = [int(num_episodes * 0.25), int(num_episodes * 0.5), num_episodes - 1]

    time_length = len(env)
    num_channels = len(env[0])
    q_table = QTable(time_length, num_channels)

    q_history, delta_q, episode_rewards = run_q_learning(num_episodes, q_table,
                                                         gamma,
                                                         initial_epsilon, min_epsilon, epsilon_decay,
                                                         initial_learning_rate, min_learning_rate, learning_rate_decay,
                                                         checkpoints)
    best_combo = q_table.get_best_route(7)
    for timestep, current_channel, best_action in best_combo:
        print(f"Timestep: {timestep}, Current channel: {current_channel}")
        if best_action.type == ActionType.SWITCH:
            print(f"Switch to channel {best_action.channel_index}")
        else:
            print(f"Stay in channel {current_channel}")
    # print(f"Best route: {q_table.get_best_route(7)}")

    # average_delta_per_episode = []
    # for i in range(num_episodes):
    #     episode_deltas = [delta_q[(t, c, a)][i] for (t, c, a) in delta_q.keys() if i < len(delta_q[(t, c, a)])]
    #     if episode_deltas:
    #         average_delta_per_episode.append(sum(episode_deltas) / len(episode_deltas))
    #     else:
    #         average_delta_per_episode.append(0)
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(average_delta_per_episode)
    # plt.title('Average Change in Q-Values per Episode')
    # plt.xlabel('Episode')
    # plt.ylabel('Average ΔQ')
    # plt.grid(True)
    #
    # # Plot the rewards
    # plt.figure(figsize=(10, 6))
    # plt.plot(episode_rewards)
    # plt.title('Total Reward per Episode')
    # plt.xlabel('Episode')
    # plt.ylabel('Total Reward')
    # plt.grid(True)
    # plt.show()