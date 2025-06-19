from environment import *
from q_table import QTable
from tqdm import tqdm
import random
import copy


# policy
def select_action(timestep:int, current_channel_index:int):
    random_float = random.random()
    if random_float <= epsilon:
        return random.choice(q_table.get_all_actions_of_state(timestep, current_channel_index))
    return q_table.get_best_action(timestep, current_channel_index)

def calculate_reward(action: Action, next_channel: Channel):
    return next_channel.channel_quality.value - energy_consumption_weight * action.type.value

def step(timestep: int, current_channel: Channel, action: Action):
    next_channel = current_channel
    if action.type != ActionType.STAY:
        next_channel_index = action.channel_index
        for channel in env[timestep]:
            if channel.channel_index == next_channel_index:
                next_channel = channel
    return next_channel, calculate_reward(action, next_channel)


def run_q_learning(num_episodes:int,
                   q_table:QTable,
                   gamma:float,
                   learning_rate:float,
                   checkpoints:list[int]):
    q_history = []
    delta_q = dict()

    for i in tqdm(range(num_episodes)):
        current_channel = Channel(7)
        for timestep in tqdm(env):
            chosen_action = select_action(timestep, current_channel.channel_index)
            next_channel, reward = step(timestep, current_channel, chosen_action)

            old_q = q_table.get_q_value(timestep=timestep,
                                       channel_index=current_channel.channel_index,
                                       action=chosen_action)
            new_q = old_q + learning_rate * (reward + gamma * q_table.get_best_q_value(timestep, current_channel.channel_index) - old_q)
            q_table.set_q_value(timestep, current_channel.channel_index, chosen_action, new_q)

            if ((timestep, current_channel), chosen_action) not in delta_q:
                delta_q[((timestep, current_channel), chosen_action)] = []
            delta_q[((timestep, current_channel), chosen_action)].append(abs(new_q - old_q))

            current_channel = next_channel

        if i in checkpoints:
            q_history.append(copy.deepcopy(q_table))

    return q_history, delta_q


if __name__ == '__main__':
    # state (t, c)
    # action (a)
    epsilon = 0.003
    learning_rate = 0.001
    gamma = 0.99
    time_length = len(env)
    num_channels = len(env[0])
    q_table = QTable(time_length, num_channels)
    # q_table[(0, Channel(0))] # (timestep: 0, channel: 0)
    # q_table[(0, 0)] # same as above

    num_episodes = 1000
    checkpoints = [int(num_episodes * 0.25), int(num_episodes * 0.5), num_episodes - 1]

    q_history, delta_q = run_q_learning(num_episodes, q_table, gamma, learning_rate, checkpoints)


