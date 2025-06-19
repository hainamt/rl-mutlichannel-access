from environment import env
from q_table import QTable
import random


# state (t, c)
# action (a)
epsilon = 0.003
time_length = len(env)
num_channels = len(env[0])
q_table = QTable(time_length, num_channels)

# policy
def select_action(timestep:int, current_channel_index:int):
    random_float = random.random()
    if random_float <= epsilon:
        return random.choice(q_table.get_all_actions_of_state(timestep, current_channel_index))
    return max(q_table.get_best_action(timestep, current_channel_index))
# Q_table[(0, Channel(0))] # (timestep: 0, channel: 0)
# Q_table[(0, 0)] # same as above
