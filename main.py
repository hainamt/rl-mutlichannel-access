from environment import env
from q_table import QTable
import itertools


# state (t, c)
# action (a)
time_length = len(env)
num_channels = len(env[0])
q_table = QTable(time_length, num_channels)
# Q_table[(0, Channel(0))] # (timestep: 0, channel: 0)
# Q_table[(0, 0)] # same as above
