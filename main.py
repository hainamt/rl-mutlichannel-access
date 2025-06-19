from environment import Channel, Action, ActionType, env
from q_table import QTable
import itertools


# state (t, c)
# action (a)
Q_dict = {}
timesteps = [_ for _ in range(len(env))]
channel_indexes= [Channel(channel_index=i) for i in range(len(env[0]))]

states = list(itertools.product(timesteps, channel_indexes))
for state in states:
    current_channel = state[1]
    actions = [Action(ActionType.SWITCH, channel_index=i.channel_index)
               for i in channel_indexes if i != current_channel.channel_index] \
            + [Action(ActionType.STAY, channel_index=current_channel.channel_index)]
    Q_dict[state] = {action: 0.0 for action in actions}

Q_table = QTable(channel_indexes)
Q_table.update(Q_dict)
# Q_table[(0, Channel(0))] # (timestep: 0, channel: 0)
# Q_table[(0, 0)] # same as above
