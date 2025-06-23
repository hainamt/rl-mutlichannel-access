from itertools import product
import random
from environment import Channel, Action, ActionType

class QTable(dict):
    def __init__(self, num_timestep, num_channels):

        super().__init__()
        self.num_timestep = num_timestep
        self.num_channels = num_channels
        self.channel_objects = [Channel(channel_index=i) for i in range(num_channels)]
        self.channel_map = {ch.channel_index: ch for ch in self.channel_objects}

        self._initialize_q_table()

    def _initialize_q_table(self):
        timesteps = list(range(self.num_timestep))
        states = list(product(timesteps, self.channel_objects))

        for state in states:
            _, current_channel = state
            actions = [Action(ActionType.SWITCH, channel_index=i)
                       for i in range(self.num_channels)
                       if i != current_channel.channel_index] + \
                      [Action(ActionType.STAY, channel_index=current_channel.channel_index)]

            self[state] = {action: 0.0 for action in actions}

    def _convert_key(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            t, c = key
            if isinstance(c, int) and c in self.channel_map:
                return t, self.channel_map[c]
        return key

    def __getitem__(self, key):
        converted_key = self._convert_key(key)
        return super().__getitem__(converted_key)

    def __setitem__(self, key, value):
        converted_key = self._convert_key(key)
        return super().__setitem__(converted_key, value)

    def __contains__(self, key):
        converted_key = self._convert_key(key)
        return super().__contains__(converted_key)

    def get_all_actions_of_state(self, timestep, channel_index):
        return list(self[timestep, channel_index].keys())

    def set_q_value(self, timestep, channel_index, action, value):
        self[timestep, channel_index][action] = value

    def get_q_value(self, timestep, channel_index, action):
        return self[timestep, channel_index][action]

    def get_best_action(self, timestep, channel_index):
        state_actions = self[timestep, channel_index]
        max_value = max(state_actions.values())
        best_actions = [action for action, value in state_actions.items() if value == max_value]
        return random.choice(best_actions)

    def get_best_q_value(self, timestep, channel_index):
        return max(self[timestep, channel_index].values())

    def get_best_route(self, start_channel_index=0):
        route = []
        current_channel_index = start_channel_index

        for timestep in range(self.num_timestep):
            best_action = self.get_best_action(timestep, current_channel_index)
            route.append((timestep, current_channel_index, best_action))
            current_channel_index = best_action.channel_index
        return route