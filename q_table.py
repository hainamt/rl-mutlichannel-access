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

    def get_best_action(self, timestep, channel_index): # (t, c)
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
    
    
    def to_polars_dataframe(self):
        import polars as pl

        all_actions = set()
        for state_actions in self.values():
            all_actions.update(state_actions.keys())
        
        sorted_actions = sorted(all_actions, key=lambda a: (a.type.value, a.channel_index))
        action_columns = [f"{a.type.name}_{a.channel_index}" for a in sorted_actions]

        data_dict = {}
        
        for state, actions_dict in self.items():
            timestep, channel = state
            channel_index = channel.channel_index
            
            if timestep not in data_dict:
                data_dict[timestep] = {}
            
            if channel_index not in data_dict[timestep]:
                data_dict[timestep][channel_index] = {}
            
            for action, q_value in actions_dict.items():
                action_str = f"{action.type.name}_{action.channel_index}"
                data_dict[timestep][channel_index][action_str] = q_value
        
        records = []
        for timestep in sorted(data_dict.keys()):
            for channel_index in sorted(data_dict[timestep].keys()):
                row = {"timestep": timestep, "channel_index": channel_index}
                # Add each action's Q-value, using 0.0 as default if the action isn't available
                for action_col in action_columns:
                    row[action_col] = data_dict[timestep][channel_index].get(action_col, 0.0)
                records.append(row)
        
        if not records:
            schema = {"timestep": pl.Int32, "channel_index": pl.Int32}
            schema.update({col: pl.Float64 for col in action_columns})
            return pl.DataFrame(schema=schema)
        
        df = pl.DataFrame(records)

        return df