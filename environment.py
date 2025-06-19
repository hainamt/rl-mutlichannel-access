from enum import Enum
from typing import NamedTuple
import random


class ChannelQuality(Enum):
    BAD = 0
    MEDIUM = 0.5
    HIGH = 1


class Channel(NamedTuple):
    channel_index: int
    channel_quality: ChannelQuality = None


class ActionType(Enum):
    STAY = 0
    SWITCH = 0.5


class Action(NamedTuple):
    type: ActionType
    channel_index: int


energy_consumption_weight = 0.5
env = [
    [
        Channel(0, ChannelQuality.MEDIUM),
        Channel(1, ChannelQuality.MEDIUM),
        Channel(2, ChannelQuality.MEDIUM),
        Channel(3, ChannelQuality.MEDIUM),
        Channel(4, ChannelQuality.MEDIUM),
        Channel(5, ChannelQuality.MEDIUM),
        Channel(6, ChannelQuality.MEDIUM),
        Channel(7, ChannelQuality.MEDIUM),
    ],
    [
        Channel(0, ChannelQuality.MEDIUM),
        Channel(1, ChannelQuality.HIGH),
        Channel(2, ChannelQuality.BAD),
        Channel(3, ChannelQuality.MEDIUM),
        Channel(4, ChannelQuality.HIGH),
        Channel(5, ChannelQuality.MEDIUM),
        Channel(6, ChannelQuality.MEDIUM),
        Channel(7, ChannelQuality.BAD),
    ],
    [
        Channel(0, ChannelQuality.BAD),
        Channel(1, ChannelQuality.MEDIUM),
        Channel(2, ChannelQuality.MEDIUM),
        Channel(3, ChannelQuality.HIGH),
        Channel(4, ChannelQuality.MEDIUM),
        Channel(5, ChannelQuality.HIGH),
        Channel(6, ChannelQuality.MEDIUM),
        Channel(7, ChannelQuality.MEDIUM),
    ],
    [
        Channel(0, ChannelQuality.HIGH),
        Channel(1, ChannelQuality.BAD),
        Channel(2, ChannelQuality.HIGH),
        Channel(3, ChannelQuality.BAD),
        Channel(4, ChannelQuality.HIGH),
        Channel(5, ChannelQuality.MEDIUM),
        Channel(6, ChannelQuality.BAD),
        Channel(7, ChannelQuality.MEDIUM),
    ],
    [
        Channel(0, ChannelQuality.MEDIUM),
        Channel(1, ChannelQuality.MEDIUM),
        Channel(2, ChannelQuality.BAD),
        Channel(3, ChannelQuality.MEDIUM),
        Channel(4, ChannelQuality.MEDIUM),
        Channel(5, ChannelQuality.MEDIUM),
        Channel(6, ChannelQuality.MEDIUM),
        Channel(7, ChannelQuality.BAD),
    ],
    [
        Channel(0, ChannelQuality.MEDIUM),
        Channel(1, ChannelQuality.HIGH),
        Channel(2, ChannelQuality.MEDIUM),
        Channel(3, ChannelQuality.MEDIUM),
        Channel(4, ChannelQuality.HIGH),
        Channel(5, ChannelQuality.BAD),
        Channel(6, ChannelQuality.MEDIUM),
        Channel(7, ChannelQuality.HIGH),
    ],
    [
        Channel(0, ChannelQuality.HIGH),
        Channel(1, ChannelQuality.MEDIUM),
        Channel(2, ChannelQuality.BAD),
        Channel(3, ChannelQuality.MEDIUM),
        Channel(4, ChannelQuality.MEDIUM),
        Channel(5, ChannelQuality.MEDIUM),
        Channel(6, ChannelQuality.HIGH),
        Channel(7, ChannelQuality.MEDIUM),
    ],
    [
        Channel(0, ChannelQuality.BAD),
        Channel(1, ChannelQuality.HIGH),
        Channel(2, ChannelQuality.MEDIUM),
        Channel(3, ChannelQuality.MEDIUM),
        Channel(4, ChannelQuality.BAD),
        Channel(5, ChannelQuality.HIGH),
        Channel(6, ChannelQuality.BAD),
        Channel(7, ChannelQuality.BAD),
    ],
    [
        Channel(0, ChannelQuality.MEDIUM),
        Channel(1, ChannelQuality.BAD),
        Channel(2, ChannelQuality.MEDIUM),
        Channel(3, ChannelQuality.MEDIUM),
        Channel(4, ChannelQuality.HIGH),
        Channel(5, ChannelQuality.BAD),
        Channel(6, ChannelQuality.MEDIUM),
        Channel(7, ChannelQuality.HIGH),
    ],
]


def get_max_Qs(Q, state):
    return max(list(Q[state].values()))


def policy(Q, s, actions, epsilon):
    random_float = random.random()
    if random_float <= epsilon:
        return random.choice(actions)
    else:
        return max(actions, key=lambda a: Q[s][a])


def calculate_reward(action: Action, next_channel: Channel):
    return next_channel.channel_quality.value - energy_consumption_weight * action.type.value


def step(timestep: int, action: Action, current_channel: Channel):
    next_channel = current_channel
    if action.type != ActionType.STAY:
        next_channel_index = action.channel_index
        for channel in env[timestep]:
            if channel.channel_index == next_channel_index:
                next_channel = channel
    reward = calculate_reward(action, next_channel)
    return next_channel, reward