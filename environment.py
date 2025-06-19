from enum import Enum
from typing import NamedTuple


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
