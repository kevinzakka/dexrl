import enum
from dataclasses import dataclass

from dexterity import goal


class ReplayStrategy(enum.Enum):
    FUTURE = enum.auto()
    EPISODE = enum.auto()
    RANDOM = enum.auto()


@dataclass
class HindsightExperienceReplay:
    goal_generator: goal.GoalGenerator
    replay_strategy: ReplayStrategy = ReplayStrategy.FUTURE

    def sample_transition(self):
        ...
