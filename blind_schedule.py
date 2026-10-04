"""Explicit hand-count blind schedules; chip amounts remain in initial-BB units."""
from __future__ import annotations

from dataclasses import dataclass

from poker_game_expresso import BlindLevel


@dataclass(frozen=True)
class BlindStage:
    first_hand: int
    blinds: BlindLevel

    def __post_init__(self) -> None:
        if type(self.first_hand) is not int or self.first_hand < 1:
            raise ValueError(f"Blind stage first_hand must be a positive integer: {self.first_hand!r}")
        if not isinstance(self.blinds, BlindLevel):
            raise ValueError(f"Blind stage requires BlindLevel, got {self.blinds!r}")


@dataclass(frozen=True)
class BlindSchedule:
    stages: tuple[BlindStage, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.stages, tuple) or not self.stages or any(not isinstance(stage, BlindStage) for stage in self.stages):
            raise ValueError(f"Blind schedule requires a nonempty tuple of BlindStage: {self.stages!r}")
        if self.stages[0].first_hand != 1:
            raise ValueError(f"Blind schedule must begin at hand 1, got {self.stages[0].first_hand}")
        for previous, current in zip(self.stages, self.stages[1:]):
            if current.first_hand <= previous.first_hand:
                raise ValueError(f"Blind stage hand numbers must strictly increase: {previous.first_hand}, {current.first_hand}")
            before, after = previous.blinds, current.blinds
            if after.small < before.small or after.big < before.big or after == before:
                raise ValueError(f"Blind levels must increase without reducing SB/BB: {before!r} -> {after!r}")

    @classmethod
    def fixed(cls, blinds: BlindLevel = BlindLevel()) -> BlindSchedule:
        """An explicit single fixed level, used by finite engine fixtures."""
        return cls((BlindStage(1, blinds),))

    def for_hand(self, hand_number: int) -> tuple[int, BlindLevel]:
        if type(hand_number) is not int or hand_number < 1:
            raise ValueError(f"Blind lookup requires a positive integer hand number: {hand_number!r}")
        index = 0
        for candidate, stage in enumerate(self.stages):
            if stage.first_hand > hand_number:
                break
            index = candidate
        return index, self.stages[index].blinds


# A configurable simulation preset, not an official real-money Expresso structure.
# The final listed level is held; physical chip amounts are never rescaled.
DEFAULT_SIMULATION_SCHEDULE = BlindSchedule(tuple(
    BlindStage(1 + 10 * index, BlindLevel(0.5 * 2 ** index, 1.0 * 2 ** index))
    for index in range(6)
))
