"""Card primitives for the isolated push/fold research tools.

The authoritative hand/player contracts live in poker_game_expresso.py.
"""
from dataclasses import dataclass
from itertools import combinations

RANKS = "23456789TJQKA"
SUITS = "♠♥♦♣"


@dataclass(frozen=True)
class Card:
    rank: int
    suit: int

    def __post_init__(self) -> None:
        if self.rank not in range(2, 15) or self.suit not in range(4):
            raise ValueError(f"Invalid card rank={self.rank}, suit={self.suit}")

    @property
    def id(self) -> int:
        return (self.rank - 2) * 4 + self.suit

    def __int__(self) -> int:
        return self.id

    def __index__(self) -> int:
        return self.id

    def __str__(self) -> str:
        return RANKS[self.rank - 2] + SUITS[self.suit]


class Deck:
    def __init__(self):
        self.cards = [Card(r, s) for r in range(2, 15) for s in range(4)]

    def get_card(self, rank: int, suit: int) -> Card:
        return Card(rank, suit)

    def all_starting_combos(self) -> list[tuple[Card, Card]]:
        return list(combinations(self.cards, 2))


def card_id_to_rank_suit(card_id: int) -> tuple[int, int]:
    if card_id not in range(52):
        raise ValueError(f"Invalid card ID {card_id}; expected 0..51")
    return card_id // 4 + 2, card_id % 4


def combo_to_169(card_1_id: int, card_2_id: int) -> str:
    if card_1_id == card_2_id:
        raise ValueError(f"Duplicate hole card {card_1_id}")
    r1, s1 = card_id_to_rank_suit(card_1_id)
    r2, s2 = card_id_to_rank_suit(card_2_id)
    high, low = sorted((r1, r2), reverse=True)
    label = RANKS[high - 2] + RANKS[low - 2]
    return label if r1 == r2 else label + ("s" if s1 == s2 else "o")


DECK = tuple(range(52))
ALL_COMBOS = tuple(combinations(DECK, 2))
