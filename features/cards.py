"""Card structure from known cards only; categories use 0=high card .. 8=straight flush."""
from __future__ import annotations
from collections import Counter
from functools import lru_cache

CARD_FEATURE_NAMES = ("made_category", "flush_draw", "straight_outs", "flush_outs", "overcards",
                      "board_pair_multiplicity", "board_max_suit", "board_rank_span", "ace_suit_blockers")


def canonical_suits(cards: tuple[int, ...]) -> tuple[tuple[int, ...], dict[int, int]]:
    mapping: dict[int, int] = {}
    result = []
    for card in cards:
        if card == 52:
            result.append(card)
            continue
        suit = card % 4
        if suit not in mapping:
            mapping[suit] = len(mapping)
        result.append(card // 4 * 4 + mapping[suit])
    return tuple(result), mapping


def has_straight(ranks: set[int]) -> bool:
    ranks = ranks | ({1} if 14 in ranks else set())
    return any(set(range(low, low + 5)) <= ranks for low in range(1, 11))


def made_category(cards: tuple[int, ...]) -> int:
    ranks = Counter(c // 4 + 2 for c in cards)
    suits = Counter(c % 4 for c in cards)
    flush_suits = [s for s, n in suits.items() if n >= 5]
    if any(has_straight({c // 4 + 2 for c in cards if c % 4 == s}) for s in flush_suits):
        return 8
    counts = sorted(ranks.values(), reverse=True)
    if counts and counts[0] == 4:
        return 7
    if counts and counts[0] >= 3 and len(counts) > 1 and counts[1] >= 2:
        return 6
    if flush_suits:
        return 5
    if has_straight(set(ranks)):
        return 4
    if counts and counts[0] >= 3:
        return 3
    if sum(n >= 2 for n in counts) >= 2:
        return 2
    return int(any(n >= 2 for n in counts))


@lru_cache(maxsize=8192)
def card_features(hero: tuple[int, int], board: tuple[int, ...]) -> tuple[float, ...]:
    """Bounded reuse of pure structural features, keyed only by observable cards."""
    known = (*hero, *board)
    if len(hero) != 2 or len(board) not in (0, 3, 4, 5) or len(set(known)) != len(known) or any(c not in range(52) for c in known):
        raise ValueError(f"Invalid observable cards: hero={hero}, board={board}")
    ranks = {c // 4 + 2 for c in known}
    suits = Counter(c % 4 for c in known)
    board_ranks = Counter(c // 4 + 2 for c in board)
    board_suits = Counter(c % 4 for c in board)
    drawing = len(board) in (3, 4)
    straight_outs = sum(has_straight(ranks | {c // 4 + 2}) for c in range(52) if c not in known) if drawing and not has_straight(ranks) else 0
    flush_outs = sum(suits[c % 4] == 4 for c in range(52) if c not in known) if drawing and max(suits.values()) < 5 else 0
    return (made_category(known) / 8, float(drawing and max(suits.values()) == 4), straight_outs / 52,
            flush_outs / 52, sum(c // 4 + 2 > max(board_ranks, default=14) for c in hero) / 2,
            max(board_ranks.values(), default=0) / 4, max(board_suits.values(), default=0) / 5,
            (max(board_ranks) - min(board_ranks)) / 12 if board else 0,
            sum(c // 4 == 12 and board_suits[c % 4] > 0 for c in hero) / 2)
