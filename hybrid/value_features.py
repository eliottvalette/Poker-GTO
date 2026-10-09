"""Lossless suit-aware range factors in a jointly canonical suit coordinate system."""

from itertools import combinations, permutations

from hybrid.learning import observation_features
from hybrid.ranges import JointRanges, Combo
from infoset import observe
from poker_game_expresso import HandState

COMBINATIONS = tuple(combinations(range(52), 2))
COMBO_INDEX = {h: i for i, h in enumerate(COMBINATIONS)}


def suit_aware_features(
    public: HandState, ranges: JointRanges, holding: Combo
) -> tuple[float, ...]:
    if (
        ranges.board != tuple(public.board)
        or ranges.ranges[public.current_player].probability(holding) <= 0
    ):
        raise ValueError(
            "Value features require feasible public ranges and query holding"
        )
    seats = [
        public.current_player,
        *[p for p in public.players if p != public.current_player],
    ]
    candidates = []
    for mapping in permutations(range(4)):

        def card(c):
            return c // 4 * 4 + mapping[c % 4]

        key = (*sorted(card(c) for c in holding), *(card(c) for c in public.board))
        candidates.append((key, mapping))
    best_key = min(key for key, _ in candidates)
    vectors = []
    for key, mapping in candidates:
        if key != best_key:
            continue
        vector = []
        for seat in seats:
            weights = [0.0] * 1326
            for hand, probability in ranges.ranges[seat].weights.items():
                mapped = tuple(sorted(c // 4 * 4 + mapping[c % 4] for c in hand))
                weights[COMBO_INDEX[mapped]] = probability
            vector.extend(weights)
        vectors.append(tuple(vector))
    view = public.clone()
    view.actor.cards = holding
    return (*observation_features(observe(view)), *min(vectors))
