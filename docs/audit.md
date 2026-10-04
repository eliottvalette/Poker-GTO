# Migration audit

Audited the working tree before replacement. Local HEAD and Poker-GTO remote HEAD
are both 1e675af5871b566d2ba9bccc5f9f1a5143785a1c. The configured origin still uses
the repository's former GTO_Bot name. No remote mutation is needed.

## Former contracts

- classes.Player uses integer stack=100 and fixed role 0=SB, 1=BB, 2=BTN.
- PokerGameExpresso consumes a mutable GameInit with arrays of length three,
  duplicates contribution accounting in main_pot/total_bet, deals private cards,
  and requires a separate deal_small_and_big_blind call. SB=1, BB=2, stack=100
  means 50 BB. No tournament lifecycle or heads-up positions exist.
- process_action accepts bet_amount but explicitly ignores it. RAISE opens to
  3*BB or adds max(last_raise,3*BB); raises are capped at four. ALL-IN is always
  advertised even when raising rights should be closed. Call is excluded when
  insufficient chips remain. Showdown side pots can silently skip ineligible
  layers and refill an exhausted deck.
- CFRPlusSolver branches at a traverser decision but calls rollout_until_terminal
  for its descendants. Future traverser decisions inside those rollouts receive
  no learning signals. traverse then samples another continuation. Opponent
  probability products multiply regrets and strategy sums despite sampling those
  opponents already. Regrets are clipped after each update.
- infoset packs u64 phase/role/hand169/board/pot/ratio/SPR/hero-board buckets. Exact
  cards, active seats, and public sequence are absent. History is disabled by
  FAST_TRAINING, otherwise truncated to five events per player.
- Export keeps top three actions, quantizes to bytes, caps visits at 120, and has
  no schema version or game objective. policy.py accepts multiple shapes and
  silently uses uniform play for unknown states. ML distills these probabilities;
  it does not learn CFR advantages. Its source has pre-existing held-out split
  edits and its weights have uncommitted changes; retain these as legacy evidence.
- Parallel workers each update a private regret table for many iterations from
  stale base tables, then merge deltas. This is not frozen-policy sample generation.
- UI duplicates betting, evaluator, buckets and packing in TypeScript. It loads
  the byte policy, constructs [100,100,100] on every newHand, fixes identities to
  roles, and starts on Overview. Unsupported policy states can look like zero
  probabilities rather than unavailable policy coverage.

All ten reported issues were confirmed in their owning modules. The saved
policy and neural weights are incompatible with the replacement and must never
be automatically loaded for current play.

## Replacement contracts

Python owns hand rules, tournament lifecycle, action sizing, and public view.
The UI consumes versioned JSON and canonical action IDs with computed targets;
it has no poker transition logic. Private cards/deck never travel to the browser.
Hand stacks include chips behind; pot equals total contributions until settlement.
Tournament totals include settled stacks or current hand stacks plus pot.
CFR defaults to tournament winner utility, centered as 1-1/N for the winner and
-1/N for others. Hand chip-delta utility is an explicit controlled-subgame mode.
No horizon cutoff is silently assigned a terminal utility: budgets raise errors.
Exact infosets retain public history and hero private observations across hands.
Neural state retains the complete observation sequence, including earlier hands,
for tournament perfect recall. Current hand history is always included in full.

## Algorithm sources

- https://mlanctot.info/files/papers/nips09mccfr.pdf
- https://proceedings.mlr.press/v97/brown19b/brown19b.pdf
- https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/external_sampling_mccfr.py

The multiplayer average pass enumerates the averaging player's own actions and
samples every other player from an explicit full-support uniform distribution.
At each averaging node the sample weight is own strategy reach divided by the
sampled opponent prefix probability. Chance uses its true distribution; this
produces chance-weighted, opponent-reach-independent average strategy estimates.
This is separate from external-sampling regret passes, which sample opponents
from their strategy and do not multiply their probabilities into regret targets.
