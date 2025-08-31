# run_game_with_keys.py
import random as rd
from poker_game_expresso import PokerGameExpresso, GameInit
from infoset import build_infoset_key_fast

# -------------------- Demo: run one game and print the key each step --------------------
def main():
    init = GameInit()
    init.stacks_init = [100, 100, 100]        # SB, BB, BTN
    init.total_bets_init = [0, 0, 0]
    init.current_bets_init = [0, 0, 0]
    init.active_init = [True, True, True]
    init.has_acted_init = [False, False, False]
    init.main_pot = 0
    init.phase = "PREFLOP"
    init.community_cards = []

    game = PokerGameExpresso(init)
    game.deal_small_and_big_blind()

    step = 0
    while game.current_phase != "SHOWDOWN":
        hero = game.players[game.current_role]

        # Affiche la clé avant l'action du joueur courant
        dense = build_infoset_key_fast(game, hero)
        print(f"[STEP {step:02d}] {hero.name} -> {readable} | key=0x{dense:016X}")


        allowed = game.update_available_actions(
            hero,
            game.current_maximum_bet,
            game.number_raise_this_game_phase,
            game.main_pot,
            game.current_phase
        )
        # Politique aléatoire pour faire tourner la partie
        action = "CHECK" if "CHECK" in allowed else "CALL" if "CALL" in allowed else rd.choice(allowed)
        game.process_action(hero, action)
        step += 1

    print("\n[END] Showdown atteint.")

if __name__ == "__main__":
    main()
