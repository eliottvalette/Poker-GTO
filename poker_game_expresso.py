# poker_game_expresso.py
"""
3-handed No Limit Texas Hold'em

Cette classe est optimisée pour intiliser une partie de poker en cours.
Dans le but d'effectuer des simulations de jeu pour l'algorithme MCCFR.
"""
import random as rd
from typing import List, Optional
from classes import Player, Card, SidePot
from utils import rank7

FAST_TRAINING = True
DEBUG_OPTI = False or not FAST_TRAINING
DEBUG_OPTI_ULTIMATE = False or not FAST_TRAINING

# Remplacement des Enum par vecteurs ordonnés
HAND_RANKS = [
    "HIGH_CARD", "PAIR", "TWO_PAIR", "THREE_OF_A_KIND",
    "STRAIGHT", "FLUSH", "FULL_HOUSE", "FOUR_OF_A_KIND",
    "STRAIGHT_FLUSH", "ROYAL_FLUSH"
]

PLAYER_ACTIONS = [
    "FOLD", "CHECK", "CALL", "RAISE", "ALL-IN"
]

GAME_PHASES = [
    "PREFLOP", "FLOP", "TURN", "RIVER", "SHOWDOWN"
]

# Pour lookup rapide
HAND_RANKS_IDX = {name: i for i, name in enumerate(HAND_RANKS)}
PLAYER_ACTIONS_IDX = {name: i for i, name in enumerate(PLAYER_ACTIONS)}
GAME_PHASES_IDX = {name: i for i, name in enumerate(GAME_PHASES)}

class GameInit:
    stacks_init: List[int]                   # ex: [25, 25, 25]
    total_bets_init: List[int]               # mises en cours (par rôle) sur la main courante
    current_bets_init: List[int]             # mises en cours (par rôle) sur la phase courante
    active_init: List[bool]                  # état actif des joueurs (par rôle)
    has_acted_init: List[bool]                # état agi des joueurs (par rôle)
    main_pot: float                                   # pot courant
    phase: str                                  # "PREFLOP"/"FLOP"/"TURN"/"RIVER"/"SHOWDOWN"
    community_cards: list[Card]                       # visibles

class PokerGameExpresso:
    """
    Classe principale qui gère l'état et la logique du jeu de poker.
    """
    # poker_game_expresso.py (remplace __init__)
    def __init__(self, init: GameInit):
        self.num_players = 3
        self.small_blind = 1
        self.big_blind = 2
        self.starting_stack = 100

        self.main_pot = float(init.main_pot)

        self.community_cards = init.community_cards.copy()
        self.side_pots = self._create_side_pots()
        self.remaining_deck = [Card(r, s) for r in range(2, 15) for s in range(4)]
        rd.shuffle(self.remaining_deck)
        # Retire d'éventuelles cartes déjà au board
        known_board = {c.id for c in self.community_cards}
        self.remaining_deck = [c for c in self.remaining_deck if c.id not in known_board]

        self.current_phase = init.phase
        self.number_raise_this_game_phase = 0
        self.last_raiser = None
        self.last_raise_amount = self.big_blind

        self.players = self._initialize_simulated_players(init)

        self.current_maximum_bet = max(p.current_player_bet for p in self.players)
        self.action_masks = {0: {a: False for a in PLAYER_ACTIONS}, 1: {a: False for a in PLAYER_ACTIONS}, 2: {a: False for a in PLAYER_ACTIONS}}
        
        self.action_history = {p.name: [] for p in self.players}

        self.current_role = 0 # SB

        self.initial_stacks = {p.name: p.stack for p in self.players}
        self.net_stack_changes = {p.name: 0.0 for p in self.players}
        self.final_stacks = {p.name: p.stack for p in self.players}

        self.deal_cards()
        
        # Affichage des joueurs et leurs stacks
        for player in self.players:
            player_status = "actif" if player.is_active else "FOLD"
            if DEBUG_OPTI:
                print(f"[GAME_OPTI] [INIT] Joueur {player.name} (role {player.role}): {player.stack}BB - {player_status}")      
        if DEBUG_OPTI:
            print("========== FIN INITIALISATION ==========\n")

    def update_available_actions(self, player: Player, current_maximum_bet: float, number_raise_this_game_phase: int, main_pot: float, phase: str):
        player_role = player.role

        if player.is_all_in:
            for action in PLAYER_ACTIONS:
                self.action_masks[player_role][action] = False
            return []

        for action in PLAYER_ACTIONS:
            self.action_masks[player_role][action] = True
        
        if player.current_player_bet < current_maximum_bet:
            self.action_masks[player_role]["CHECK"] = False
        
        if self.action_masks[player_role]["CHECK"]:
            self.action_masks[player_role]["FOLD"] = False
        
        # Gestion de CALL / ALL-IN
        call_amount = current_maximum_bet - player.current_player_bet
        if call_amount <= 0:
            # Pas de mise à suivre → CALL désactivé
            self.action_masks[player_role]["CALL"] = False
        elif call_amount >= player.stack:
            # Le joueur ne peut que faire tapis (ALL-IN est en réalité un call avec tout son stack)
            self.action_masks[player_role]["CALL"] = False
            self.action_masks[player_role]["ALL-IN"] = True
        else:
            # Le joueur peut payer normalement
            self.action_masks[player_role]["CALL"] = True
        
        # Si le joueur ne peut pas couvrir le min-raise, on désactive RAISE plus bas
        if current_maximum_bet == 0:
            min_raise = self.big_blind
        else:
            min_raise = (current_maximum_bet - player.current_player_bet) * 2
        
        if player.stack < min_raise:
            self.action_masks[player_role]["RAISE"] = False

        if number_raise_this_game_phase >= 4:
            self.action_masks[player_role]["RAISE"] = False
        
        """
        pot_raise_actions = [
            PlayerAction.RAISE_25_POT,
            PlayerAction.RAISE_50_POT,
            PlayerAction.RAISE_75_POT,
            PlayerAction.RAISE_100_POT,
            PlayerAction.RAISE_150_POT,
            PlayerAction.RAISE_2X_POT,
            PlayerAction.RAISE_3X_POT
        ]
        raise_percentages = {
            PlayerAction.RAISE_25_POT: 0.25,
            PlayerAction.RAISE_50_POT: 0.50,
            PlayerAction.RAISE_75_POT: 0.75,
            PlayerAction.RAISE_100_POT: 1.00,
            PlayerAction.RAISE_150_POT: 1.50,
            PlayerAction.RAISE_2X_POT: 2.00,
            PlayerAction.RAISE_3X_POT: 3.00
        }

        def pot_to_amount(main_pot, current_max_bet, player_cur_bet, pct):
            call_amt = max(0.0, current_max_bet - player_cur_bet)
            target_to = current_max_bet + pct * (main_pot + call_amt)
            return target_to

        for action in pot_raise_actions:
            if number_raise_this_game_phase >= 4:
                self.masks[player_role][action] = False
            else:
                percentage = raise_percentages[action]
                target_to = pot_to_amount(main_pot, current_maximum_bet, player.current_player_bet, percentage)
                add_required = target_to - player.current_player_bet

                # min raise ≈ 2× le gap à caller si une mise existe, sinon BB
                min_raise = self.big_blind if current_maximum_bet == 0 else 2 * max(0.0, current_maximum_bet - player.current_player_bet)

                if add_required < min_raise or player.stack < add_required:
                    self.masks[player_role][action] = False
        """
        
        
        if phase == "SHOWDOWN":
            for action in PLAYER_ACTIONS:
                self.action_masks[player_role][action] = False

        return [a for a, enabled in self.action_masks[player_role].items() if enabled]

    def deal_cards(self):
        """
        Distribue les cartes aux joueurs actifs qui n'en ont pas
        """
        if DEBUG_OPTI:
            print("[GAME_OPTI] Distribution des cartes privées")
        for player in self.players:
            if player.is_active and not player.has_folded and not player.cards:
                player.cards = [self.remaining_deck.pop(0), self.remaining_deck.pop(0)]
                if DEBUG_OPTI:
                    print(f"[GAME_OPTI] {player.name} reçoit: {player.cards[0]} {player.cards[1]}")


    def _next_player(self):
        """
        Passe au prochain joueur actif et n'ayant pas fold dans le sens horaire.
        Skip les joueurs all-in.
        """
        initial_role_playing = self.current_role
        self.current_role = (self.current_role + 1) % self.num_players
        
        # Vérifier qu'on ne boucle pas indéfiniment
        while (not self.players[self.current_role].is_active or 
               self.players[self.current_role].has_folded or 
               self.players[self.current_role].is_all_in):
            # Ajouter le fait que le joueur a passé son tour dans l'historique
            skipped_player = self.players[self.current_role]
            if skipped_player.has_folded:
                self.action_history[skipped_player.name].append("none")
                # On ne garde que les 5 dernières actions du joueur
                if len(self.action_history[skipped_player.name]) > 5:
                    self.action_history[skipped_player.name].pop(0)
            
            self.current_role = (self.current_role + 1) % self.num_players
            if self.current_role == initial_role_playing:
                # Affiche l'état de chaque joueur pour faciliter le débogage
                for p in self.players:
                    print(f"[GAME_OPTI] players : {p}")
                details = [(p.name, p.is_active, p.has_folded, p.is_all_in, p.has_acted) for p in self.players]
                raise RuntimeError(
                    "[GAME_OPTI] Aucun joueur valide trouvé. Cela signifie que tous les joueurs sont inactifs, foldés ou all-in. "
                    f"[GAME_OPTI] Détails des joueurs : {details}"
                )

        if DEBUG_OPTI:
            print(f"\n[GAME_OPTI] [NEXT_PLAYER] On passe du joueur {self.players[initial_role_playing].name} au joueur {self.players[self.current_role].name}\n")

    def deal_small_and_big_blind(self):
        """
        Méthode à run en début de main pour distribuer automatiquement les blindes
        """
        players = [p for p in self.players if p.is_active]
        sb_player = players[0]
        bb_player = players[1]

        # SB
        if sb_player.stack >= self.small_blind:
            sb_player.stack -= self.small_blind
            self.main_pot += self.small_blind
            sb_player.total_bet = self.small_blind
            sb_player.current_player_bet = self.small_blind
            sb_player.has_acted = False
        else:
            sb_player.is_all_in = True
            sb_player.current_player_bet = sb_player.stack
            self.main_pot += sb_player.stack
            sb_player.total_bet = sb_player.stack
            sb_player.stack = 0
            sb_player.has_acted = True

        if DEBUG_OPTI:
            print(f"[GAME_OPTI] {sb_player.name} a deal la SB : {self.small_blind}BB")

        self.current_maximum_bet = self.small_blind
        self._next_player()
        
        # MAJ du montant de la dernière relance légale
        self.last_raise_amount = self.big_blind

        # BB
        if bb_player.stack >= self.big_blind:
            bb_player.stack -= self.big_blind
            self.main_pot += self.big_blind
            bb_player.total_bet = self.big_blind
            bb_player.current_player_bet = self.big_blind
            bb_player.has_acted = False
        else:
            bb_player.is_all_in = True
            bb_player.current_player_bet = bb_player.stack
            self.main_pot += bb_player.stack
            bb_player.total_bet = bb_player.stack
            bb_player.stack = 0
            bb_player.has_acted = True

        if DEBUG_OPTI:
            print(f"[GAME_OPTI] {bb_player.name} a deal la BB : {self.big_blind}BB")

        self.current_maximum_bet = self.big_blind
        self._next_player()
        
    def check_phase_completion(self):
        """
        Vérifie si le tour d'enchères actuel est terminé et gère la progression du jeu.
        
        Le tour est terminé quand :
        1. Tous les joueurs actifs ont agi
        2. Tous les joueurs ont égalisé la mise maximale (ou sont all-in)
        3. Cas particuliers : un seul joueur reste, tous all-in, ou BB preflop
        """

        # Récupérer les joueurs actifs
        in_game_players = [p for p in self.players if p.is_active and not p.has_folded]
        all_in_players = [p for p in in_game_players if p.is_all_in]

        # Victoire directe si un seul joueur actif
        if len(in_game_players) == 1:
            if DEBUG_OPTI_ULTIMATE:
                print("Moving to showdown (only one player remains)")
            self.handle_showdown()
            return

        # Cas all-in : showdown forcé si plus de mise possible
        if any(p.is_all_in for p in in_game_players):
            everyone_capped = all(
                p.is_all_in or p.current_player_bet == self.current_maximum_bet
                for p in in_game_players
            )
            at_most_one_live = sum(1 for p in in_game_players if not p.is_all_in) <= 1
            if everyone_capped and at_most_one_live and len(in_game_players) > 1:
                if DEBUG_OPTI_ULTIMATE:
                    print("Moving to showdown (all-in present, no further betting possible)")
                while len(self.community_cards) < 5 and self.remaining_deck:
                    self.community_cards.append(self.remaining_deck.pop(0))
                self.handle_showdown()
                return

        # Tous les joueurs restants sont all-in
        if (len(all_in_players) == len(in_game_players)) and (len(in_game_players) > 1):
            if DEBUG_OPTI_ULTIMATE:
                print("Moving to showdown (all remaining players are all-in)")
            while len(self.community_cards) < 5 and self.remaining_deck:
                self.community_cards.append(self.remaining_deck.pop(0))
            self.handle_showdown()
            return

        # Vérification des actions des joueurs
        for player in in_game_players:
            if not player.has_acted:
                if DEBUG_OPTI_ULTIMATE:
                    print(f"{player.name} n'a pas encore agi")
                self._next_player()
                return
            if player.current_player_bet < self.current_maximum_bet and not player.is_all_in:
                if DEBUG_OPTI_ULTIMATE:
                    print("Un des joueurs en jeu n'a pas égalisé la mise maximale")
                self._next_player()
                return

        # Ici, toutes les conditions pour avancer la phase sont remplies
        if self.current_phase == "RIVER":
            if DEBUG_OPTI_ULTIMATE:
                print("River complete - going to showdown")
            self.handle_showdown()
        else:
            self.advance_phase()
            if DEBUG_OPTI_ULTIMATE:
                print(f"[GAME_OPTI] Advanced to {self.current_phase}")
            for p in self.players:
                if p.is_active and not p.has_folded and not p.is_all_in:
                    p.has_acted = False

        
    def deal_community_cards(self):
        if DEBUG_OPTI:
            print(f"[GAME_OPTI] \n[DISTRIBUTION] Distribution des cartes communes pour phase {self.current_phase}")

        if self.current_phase == "PREFLOP":
            raise ValueError(
                "[GAME_OPTI] Erreur d'état : Distribution des community cards pendant le pré-flop."
            )

        if self.current_phase == "FLOP":
            if len(self.remaining_deck) < 3:
                raise ValueError("[GAME_OPTI] Deck épuisé pour le flop")
            for _ in range(3):
                self.community_cards.append(self.remaining_deck.pop(0))

        elif self.current_phase in ["TURN", "RIVER"]:
            if not self.remaining_deck:
                raise ValueError(f"[GAME_OPTI] Deck épuisé pour {self.current_phase}")
            self.community_cards.append(self.remaining_deck.pop(0))

        if DEBUG_OPTI:
            print(f"[GAME_OPTI] [DISTRIBUTION] Board: {self.community_cards}")

    def advance_phase(self):
        """
        Passe à la phase suivante du jeu (préflop -> flop -> turn -> river).
        Distribue les cartes communes appropriées et réinitialise les mises.
        """
        if DEBUG_OPTI:
            print(f"[GAME_OPTI] current_phase {self.current_phase}")
        self.last_raiser = None  # Réinitialiser le dernier raiser pour la nouvelle phase
        
        # Normal phase progression
        if self.current_phase == "PREFLOP":
            self.current_phase = "FLOP"
        elif self.current_phase == "FLOP":
            self.current_phase = "TURN"
        elif self.current_phase == "TURN":
            self.current_phase = "RIVER"
        
        # Increment round number when moving to a new phase
        self.number_raise_this_game_phase = 0
        
        # Reset last raise amount for new phase
        self.last_raise_amount = self.big_blind
        
        # Deal community cards for the new phase
        self.deal_community_cards()
        
        # Réinitialiser les mises pour la nouvelle phase
        self.current_maximum_bet = 0
        for player in self.players:
            if player.is_active:
                player.current_player_bet = 0
                if not player.has_folded and not player.is_all_in:
                    player.has_acted = False  # Réinitialisation du flag
        
        # postflop : SB parle en premier (puis BB, puis BTN)
        self.current_role = 0 # SB
        while (not self.players[self.current_role].is_active or 
               self.players[self.current_role].has_folded or 
               self.players[self.current_role].is_all_in):
            self.current_role = (self.current_role + 1) % self.num_players
        
        if DEBUG_OPTI:
            print(f"[GAME_OPTI] [PHASE] Premier joueur à agir: {self.players[self.current_role].name} (role : {self.current_role})")
            print("========== FIN CHANGEMENT PHASE ==========\n")

    def process_action(self, player: Player, action: str, bet_amount: Optional[int] = None):
        """
        Traite l'action d'un joueur, met à jour l'état du jeu et gère la progression du tour.

        Cette méthode réalise plusieurs vérifications essentielles :
        - S'assurer que le joueur dispose de suffisamment de fonds.
        - Interrompre le traitement en cas de phase SHOWDOWN.
        - Construire un historique des actions pour le suivi.
        - Gérer distinctement les différents types d'actions : FOLD, CHECK, CALL, RAISE et ALL_IN.
        - Mettre à jour le pot, les mises des joueurs et la mise maximale en cours.
        - Traiter les situations d'all-in et créer des side pots le cas échéant.
        - Déterminer, à l'issue de l'action, si le tour d'enchères est clôturé ou s'il faut passer au joueur suivant.
        
        - bet_amount n'est pas utilisé pour le momentcar la raise est systématiquement la min-raise.

        Returns:
            PlayerAction: L'action traitée (pour garder une cohérence dans le type de retour).
        """
        #----- Vérification que c'est bien au tour du joueur de jouer -----
        if player is not self.players[self.current_role]:
            current_turn_player = self.players[self.current_role].name
            raise ValueError(f"[GAME_OPTI] Erreur d'action : Ce n'est pas le tour de {player.name}. "
                             f"C'est au tour de {current_turn_player} d'agir.")

        if not player.is_active or player.is_all_in or player.has_folded or self.current_phase == "SHOWDOWN":
            raise ValueError(f"[GAME_OPTI] {player.name} n'était pas censé pouvoir faire une action, ...")

        available_actions = self.update_available_actions(player, self.current_maximum_bet, self.number_raise_this_game_phase, self.main_pot, self.current_phase)
        if not any(valid_action == action for valid_action in available_actions):
            raise ValueError(f"[GAME_OPTI] {player.name} n'a pas le droit de faire cette action, actions valides : {available_actions}")
           
        #----- Affichage de débogage (pour le suivi durant l'exécution) -----
        if DEBUG_OPTI_ULTIMATE:
            print(f"[GAME_OPTI] \n=== Action qui va etre effectuée par {player.name} ===")
            print(f"[GAME_OPTI] Joueur actif : {player.is_active}")
            print(f"[GAME_OPTI] Action choisie : {action}")
            print(f"[GAME_OPTI] Phase actuelle : {self.current_phase}")
            print(f"[GAME_OPTI] Pot actuel : {self.main_pot}BB")
            print(f"[GAME_OPTI] A agi : {player.has_acted}")
            print(f"[GAME_OPTI] Est all-in : {player.is_all_in}")
            print(f"[GAME_OPTI] Est folded : {player.has_folded}")
            print(f"[GAME_OPTI] Mise maximale actuelle : {self.current_maximum_bet}BB")
            print(f"[GAME_OPTI] Stack du joueur avant action : {player.stack}BB")
            print(f"[GAME_OPTI] Mise actuelle du joueur : {player.current_player_bet}BB")
        
        #----- Traitement de l'action en fonction de son type -----
        if action == "FOLD":
            # Le joueur se couche il n'est plus actif pour ce tour.
            player.has_folded = True
            if DEBUG_OPTI : 
                print(f"[GAME_OPTI] {player.name} se couche (Fold).")
        
        elif action == "CHECK":
            if DEBUG_OPTI : 
                print(f"[GAME_OPTI] {player.name} check.")
        
        elif action == "CALL":
            if DEBUG_OPTI : 
                print(f"[GAME_OPTI] {player.name} call.")
            call_amount = self.current_maximum_bet - player.current_player_bet
            if call_amount > player.stack: 
                print(f"[GAME_OPTI] {player.name} a {player.stack}BB tandis que le montant "
                    f"additionnel requis est {call_amount}BB. Mise minimum requise : {self.current_maximum_bet}BB.")
                raise ValueError(f"[GAME_OPTI] {player.name} n'a pas assez de jetons pour suivre la mise maximale, il n'aurait pas du avoir le droit de call")
        
            player.stack -= call_amount
            player.current_player_bet += call_amount
            self.main_pot += call_amount
            player.total_bet += call_amount
            if player.stack == 0:
                player.is_all_in = True
            if DEBUG_OPTI : 
                print(f"[GAME_OPTI] {player.name} a call {call_amount}BB")

        elif action == "RAISE":
            if DEBUG_OPTI:
                print(f"[GAME_OPTI] {player.name} raise.")

            prev_max = self.current_maximum_bet
            gap = max(0.0, prev_max - player.current_player_bet)

            # min raise-to (valeur ABSOLUE à atteindre)
            if prev_max == 0:
                raise_amount = self.big_blind
            else:
                # au moins la dernière taille de relance légale (classique NLHE)
                raise_amount = prev_max + max(self.last_raise_amount, self.big_blind)

            # impossible de « descendre » sous sa mise actuelle
            raise_amount = max(raise_amount, player.current_player_bet)

            add_required = raise_amount - player.current_player_bet
            if add_required <= 0:
                raise ValueError("[GAME_OPTI] Raise invalide (montant non positif).")
            if add_required > player.stack:
                raise ValueError("[GAME_OPTI] Fonds insuffisants pour raise.")

            player.stack -= add_required
            player.current_player_bet = raise_amount
            self.main_pot += add_required

            # MAJ des compteurs de la phase
            self.number_raise_this_game_phase += 1
            self.last_raiser = self.current_role
            self.last_raise_amount = raise_amount - prev_max
            self.current_maximum_bet = raise_amount
            player.total_bet += add_required
            player.is_all_in = (player.stack == 0)

            if DEBUG_OPTI:
                print(f"[GAME_OPTI] {player.name} a raise à {raise_amount}BB")

        elif action == "ALL-IN":
            if DEBUG_OPTI : 
                print(f"[GAME_OPTI] {player.name} all-in.")
            all_in_amount = player.stack
            
            # Mise à jour de la mise maximale seulement si l'all-in est supérieur
            if all_in_amount + player.current_player_bet > self.current_maximum_bet:  # Si le all-in est supérieur à la mise maximale, on met à jour la mise maximale
                self.current_maximum_bet = all_in_amount + player.current_player_bet  # On met à jour la mise maximale
                self.number_raise_this_game_phase += 1  # On incrémente le nombre de raise dans la phase
                self.last_raiser = self.current_role  # Enregistrer le all-in comme raise
            
            player.stack -= all_in_amount  # On retire le all-in du stack du joueur
            player.current_player_bet += all_in_amount  # On ajoute le all-in à la mise du joueur
            self.main_pot += all_in_amount  # On ajoute le all-in au pot de la phase
            player.total_bet += all_in_amount  # On ajoute le all-in à la mise totale du joueur
            player.is_all_in = True  # On indique que le joueur est all-in
            if DEBUG_OPTI : 
                print(f"[GAME_OPTI] {player.name} a all-in {all_in_amount}BB")
        
        else:
            raise ValueError(f"[GAME_OPTI] Action invalide : {action}")
        
        # --- Nouvelles actions pot-based ---
        """
        elif action.value in {
            PlayerAction.RAISE_25_POT.value,
            PlayerAction.RAISE_50_POT.value,
            PlayerAction.RAISE_75_POT.value,
            PlayerAction.RAISE_100_POT.value,
            PlayerAction.RAISE_150_POT.value,
            PlayerAction.RAISE_2X_POT.value,
            PlayerAction.RAISE_3X_POT.value
        }:
            raise_percentages = {
                PlayerAction.RAISE_25_POT.value: 0.25,
                PlayerAction.RAISE_50_POT.value: 0.50,
                PlayerAction.RAISE_75_POT.value: 0.75,
                PlayerAction.RAISE_100_POT.value: 1.00,
                PlayerAction.RAISE_150_POT.value: 1.50,
                PlayerAction.RAISE_2X_POT.value: 2.00,
                PlayerAction.RAISE_3X_POT.value: 3.00
            }
            percentage = raise_percentages[action.value]

            call_amt = max(0.0, self.current_maximum_bet - player.current_player_bet)
            target_to = self.current_maximum_bet + percentage * (self.main_pot + call_amt)

            # min raise ≈ 2× le gap à caller si une mise existe, sinon BB
            min_raise = self.big_blind if self.current_maximum_bet == 0 else 2 * call_amt
            if target_to - player.current_player_bet < min_raise:
                target_to = player.current_player_bet + min_raise

            bet_amount = target_to

            if player.stack < (bet_amount - player.current_player_bet):
                raise ValueError(
                    f"[GAME_OPTI] Fonds insuffisants pour raise : {player.name} a {player.stack}BB, "
                    f"requis {bet_amount - player.current_player_bet}BB."
                )

            actual_bet = bet_amount - player.current_player_bet
            player.stack -= actual_bet
            player.current_player_bet = bet_amount
            self.main_pot += actual_bet
            player.total_bet += actual_bet
            self.current_maximum_bet = bet_amount
            self.number_raise_this_game_phase += 1
            self.last_raiser = self.current_role
            player.is_all_in = player.is_active and (player.stack == 0)

            if DEBUG_OPTI_ULTIMATE:
                print(f"[GAME_OPTI] {player.name} a raise (pot-based {percentage*100:.0f}%) à {bet_amount}BB")
        """
        
        player.has_acted = True
        self.check_phase_completion()
        
        # Mise à jour de l'historique des actions du joueur
        """
        elif action in {
            PlayerAction.RAISE_25_POT,
            PlayerAction.RAISE_50_POT,
            PlayerAction.RAISE_75_POT,
            PlayerAction.RAISE_100_POT,
            PlayerAction.RAISE_150_POT,
            PlayerAction.RAISE_2X_POT,
            PlayerAction.RAISE_3X_POT
        }:
            action_text += f" {bet_amount}BB"
        """
        if not FAST_TRAINING:
            action_text = f"{action}"
            if action == "RAISE":
                action_text += f" {raise_amount}BB"
            elif action == "ALL-IN":
                action_text += f" {all_in_amount}BB"
            elif action == "CALL":
                action_text += f" {call_amount}BB"
            self.action_history[player.name].append(action_text)
            if len(self.action_history[player.name]) > 5:
                self.action_history[player.name].pop(0)
        
        if DEBUG_OPTI_ULTIMATE:
            print(f"[GAME_OPTI] \n=== Etat de la partie après action de {player.name} ===")
            print(f"[GAME_OPTI] Joueur actif : {player.is_active}")
            print(f"[GAME_OPTI] Action choisie : {action}")
            print(f"[GAME_OPTI] Phase actuelle : {self.current_phase}")
            print(f"[GAME_OPTI] Pot actuel : {self.main_pot}BB")
            print(f"[GAME_OPTI] A agi : {player.has_acted}")
            print(f"[GAME_OPTI] Est all-in : {player.is_all_in}")
            print(f"[GAME_OPTI] Est folded : {player.has_folded}")
            print(f"[GAME_OPTI] Mise maximale actuelle : {self.current_maximum_bet}BB")
            print(f"[GAME_OPTI] Stack du joueur avant action : {player.stack}BB")
            print(f"[GAME_OPTI] Mise actuelle du joueur : {player.current_player_bet}BB")

    def handle_showdown(self):
        if DEBUG_OPTI:
            print("\n=== DÉBUT SHOWDOWN SIMULATION ===")

        self.current_phase = "SHOWDOWN"
        self.current_maximum_bet = 0

        # Figer les actions
        for player in self.players:
            self.update_available_actions(
                player, self.current_maximum_bet, self.number_raise_this_game_phase, self.main_pot, self.current_phase
            )

        active_players = [p for p in self.players if p.is_active and not p.has_folded]
        if DEBUG_OPTI:
            print(f"[GAME_OPTI] [SHOWDOWN] Joueurs actifs: {len(active_players)}")

        # Complète le board à 5 cartes
        while len(self.community_cards) < 5:
            if not self.remaining_deck:
                # Reconstruit un pack si besoin (sécurité)
                self.remaining_deck = [Card(r, s) for r in range(2, 15) for s in range(4)]
                rd.shuffle(self.remaining_deck)
                known = {c.id for p in self.players for c in getattr(p, "cards", [])} | {c.id for c in self.community_cards}
                self.remaining_deck = [c for c in self.remaining_deck if c.id not in known]
            self.community_cards.append(self.remaining_deck.pop(0))

        # Victoire par fold
        if len(active_players) == 1:
            winner = active_players[0]
            winner.stack += self.main_pot
            if DEBUG_OPTI:
                print(f"[GAME_OPTI] Victoire par fold - {winner.name} gagne {self.main_pot:.2f}BB")
            self.main_pot = 0
        else:
            # Évaluation Treys: plus GRAND = meilleur (rank7 renvoie -score Treys)
            b0, b1, b2, b3, b4 = [c.id for c in self.community_cards[:5]]
            best = None
            winners = []
            for p in active_players:
                h0, h1 = p.cards[0].id, p.cards[1].id
                s = rank7((h0, h1, b0, b1, b2, b3, b4))
                if best is None or s > best:
                    best = s
                    winners = [p]
                elif s == best:
                    winners.append(p)

            if winners:
                share = self.main_pot / len(winners)
                for w in winners:
                    w.stack += share
                    if DEBUG_OPTI:
                        print(f"[GAME_OPTI] [SHOWDOWN] {w.name} gagne {share:.2f}BB")
                self.main_pot = 0
            else:
                raise RuntimeError("[GAME_OPTI] Aucun joueur n'a gagné au showdown")

        self.net_stack_changes = {p.name: (p.stack - self.initial_stacks.get(p.name, 0)) for p in self.players}
        self.final_stacks = {p.name: p.stack for p in self.players}

        if DEBUG_OPTI:
            print("[SHOWDOWN] Stacks finaux:")
            for p in self.players:
                delta = self.net_stack_changes[p.name]
                sign = "+" if delta >= 0 else ""
                print(f"[GAME_OPTI] [SHOWDOWN] {p.name}: {p.stack:.2f}BB ({sign}{delta:.2f}BB)")
            print("========== FIN SHOWDOWN ==========\n")

    def _create_side_pots(self) -> List[SidePot]:
        """
        Crée 6 side pots vierges.
        
        Returns:
            List[SidePot]: Liste de 6 side pots vierges
        """

        side_pots = []
        for i in range(2):
            side_pots.append(SidePot(id=i))

        return side_pots

    def _distribute_side_pots(self, in_game_players: List[Player], side_pots: List[SidePot], main_pot: float):
        """
        Répartit les surplus en side pots.

        Pour la répartition en side pots, on laisse les joueurs - qui ne seront pas capable d'atteindre le maxbet - all-in dans le main pot 
        On attend la fin de la phase.
        Si les joueurs les plus pauvres sont all-in et que les plus riches sont soit all-in aussi soit à un bet égal au bet_maximum, 
        La phase est terminée et on réparti les surplus en side pots.
        
        _distribute_side_pots est appelée sachant qu'on moins un joueur est all-in, et que tous les non-all-in égalisent la mise maximale.
        side_pot_list est une liste de 4 SidePot, on verra d'après la logique suivante que maximum 4 SidePots distincts seront nécessaires pour une partie à 6 joueurs.

        Exemple : J1 et J2 sont pauvres et sont all-in. J3 et J4 sont plus riches qu'eux, non all-in avec des bets égaux à la mise maximale.
        Notons s_i la mise du joueur i. s_1 < s_2 < s_3 = s_4.
        J1 met toute sa mise dans le main pot.
        J2 met s_1 dans le main pot et met s_2 - s_1 dans le premier side pot.
        J3 et J4 mettent s_1 dans le main pot, puis s_2 - s_1 dans le premier side pot.
        Il leur reste s_3 - s_2 qu'il mettent dans le deuxième side pot.        
        """
        ordered_players = sorted(in_game_players, key=lambda x: x.current_player_bet)
        ordered_bets = [p.current_player_bet for p in ordered_players]
        
        nb_equal_diff = 0
        for i in range(len(ordered_players)):

            diff_bet = ordered_bets[i] - (ordered_bets[i+1] if i < len(ordered_players) - 1 else 0)
            if diff_bet < 0:
                for player in ordered_players[i-nb_equal_diff:]:
                    side_pots[i].contributions_dict[player] = diff_bet
                    ordered_bets[i] -= diff_bet
                side_pots[i].sum_of_contributions = diff_bet * (len(ordered_players) - i - nb_equal_diff +1)
                nb_equal_diff = 0
            else:
                nb_equal_diff +=1
        
        return side_pots

    def _initialize_simulated_players(self, init: GameInit):
        """
        Initialise 6 joueurs simulés pour une partie MCCFR.
        """
        players = []
        
        # Extraction des stacks 
        stacks = init.stacks_init
        
        # Extraction des mises de ma main actuelle
        total_bets = init.total_bets_init

        # Extraction des mises de la phase courante
        current_bets = init.current_bets_init
        
        # Extraction de l'état actif des joueurs 
        active_states = init.active_init
        
        # Extraction des actions effectuées
        has_acted_states = init.has_acted_init
        
        # ordre fixe par rôle: 0=SB, 1=BB, 2=BTN
        for i in range(3):
            player = Player(name=f"Player_{i}", stack=stacks[i])
            player.role = i                      
            player.is_active = active_states[i]
            player.has_folded = not active_states[i]
            player.is_all_in = player.is_active and (player.stack == 0)
            player.current_player_bet = current_bets[i]
            player.total_bet = total_bets[i]
            player.cards = []
            player.has_acted = has_acted_states[i]
            players.append(player)
        
        return players

    def _round_value(self, value, decimals=4):
        """Arrondit une valeur à un nombre spécifié de décimales pour éviter les erreurs de précision."""
        return round(value, decimals)
    
    def snapshot(self):
        # Copie *légère* des champs mutés par process_action/phase/showdown
        players_state = [
            (p.stack, p.current_player_bet, p.total_bet,
             p.is_active, p.has_folded, p.is_all_in, p.has_acted)
            for p in self.players
        ]
        return {
            "current_phase": self.current_phase,
            "number_raise_this_game_phase": self.number_raise_this_game_phase,
            "last_raiser": self.last_raiser,
            "last_raise_amount": self.last_raise_amount,
            "current_role": self.current_role,
            "current_maximum_bet": self.current_maximum_bet,
            "main_pot": self.main_pot,
            "players": players_state,
            "community_cards": list(self.community_cards),   # copie rapide (≤5)
            "remaining_deck": list(self.remaining_deck),     # 52 max
            "net_stack_changes": dict(self.net_stack_changes),
            "final_stacks": dict(self.final_stacks),
        }

    def restore(self, snap):
        self.current_phase = snap["current_phase"]
        self.number_raise_this_game_phase = snap["number_raise_this_game_phase"]
        self.last_raiser = snap["last_raiser"]
        self.last_raise_amount = snap["last_raise_amount"]
        self.current_role = snap["current_role"]
        self.current_maximum_bet = snap["current_maximum_bet"]
        self.main_pot = snap["main_pot"]
        for p, st in zip(self.players, snap["players"]):
            (p.stack, p.current_player_bet, p.total_bet,
             p.is_active, p.has_folded, p.is_all_in, p.has_acted) = st
        self.community_cards = snap["community_cards"]
        self.remaining_deck = snap["remaining_deck"]
        self.net_stack_changes = snap["net_stack_changes"]
        self.final_stacks = snap["final_stacks"]


if __name__ == "__main__":
    # Setup d'une main très simple (3-handed, stacks even, aucun board au départ)
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

    print("=== Nouvelle main (3-handed) ===")
    print(f"Stacks initiaux: {[p.stack for p in game.players]}  | Pot: {game.main_pot}BB")
    print("Ordre des rôles: 0=SB, 1=BB, 2=BTN")
    print(f"Premier à parler: Player_{game.current_role} (role {game.current_role})\n")

    # Politique ultra simple pour démontrer l'exécution :
    # BTN open 3BB si possible, puis SB et BB foldent => fin de main par fold.
    while game.current_phase != "SHOWDOWN":
        p = game.players[game.current_role]

        allowed = game.update_available_actions(
            p,
            game.current_maximum_bet,
            game.number_raise_this_game_phase,
            game.main_pot,
            game.current_phase
        )

        if DEBUG_OPTI:
            print(f"[GAME_OPTI] les actions valides sont : {[a for a in allowed]}")

        action = rd.choice(allowed)

        if DEBUG_OPTI:
            print(f"[GAME_OPTI] {p.name} fait l'action {action}")

        game.process_action(p, action)

    print("\n=== Showdown (main terminée) ===")
    for player in game.players:
        delta = player.stack - game.initial_stacks[player.name]
        sign = "+" if delta >= 0 else ""
        print(f"{player.name} (role {player.role}) stack: {player.stack}BB ({sign}{delta}BB)")
    
    for player in game.players:
        print(f"Cartes du joueur {player.name} : [{player.cards[0]}, {player.cards[1]}]")

    print(f"Cartes communes : ")
    community_cards_str = "["
    for card in game.community_cards:
        community_cards_str += f"{card} "
    community_cards_str += "]"
    print(community_cards_str)