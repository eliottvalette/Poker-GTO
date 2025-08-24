# poker_game_expresso.py
"""
3-handed No Limit Texas Hold'em

Cette classe est optimisée pour intiliser une partie de poker en cours.
Dans le but d'effectuer des simulations de jeu pour l'algorithme MCCFR.
"""
import random as rd
from typing import List, Dict, Optional, Tuple
from collections import Counter
import numpy as np
from classes import Player, PlayerAction, GamePhase, Card, SidePot, HandRank, ActionValidator, DECK, card_rank, card_suit
from utils import rank7, rank7_info

DEBUG_OPTI = True
DEBUG_OPTI_ULTIMATE = True

class GameInit:
    stacks_init: List[int]                   # ex: [25, 25, 25]
    total_bets_init: List[int]               # mises en cours (par rôle) sur la main courante
    current_bets_init: List[int]             # mises en cours (par rôle) sur la phase courante
    active_init: List[bool]                  # état actif des joueurs (par rôle)
    has_acted_init: List[bool]                # état agi des joueurs (par rôle)
    main_pot: float                                   # pot courant
    phase: GamePhase                                  # PREFLOP/FLOP/TURN/RIVER
    community_cards: list[Card]                       # visibles

class PokerGameExpresso:
    """
    Classe principale qui gère l'état et la logique du jeu de poker.
    """
    # poker_game_expresso.py (remplace __init__)
    def __init__(self, init: GameInit):
        self.num_players = 3
        self.small_blind = 0.5
        self.big_blind = 1
        self.starting_stack = 100

        self.main_pot = float(init.main_pot)

        self.community_cards = init.community_cards.copy()
        self.side_pots = self._create_side_pots()
        self.remaining_deck = [Card(r, s) for r in range(2, 15) for s in range(4)]
        rd.shuffle(self.remaining_deck)
        # Retire d'éventuelles cartes déjà au board
        known_board = {c.to_int() for c in self.community_cards}
        self.remaining_deck = [c for c in self.remaining_deck if c.to_int() not in known_board]

        self.current_phase = init.phase
        self.number_raise_this_game_phase = 0
        self.last_raiser = None

        self.players = self._initialize_simulated_players(init)

        self.current_maximum_bet = max(p.current_player_bet for p in self.players)
        self.action_validator = ActionValidator(self.players, self.big_blind)
        self.action_history = {p.name: [] for p in self.players}

        self.current_role = 0 # SB

        self.initial_stacks = {p.name: p.stack for p in self.players}
        self.net_stack_changes = {p.name: 0.0 for p in self.players}
        self.final_stacks = {p.name: p.stack for p in self.players}
        
        # Affichage des joueurs et leurs stacks
        for player in self.players:
            player_status = "actif" if player.is_active else "fold"
            if DEBUG_OPTI:
                print(f"[GAME_OPTI] [INIT] Joueur {player.name} (role {player.role}): {player.stack}BB - {player_status}")        
        if DEBUG_OPTI:
            print("========== FIN INITIALISATION ==========\n")


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
            print(f"[GAME_OPTI] [NEXT_PLAYER] On passe du joueur {self.players[initial_role_playing].name} au joueur {self.players[self.current_role].name}")

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

        self.current_maximum_bet = self.small_blind
        self._next_player()

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
        
        # Récupérer les joueurs actifs et all-in
        in_game_players = [p for p in self.players if p.is_active and not p.has_folded]
        all_in_players = [p for p in in_game_players if p.is_all_in]
        
        # Vérifier s'il ne reste qu'un seul joueur actif
        if len(in_game_players) == 1:
            if DEBUG_OPTI_ULTIMATE:
                print("Moving to showdown (only one player remains)")
            self.handle_showdown()
            return

        current_player = self.players[self.current_role]

        # Si au moins un joueur est all-in et qu'au plus un joueur reste non all-in,
        # et que tous les non all-in ont égalisé la mise max, on court-circuite vers le showdown.
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

        # Cas particulier, au PREFLOP, si la BB est limpée, elle doit avoir un droit de parole
        # Vérification d'un cas particulier en phase préflop :
        # En phase préflop, l'ordre d'action est particulier car après avoir posté les blinds
        # l'action se prolonge jusqu'à ce que le joueur en petite blinde (role_position == 0) puisse agir.
        # Même si, en apparence, tous les joueurs ont déjà joué et égalisé la mise maximale,
        # il est nécessaire de laisser le temps au joueur en small blind d'intervenir.
        # C'est pourquoi, si le joueur actif est en position 0 durant le préflop,
        # la méthode retourne False et indique que la phase d'enchères ne peut pas encore être terminée

        # Si tous les joueurs actifs sont all-in, la partie est terminée, on va vers le showdown pour déterminer le vainqueur
        if (len(all_in_players) == len(in_game_players)) and (len(in_game_players) > 1):
            if DEBUG_OPTI_ULTIMATE:
                print("Moving to showdown (all remaining players are all-in)")
            while len(self.community_cards) < 5 and self.remaining_deck:
                self.community_cards.append(self.remaining_deck.pop(0))
            self.handle_showdown()
            return # Ne rien faire d'autre, la partie est terminée
        
        for player in in_game_players:
            # Si le joueur n'a pas encore agi dans la phase, le tour n'est pas terminé
            if not player.has_acted:
                if DEBUG_OPTI_ULTIMATE:
                    print(f'{player.name} n\'a pas encore agi')
                self._next_player()
                return # Ne rien faire de plus, la phase ne peut pas encore être terminée

            # Si le joueur n'a pas égalisé la mise maximale et n'est pas all-in, le tour n'est pas terminé
            if player.current_player_bet < self.current_maximum_bet and not player.is_all_in:
                if DEBUG_OPTI:
                    print('Un des joueurs en jeu n\'a pas égalisé la mise maximale')
                self._next_player()
                return # Ne rien faire de plus, la phase ne peut pas encore être terminée
        
        # Atteindre cette partie du code signifie que la phase est terminée
        if self.current_phase == GamePhase.RIVER:
            if DEBUG_OPTI_ULTIMATE:
                print("River complete - going to showdown")
            self.handle_showdown()
            return # Ne rien faire de plus, la phase ne peut pas encore être terminée
        else:
            self.advance_phase()
            if DEBUG_OPTI_ULTIMATE:
                print(f"[GAME_OPTI] Advanced to {self.current_phase}")
            # Réinitialiser has_acted pour tous les joueurs actifs et non fold au début d'une nouvelle phase
            for p in self.players:
                if p.is_active and not p.has_folded and not p.is_all_in:
                    p.has_acted = False

        # Si l'action revient au dernier raiser, terminer le tour d'enchères
        if self.last_raiser is not None and self.current_role == self.last_raiser:
            if not (self.current_phase == GamePhase.PREFLOP and
                    (current_player.role == 1) and
                    (current_player.current_player_bet == self.big_blind)):
                if self.current_phase == GamePhase.RIVER:
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
                return
        
    def deal_community_cards(self):
        if DEBUG_OPTI:
            print(f"[GAME_OPTI] \n[DISTRIBUTION] Distribution des cartes communes pour phase {self.current_phase}")

        if self.current_phase == GamePhase.PREFLOP:
            raise ValueError(
                "[GAME_OPTI] Erreur d'état : Distribution des community cards pendant le pré-flop."
            )

        if self.current_phase == GamePhase.FLOP:
            if len(self.remaining_deck) < 3:
                raise ValueError("[GAME_OPTI] Deck épuisé pour le flop")
            for _ in range(3):
                self.community_cards.append(self.remaining_deck.pop(0))

        elif self.current_phase in [GamePhase.TURN, GamePhase.RIVER]:
            if not self.remaining_deck:
                raise ValueError(f"[GAME_OPTI] Deck épuisé pour {self.current_phase}")
            self.community_cards.append(self.remaining_deck.pop(0))

        if DEBUG_OPTI:
            print(f"[GAME_OPTI] [DISTRIBUTION] Board: {self.community_cards}")


    def deal_cards(self):
        """
        Distribue deux cartes à chaque joueur actif.
        Réinitialise et mélange le jeu avant la distribution.
        """
        # Deal two cards to each active player
        for player in self.players:
            if not player.is_active:
                continue
            elif player.role == self.hero_role_position:
                player.cards = self.hero_cards
            else:
                player.cards = self.rd_opponents_cards.pop()

        if DEBUG_OPTI:
            for player in self.players:
                print(f"[GAME_OPTI] [CARDS] player : {player.name}, cards : {[card.rank for card in player.cards]} {[card.suit for card in player.cards]}")

    def advance_phase(self):
        """
        Passe à la phase suivante du jeu (préflop -> flop -> turn -> river).
        Distribue les cartes communes appropriées et réinitialise les mises.
        """
        if DEBUG_OPTI_ULTIMATE:
            print(f"[GAME_OPTI] current_phase {self.current_phase}")
        self.last_raiser = None  # Réinitialiser le dernier raiser pour la nouvelle phase
        
        # Normal phase progression
        if self.current_phase == GamePhase.PREFLOP:
            self.current_phase = GamePhase.FLOP
        elif self.current_phase == GamePhase.FLOP:
            self.current_phase = GamePhase.TURN
        elif self.current_phase == GamePhase.TURN:
            self.current_phase = GamePhase.RIVER
        
        # Increment round number when moving to a new phase
        self.number_raise_this_game_phase = 0
        
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
        
        if DEBUG_OPTI_ULTIMATE:
            print(f"[GAME_OPTI] [PHASE] Premier joueur à agir: {self.players[self.current_role].name} (role : {self.current_role})")
            print("========== FIN CHANGEMENT PHASE ==========\n")

    def process_action(self, player: Player, action: PlayerAction, bet_amount: Optional[int] = None):
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
        
        Returns:
            PlayerAction: L'action traitée (pour garder une cohérence dans le type de retour).
        """
        #----- Vérification que c'est bien au tour du joueur de jouer -----
        if player is not self.players[self.current_role]:
            current_turn_player = self.players[self.current_role].name
            raise ValueError(f"[GAME_OPTI] Erreur d'action : Ce n'est pas le tour de {player.name}. "
                             f"C'est au tour de {current_turn_player} d'agir.")

        if not player.is_active or player.is_all_in or player.has_folded or self.current_phase == GamePhase.SHOWDOWN:
            raise ValueError(f"[GAME_OPTI] {player.name} n'était pas censé pouvoir faire une action, ...")

        available_actions = self.action_validator.update_available_actions(player, self.current_maximum_bet, self.number_raise_this_game_phase, self.main_pot, self.current_phase)
        if not any(valid_action.value == action.value for valid_action in available_actions):
            raise ValueError(f"[GAME_OPTI] {player.name} n'a pas le droit de faire cette action, actions valides : {available_actions}")
        #----- Vérification des fonds disponibles -----
        if not player.is_active or player.is_all_in or player.has_folded or self.current_phase == GamePhase.SHOWDOWN:
            raise ValueError(f"[GAME_OPTI] {player.name} n'était pas censé pouvoir faire une action, Raisons : actif = {player.is_active}, all-in = {player.is_all_in}, folded = {player.has_folded}")
        
        if not any(valid_action.value == action.value for valid_action in available_actions):
            raise ValueError(f"[GAME_OPTI] {player.name} n'a pas le droit de faire cette action, actions valides : {available_actions}")
        
        #----- Affichage de débogage (pour le suivi durant l'exécution) -----
        if DEBUG_OPTI_ULTIMATE:
            print(f"[GAME_OPTI] \n=== Action qui va etre effectuée par {player.name} ===")
            print(f"[GAME_OPTI] Joueur actif : {player.is_active}")
            print(f"[GAME_OPTI] Action choisie : {action.value}")
            print(f"[GAME_OPTI] Phase actuelle : {self.current_phase}")
            print(f"[GAME_OPTI] Pot actuel : {self.main_pot}BB")
            print(f"[GAME_OPTI] A agi : {player.has_acted}")
            print(f"[GAME_OPTI] Est all-in : {player.is_all_in}")
            print(f"[GAME_OPTI] Est folded : {player.has_folded}")
            print(f"[GAME_OPTI] Mise maximale actuelle : {self.current_maximum_bet}BB")
            print(f"[GAME_OPTI] Stack du joueur avant action : {player.stack}BB")
            print(f"[GAME_OPTI] Mise actuelle du joueur : {player.current_player_bet}BB")
        
        #----- Traitement de l'action en fonction de son type -----
        if action.value == PlayerAction.FOLD.value:
            # Le joueur se couche il n'est plus actif pour ce tour.
            player.has_folded = True
            if DEBUG_OPTI_ULTIMATE : 
                print(f"[GAME_OPTI] {player.name} se couche (Fold).")
        
        elif action.value == PlayerAction.CHECK.value:
            if DEBUG_OPTI_ULTIMATE : 
                print(f"[GAME_OPTI] {player.name} check.")
        
        elif action.value == PlayerAction.CALL.value:
            if DEBUG_OPTI_ULTIMATE : 
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
            if DEBUG_OPTI_ULTIMATE : 
                print(f"[GAME_OPTI] {player.name} a call {call_amount}BB")

        elif action.value == PlayerAction.RAISE.value:
            if DEBUG_OPTI_ULTIMATE : 
                print(f"[GAME_OPTI] {player.name} raise.")
            # Si aucune mise n'a encore été faite, fixer un minimum raise basé sur la big blind.
            if self.current_maximum_bet == 0:
                min_raise = self.big_blind
            else:
                min_raise = (self.current_maximum_bet - player.current_player_bet) * 2
        
            # Si aucune valeur n'est fournie ou si elle est inférieure au minimum, utiliser le minimum raise.
            if bet_amount is None or (bet_amount < min_raise and action != PlayerAction.ALL_IN):
                bet_amount = min_raise
        
            # Vérifier si le joueur a assez de jetons pour couvrir le montant de raise.
            if player.stack < (bet_amount - player.current_player_bet):
                raise ValueError(
                    f"[GAME_OPTI] Fonds insuffisants pour raise : {player.name} a {player.stack}BB tandis que le montant "
                    f"additionnel requis est {bet_amount - player.current_player_bet}BB. Mise minimum requise : {min_raise}BB."
                )
        
            # Traitement du raise standard
            actual_bet = bet_amount - player.current_player_bet  # Calculer combien de jetons le joueur doit ajouter
            player.stack -= actual_bet
            player.current_player_bet = bet_amount
            self.main_pot += actual_bet
            player.total_bet += actual_bet
            self.current_maximum_bet = bet_amount
            self.number_raise_this_game_phase += 1
            self.last_raiser = self.current_role
            player.is_all_in = player.is_active and (player.stack == 0)
        
            if DEBUG_OPTI_ULTIMATE : 
                print(f"[GAME_OPTI] {player.name} a raise {bet_amount}BB")
        
        # --- Nouvelles actions pot-based ---
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
            # Calcul de la raise additionnelle basée sur le pourcentage du pot
            custom_raise_amount = self.main_pot * percentage
            # La nouvelle mise est : la mise actuelle + montant pour caller + raise additionnel
            computed_bet = player.current_player_bet + custom_raise_amount
        
            if self.current_maximum_bet == 0:
                min_raise = self.big_blind
            else:
                min_raise = (self.current_maximum_bet - player.current_player_bet) * 2
        
            # Vérifier que le montant additionnel respecte le minimum exigé
            if computed_bet - player.current_player_bet < min_raise:
                computed_bet = player.current_player_bet + min_raise
        
            bet_amount = computed_bet
        
            # Vérifier que le joueur a suffisamment de jetons pour cette raise
            if player.stack < (bet_amount - player.current_player_bet):
                raise ValueError(
                    f"[GAME_OPTI] Fonds insuffisants pour raise : {player.name} a {player.stack}BB tandis que le montant "
                    f"additionnel requis est {bet_amount - player.current_player_bet}BB. Mise minimum requise : {min_raise}BB."
                )
        
            # Traitement de la raise pot-based
            actual_bet = bet_amount - player.current_player_bet  # Calcul du supplément à miser
            player.stack -= actual_bet
            player.current_player_bet = bet_amount
            self.main_pot += actual_bet
            player.total_bet += actual_bet
            self.current_maximum_bet = bet_amount
            self.number_raise_this_game_phase += 1
            self.last_raiser = self.current_role
            player.is_all_in = player.is_active and (player.stack == 0)
        
            if DEBUG_OPTI_ULTIMATE : 
                print(f"[GAME_OPTI] {player.name} a raise (pot-based {percentage*100:.0f}%) à {bet_amount}BB")
        
        
        elif action.value == PlayerAction.ALL_IN.value:
            if DEBUG_OPTI_ULTIMATE : 
                print(f"[GAME_OPTI] {player.name} all-in.")
            # Si aucune valeur n'est passée pour bet_amount, on assigne automatiquement tout le stack
            if bet_amount is None:
                bet_amount = player.stack
            elif bet_amount != player.stack:
                raise ValueError(
                    f"[GAME_OPTI] Erreur ALL-IN : {player.name} doit miser exactement tout son stack ({player.stack}BB)."
                )
            
            # Mise à jour de la mise maximale seulement si l'all-in est supérieur
            if bet_amount + player.current_player_bet > self.current_maximum_bet:  # Si le all-in est supérieur à la mise maximale, on met à jour la mise maximale
                self.current_maximum_bet = bet_amount + player.current_player_bet  # On met à jour la mise maximale
                self.number_raise_this_game_phase += 1  # On incrémente le nombre de raise dans la phase
                self.last_raiser = self.current_role  # Enregistrer le all-in comme raise
            
            player.stack -= bet_amount  # On retire le all-in du stack du joueur
            player.current_player_bet += bet_amount  # On ajoute le all-in à la mise du joueur
            self.main_pot += bet_amount  # On ajoute le all-in au pot de la phase
            player.total_bet += bet_amount  # On ajoute le all-in à la mise totale du joueur
            player.is_all_in = True  # On indique que le joueur est all-in
            if DEBUG_OPTI_ULTIMATE : 
                print(f"[GAME_OPTI] {player.name} a all-in {bet_amount}BB")
        
        else:
            raise ValueError(f"[GAME_OPTI] Action invalide : {action}")
        
        player.has_acted = True
        self.check_phase_completion()
        
        # Mise à jour de l'historique des actions du joueur
        action_text = f"{action.value}"
        if action in [PlayerAction.RAISE, PlayerAction.ALL_IN] or action in {
            PlayerAction.RAISE_25_POT,
            PlayerAction.RAISE_50_POT,
            PlayerAction.RAISE_75_POT,
            PlayerAction.RAISE_100_POT,
            PlayerAction.RAISE_150_POT,
            PlayerAction.RAISE_2X_POT,
            PlayerAction.RAISE_3X_POT
        }:
            action_text += f" {bet_amount}BB"
        elif action == PlayerAction.CALL:
            call_amount = self.current_maximum_bet - player.current_player_bet
            action_text += f" {call_amount}BB"
        
        # Add action to player's history
        self.action_history[player.name].append(action_text)
        # Keep only last 5 actions per player
        if len(self.action_history[player.name]) > 5:
            self.action_history[player.name].pop(0)
        
        if DEBUG_OPTI_ULTIMATE:
            print(f"[GAME_OPTI] \n=== Etat de la partie après action de {player.name} ===")
            print(f"[GAME_OPTI] Joueur actif : {player.is_active}")
            print(f"[GAME_OPTI] Action choisie : {action.value}")
            print(f"[GAME_OPTI] Phase actuelle : {self.current_phase}")
            print(f"[GAME_OPTI] Pot actuel : {self.main_pot}BB")
            print(f"[GAME_OPTI] A agi : {player.has_acted}")
            print(f"[GAME_OPTI] Est all-in : {player.is_all_in}")
            print(f"[GAME_OPTI] Est folded : {player.has_folded}")
            print(f"[GAME_OPTI] Mise maximale actuelle : {self.current_maximum_bet}BB")
            print(f"[GAME_OPTI] Stack du joueur avant action : {player.stack}BB")
            print(f"[GAME_OPTI] Mise actuelle du joueur : {player.current_player_bet}BB")

    def evaluate_final_hand(self, player: Player) -> Tuple[HandRank, List[int]]:
        if not player.cards:
            raise ValueError(
                f"[GAME_OPTI] Erreur d'évaluation : le joueur {player.name} n'a pas de cartes pour évaluer sa main. "
                "Assurez-vous que les cartes ont été distribuées correctement."
            )
        """
        Évalue la meilleure main possible d'un joueur.
        """
        if not player.cards:
            raise ValueError("[GAME_OPTI] Cannot evaluate hand - player has no cards")
        
        # Combine les cartes du joueur avec les cartes communes
        all_cards = player.cards + self.community_cards
        # Extrait les valeurs et couleurs de toutes les cartes
        values = [card.rank for card in all_cards]
        suits = [card.suit for card in all_cards]
        
        # Vérifie si une couleur est possible (5+ cartes de même couleur)
        suit_counts = Counter(suits)
        # Trouve la première couleur qui apparaît 5 fois ou plus, sinon None
        flush_suit = next((suit for suit, count in suit_counts.items() if count >= 5), None)
        
        # Si une couleur est possible, on vérifie d'abord les mains les plus fortes
        if flush_suit:
            # Trie les cartes de la couleur par valeur décroissante
            flush_cards = sorted([card for card in all_cards if card.suit == flush_suit], key=lambda x: x.rank, reverse=True)
            flush_values = [card.rank for card in flush_cards]
            
            # Vérifie si on a une quinte flush
            for i in range(len(flush_values) - 4):
                # Vérifie si 5 cartes consécutives de même couleur
                if flush_values[i] - flush_values[i+4] == 4:
                    # Si la plus haute carte est un As, c'est une quinte flush royale
                    if flush_values[i] == 14 and flush_values[i+4] == 10:
                        return (HandRank.ROYAL_FLUSH, [14])
                    # Sinon c'est une quinte flush normale
                    return (HandRank.STRAIGHT_FLUSH, [flush_values[i]])
            
            # Vérifie la quinte flush basse (As-5)
            if set([14,2,3,4,5]).issubset(set(flush_values)):
                return (HandRank.STRAIGHT_FLUSH, [5])
        
        # Compte les occurrences de chaque valeur
        value_counts = Counter(values)
        
        # Vérifie le carré (4 cartes de même valeur)
        if 4 in value_counts.values():
            quads = [v for v, count in value_counts.items() if count == 4][0]
            # Trouve la plus haute carte restante comme kicker
            kicker = max(v for v in values if v != quads)
            return (HandRank.FOUR_OF_A_KIND, [quads, kicker])
        
        # Vérifie le full house (brelan + paire)
        if 3 in value_counts.values():
            # Trouve tous les brelans, triés par valeur décroissante
            trips = sorted([v for v, count in value_counts.items() if count >= 3], reverse=True)
            # Trouve toutes les paires potentielles, y compris les brelans qui peuvent servir de paire
            pairs = []
            for value, count in value_counts.items():
                if count >= 2:  # La carte peut former une paire
                    if count >= 3 and value != trips[0]:  # C'est un second brelan
                        pairs.append(value)
                    elif count == 2:  # C'est une paire simple
                        pairs.append(value)
            
            if pairs:  # Si on a au moins une paire ou un second brelan utilisable comme paire
                return (HandRank.FULL_HOUSE, [trips[0], max(pairs)])
        
        # Vérifie la couleur simple
        if flush_suit:
            flush_cards = sorted([card.rank for card in all_cards if card.suit == flush_suit], reverse=True)
            return (HandRank.FLUSH, flush_cards[:5])
        
        # Vérifie la quinte (5 cartes consécutives)
        unique_values = sorted(set(values), reverse=True)
        for i in range(len(unique_values) - 4):
            if unique_values[i] - unique_values[i+4] == 4:
                return (HandRank.STRAIGHT, [unique_values[i]])
                
        # Vérifie la quinte basse (As-5)
        if set([14,2,3,4,5]).issubset(set(values)):
            return (HandRank.STRAIGHT, [5])
        
        # Vérifie le brelan
        if 3 in value_counts.values():
            # Trouve tous les brelans et sélectionne le plus haut
            trips = max(v for v, count in value_counts.items() if count >= 3)
            # Garde les 2 meilleures cartes restantes comme kickers
            kickers = sorted([v for v in values if v != trips], reverse=True)[:2]
            return (HandRank.THREE_OF_A_KIND, [trips] + kickers)
        
        # Vérifie la double paire
        pairs = sorted([v for v, count in value_counts.items() if count >= 2], reverse=True)
        if len(pairs) >= 2:
            # Garde la meilleure carte restante comme kicker
            kickers = [v for v in values if v not in pairs[:2]]
            kicker = max(kickers) if kickers else 0
            return (HandRank.TWO_PAIR, pairs[:2] + [kicker])
        
        # Vérifie la paire simple
        if pairs:
            # Garde les 3 meilleures cartes restantes comme kickers
            kickers = sorted([v for v in values if v != pairs[0]], reverse=True)[:3]
            return (HandRank.PAIR, [pairs[0]] + kickers)
        
        # Si aucune combinaison, retourne la carte haute avec les 5 meilleures cartes
        return (HandRank.HIGH_CARD, sorted(values, reverse=True)[:5])

    def handle_showdown(self):
        if DEBUG_OPTI_ULTIMATE:
            print("\n=== DÉBUT SHOWDOWN SIMULATION ===")

        self.current_phase = GamePhase.SHOWDOWN
        self.current_maximum_bet = 0

        # Figer les actions
        for player in self.players:
            self.action_validator.update_available_actions(
                player, self.current_maximum_bet, self.number_raise_this_game_phase, self.main_pot, self.current_phase
            )

        active_players = [p for p in self.players if p.is_active and not p.has_folded]
        if DEBUG_OPTI_ULTIMATE:
            print(f"[GAME_OPTI] [SHOWDOWN] Joueurs actifs: {len(active_players)}")

        # Complète le board à 5 cartes
        while len(self.community_cards) < 5:
            if not self.remaining_deck:
                # Reconstruit un pack si besoin (sécurité)
                self.remaining_deck = [Card(r, s) for r in range(2, 15) for s in range(4)]
                rd.shuffle(self.remaining_deck)
                known = {c.to_int() for p in self.players for c in getattr(p, "cards", [])} | {c.to_int() for c in self.community_cards}
                self.remaining_deck = [c for c in self.remaining_deck if c.to_int() not in known]
            self.community_cards.append(self.remaining_deck.pop(0))

        # Donne 2 cartes aux joueurs actifs qui n'en ont pas
        for p in active_players:
            if not p.cards:
                p.cards = [self.remaining_deck.pop(0), self.remaining_deck.pop(0)]
                if DEBUG_OPTI_ULTIMATE:
                    print(f"[GAME_OPTI] [SHOWDOWN] {p.name} reçoit: {p.cards[0]} {p.cards[1]}")

        # Victoire par fold
        if len(active_players) == 1:
            winner = active_players[0]
            winner.stack += self.main_pot
            if DEBUG_OPTI_ULTIMATE:
                print(f"[GAME_OPTI] Victoire par fold - {winner.name} gagne {self.main_pot:.2f}BB")
            self.main_pot = 0
        else:
            # Évaluation Treys: plus GRAND = meilleur (rank7 renvoie -score Treys)
            b0, b1, b2, b3, b4 = [c.to_int() for c in self.community_cards[:5]]
            best = None
            winners = []
            for p in active_players:
                h0, h1 = p.cards[0].to_int(), p.cards[1].to_int()
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
                    if DEBUG_OPTI_ULTIMATE:
                        print(f"[GAME_OPTI] [SHOWDOWN] {w.name} gagne {share:.2f}BB")
                self.main_pot = 0
            else:
                raise RuntimeError("[GAME_OPTI] Aucun joueur n'a gagné au showdown")

        self.net_stack_changes = {p.name: (p.stack - self.initial_stacks.get(p.name, 0)) for p in self.players}
        self.final_stacks = {p.name: p.stack for p in self.players}

        if DEBUG_OPTI_ULTIMATE:
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
        for i in range(6):
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
    

if __name__ == "__main__":
    # Setup d'une main très simple (3-handed, stacks even, aucun board au départ)
    init = GameInit()
    init.stacks_init = [100, 100, 100]        # SB, BB, BTN
    init.total_bets_init = [0, 0, 0]
    init.current_bets_init = [0, 0, 0]
    init.active_init = [True, True, True]
    init.has_acted_init = [False, False, False]
    init.main_pot = 0
    init.phase = GamePhase.PREFLOP
    init.community_cards = []

    game = PokerGameExpresso(init)
    game.deal_small_and_big_blind()

    print("=== Nouvelle main (3-handed) ===")
    print(f"Stacks initiaux: {[p.stack for p in game.players]}  | Pot: {game.main_pot}BB")
    print("Ordre des rôles: 0=SB, 1=BB, 2=BTN")
    print(f"Premier à parler: Player_{game.current_role} (role {game.current_role})\n")

    # Politique ultra simple pour démontrer l'exécution :
    # BTN open 3BB si possible, puis SB et BB foldent => fin de main par fold.
    while game.current_phase != GamePhase.SHOWDOWN:
        p = game.players[game.current_role]

        allowed = game.action_validator.update_available_actions(
            p,
            game.current_maximum_bet,
            game.number_raise_this_game_phase,
            game.main_pot,
            game.current_phase
        )

        if DEBUG_OPTI:
            print(f"[GAME_OPTI] les actions valides sont : {[a.name for a in allowed]}")

        action = rd.choice(allowed)

        if DEBUG_OPTI:
            print(f"[GAME_OPTI] {p.name} fait l'action {action.name}")

        game.process_action(p, action)

    print("\n=== Showdown (main terminée) ===")
    for pl in game.players:
        delta = pl.stack - game.initial_stacks[pl.name]
        sign = "+" if delta >= 0 else ""
        print(f"{pl.name} (role {pl.role}) stack: {pl.stack}BB ({sign}{delta}BB)")
    print(f"Pot final: {game.main_pot}BB\n")
