# classes.py
# ------------------------------------------------------------
# Classes simplifiées pour la gestion des cartes et du deck
# ------------------------------------------------------------

class Card:
    """Représente une carte avec son rang et sa couleur"""
    def __init__(self, rank, suit):
        self.rank = rank  # 2..14 (A=14)
        self.suit = suit  # 0..3
    
    def to_int(self):
        """Convertit la carte en sa représentation entière (0..51)"""
        return (self.rank - 2) * 4 + self.suit
    
    def __str__(self):
        ranks = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T', 
                9: '9', 8: '8', 7: '7', 6: '6', 5: '5', 
                4: '4', 3: '3', 2: '2'}
        suits = {0: '♠', 1: '♥', 2: '♦', 3: '♣'}
        return f"{ranks[self.rank]}{suits[self.suit]}"


class Deck:
    """Représente un deck de 52 cartes"""
    
    def __init__(self):
        self.cards = [Card(r, s) for r in range(2, 15) for s in range(4)]
    
    def get_card(self, rank, suit):
        """Retourne la carte avec le rang et la couleur spécifiés"""
        return Card(rank, suit)
    
    def all_starting_combos(self):
        """Retourne tous les combos de départ possibles (1326 combos)"""
        combos = []
        for i in range(52):
            for j in range(i+1, 52):
                combos.append((self.cards[i], self.cards[j]))
        return combos
    
    def combo_to_169(self, c1, c2):
        """Convertit un combo en notation 169 (AKs, AKo, etc.)"""
        RANK_TO_STR = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T', 
                       9: '9', 8: '8', 7: '7', 6: '6', 5: '5', 
                       4: '4', 3: '3', 2: '2'}
        
        r1, r2 = c1.rank, c2.rank
        s1, s2 = c1.suit, c2.suit
        hi, lo = max(r1, r2), min(r1, r2)
        
        if r1 == r2:
            return f"{RANK_TO_STR[hi]}{RANK_TO_STR[lo]}"
        
        suited = (s1 == s2)
        if hi == r1:
            top, bot = r1, r2
        else:
            top, bot = r2, r1
        
        return f"{RANK_TO_STR[top]}{RANK_TO_STR[bot]}{'s' if suited else 'o'}"


# Fonctions utilitaires simples
def card(rank, suit):
    """Retourne l'entier de la carte"""
    return Card(rank, suit).to_int()

def card_rank(c):
    """Retourne le rang de la carte"""
    return (c // 4) + 2

def card_suit(c):
    """Retourne la couleur de la carte"""
    return c % 4

def all_starting_combos():
    """Retourne tous les combos de départ possibles en entiers"""
    return [(i, j) for i in range(52) for j in range(i+1, 52)]

def combo_to_169(c1, c2):
    """Convertit un combo en notation 169"""
    card1 = Card(card_rank(c1), card_suit(c1))
    card2 = Card(card_rank(c2), card_suit(c2))
    deck = Deck()
    return deck.combo_to_169(card1, card2)

def filter_combos_excluding(combos, blocked):
    """Filtre les combos en excluant les cartes bloquées"""
    out = []
    for a, b in combos:
        if a in blocked or b in blocked:
            continue
        out.append((a, b))
    return out

# Instances globales
DECK = [card(r, s) for r in range(2, 15) for s in range(4)]
ALL_COMBOS = all_starting_combos()








# ------------------------------------------------------------
# Classes de poker_game
# ------------------------------------------------------------
from enum import Enum
from typing import Tuple, List
from typing import Dict

class HandRank(Enum):
    """
    Énumération des combinaisons possibles au poker, de la plus faible à la plus forte.
    """
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    ROYAL_FLUSH = 9

class PlayerAction(Enum):
    """
    Énumération des actions possibles pour un joueur pendant son tour.
    """
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    # -----------------------------------------------------
    RAISE = "raise" # Raise minimum (2x la mise précédente)
    RAISE_25_POT = "raise-25%"     # Raise de 25% du pot
    RAISE_50_POT = "raise-50%"     # Raise de 50% du pot
    RAISE_75_POT = "raise-75%"     # Raise de 75% du pot
    RAISE_100_POT = "raise-100%"   # Raise égal au pot
    RAISE_150_POT = "raise-150%"   # Raise de 150% du pot
    RAISE_2X_POT = "raise-200%"    # Raise de 2x le pot
    RAISE_3X_POT = "raise-300%"    # Raise de 3x le pot
    # -----------------------------------------------------
    ALL_IN = "all-in"

class GamePhase(Enum):
    """
    Énumération des phases d'une partie de poker, de la distribution au showdown.
    """
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"

class Player:
    """
    Représente un joueur de poker avec ses cartes, son stack et son état de jeu.
    """
    def __init__(self, name: str = "Player", stack: int = 100):
        """
        Initialise un joueur avec son agent associé, son stack et sa position.
        
        Args:
            agent (PokerAgent): L'agent qui contrôle ce joueur
            stack (int): Stack de départ en jetons
            position (int): Position à la table (0-5)
        """
        self.name = name
        self.stack = stack
        self.role_position = None # 0-2 (0 = SB, 1 = BB, 2 = BTN)
        self.cards: List[Card] = []
        self.is_active = True # True si le joueur a assez de fonds pour jouer (stack > big_blind)
        self.has_folded = False
        self.show_cards = True # True si on veut voir les cartes du joueur
        self.is_all_in = False
        self.range = None # Range du joueur (à initialiser comme l'ensemble des mains possibles)
        self.current_player_bet = 0 # Montant de la mise actuelle du joueur
        self.total_bet = 0  # Cumul des mises effectuées dans la main
        self.has_acted = False # True si le joueur a fait une action dans la phase courante (nécessaire pour savoir si le tour est terminé, car si le premier joueur de la phase check, tous les jouers sont a bet égal et ca déclencherait la phase suivante)
    
    def __str__(self):
        role_map = {0: "SB", 1: "BB", 2: "BTN"}
        role = role_map.get(self.role_position, "?")
        return (f"Player(name={self.name}, role={role}, stack={self.stack}, cards={self.cards}, "
                f"is_active={self.is_active}, has_folded={self.has_folded}, is_all_in={self.is_all_in}, "
                f"current_bet={self.current_player_bet}, total_bet={self.total_bet}, has_acted={self.has_acted})")

class SidePot:
    """
    Représente un pot additionnel qui peut être créé lors d'un all-in.

    Pour la répartition en side pots, on laisse les plus pauvres all-in dans le main pot et on attend la fin de la phase.
    Si les joeurs les plus pauvres son all-in et que les plus riches sont soit all-in aussi soit à un bet égal, la phase est terminée et on réparti les surplus en side pots.
    """
    def __init__(self, id: int):
        self.id = id # 0-1 (2 Pots max pour 3 joueurs, un main pot et 1 side pot)
        self.players = []
        self.contributions_dict = {} # Dictionnaire des contributions de chaque joueur dans le side pot
        self.sum_of_contributions = 0 # Montant total dans le side pot

class ActionValidator:
    def __init__(self, players: List[Player], big_blind: float):
        self.players = players
        self.big_blind = big_blind
        self.masks = {0: {a: False for a in PlayerAction}, 1: {a: False for a in PlayerAction}, 2: {a: False for a in PlayerAction}}

    def update_available_actions(self, player: Player, current_maximum_bet: float, number_raise_this_game_phase: int, main_pot: float, phase: GamePhase):
        player_role = player.role

        if player.is_all_in:
            for action in PlayerAction:
                self.masks[player_role][action] = False
            return []

        for action in PlayerAction:
            self.masks[player_role][action] = True
        
        if player.current_player_bet < current_maximum_bet:
            self.masks[player_role][PlayerAction.CHECK] = False
        
        if self.masks[player_role][PlayerAction.CHECK]:
            self.masks[player_role][PlayerAction.FOLD] = False
        
        if player.current_player_bet == current_maximum_bet:
            self.masks[player_role][PlayerAction.CALL] = False
        elif player.stack < (current_maximum_bet - player.current_player_bet):
            self.masks[player_role][PlayerAction.CALL] = False
            if player.stack > 0:
                self.masks[player_role][PlayerAction.ALL_IN] = True
            self.masks[player_role][PlayerAction.RAISE] = False
        
        if current_maximum_bet == 0:
            min_raise = self.big_blind
        else:
            min_raise = (current_maximum_bet - player.current_player_bet) * 2
        
        if player.stack < min_raise:
            self.masks[player_role][PlayerAction.RAISE] = False

        if number_raise_this_game_phase >= 4:
            self.masks[player_role][PlayerAction.RAISE] = False
        
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
        for action in pot_raise_actions:
            if number_raise_this_game_phase >= 4:
                self.masks[player_role][action] = False
            else:
                percentage = raise_percentages[action]
                required_increase = main_pot * percentage
                if player.stack < required_increase or required_increase < min_raise:
                    self.masks[player_role][action] = False
        
        self.masks[player_role][PlayerAction.ALL_IN] = player.stack > 0
        
        if phase == GamePhase.SHOWDOWN:
            for action in PlayerAction:
                self.masks[player_role][action] = False

        return [a for a, enabled in self.masks[player_role].items() if enabled]
