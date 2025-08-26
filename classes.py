# classes.py
# ------------------------------------------------------------
# Classes simplifiées pour la gestion des cartes et du deck
# ------------------------------------------------------------

from typing import Tuple, List
from typing import Dict

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
        self.role = None # 0-2 (0 = SB, 1 = BB, 2 = BTN)
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
        return (f"Player(name={self.name}, role={self.role}, stack={self.stack}, cards={self.cards}, "
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