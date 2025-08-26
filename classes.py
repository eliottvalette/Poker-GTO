# classes.py
# ------------------------------------------------------------
# Classes simplifiées pour la gestion des cartes et du deck
# ------------------------------------------------------------

from typing import Tuple, List
from typing import Dict

class Card:
    __slots__ = ("rank", "suit", "id")

    def __init__(self, rank: int, suit: int):
        self.rank = rank                  # 2..14
        self.suit = suit                  # 0..3
        self.id   = (rank - 2) * 4 + suit # 0..51

    def __int__(self) -> int:
        return self.id

    def __index__(self) -> int:  # permet usage dans arrays si besoin
        return self.id

    def __str__(self):
        ranks = {14:'A',13:'K',12:'Q',11:'J',10:'T',9:'9',8:'8',7:'7',6:'6',5:'5',4:'4',3:'3',2:'2'}
        suits = {0:'♠',1:'♥',2:'♦',3:'♣'}
        return f"{ranks[self.rank]}{suits[self.suit]}"

    def __repr__(self):
        return f"Card(rank={self.rank}, suit={self.suit}, id={self.id})"


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