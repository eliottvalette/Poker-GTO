# classes.py
# ------------------------------------------------------------
# Classes pour la gestion des cartes et du deck
# ------------------------------------------------------------
from __future__ import annotations
from typing import List, Tuple, Set, Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Card:
    """Représente une carte avec son rang et sa couleur"""
    rank: int  # 2..14 (A=14)
    suit: int  # 0..3
    
    def __post_init__(self):
        if not (2 <= self.rank <= 14):
            raise ValueError(f"Rang invalide: {self.rank}")
        if not (0 <= self.suit <= 3):
            raise ValueError(f"Couleur invalide: {self.suit}")
    
    @classmethod
    def from_int(cls, card_int: int) -> Card:
        """Crée une carte à partir de sa représentation entière (0..51)"""
        rank = (card_int // 4) + 2
        suit = card_int % 4
        return cls(rank, suit)
    
    def to_int(self) -> int:
        """Convertit la carte en sa représentation entière (0..51)"""
        return (self.rank - 2) * 4 + self.suit
    
    def __str__(self) -> str:
        ranks = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T', 
                9: '9', 8: '8', 7: '7', 6: '6', 5: '5', 
                4: '4', 3: '3', 2: '2'}
        suits = {0: '♠', 1: '♥', 2: '♦', 3: '♣'}
        return f"{ranks[self.rank]}{suits[self.suit]}"
    
    def __repr__(self) -> str:
        return f"Card({self.rank}, {self.suit})"


class Deck:
    """Représente un deck de 52 cartes"""
    
    RANKS = list(range(2, 15))  # 2..14 (A=14)
    SUITS = list(range(4))      # 0..3
    
    def __init__(self):
        self.cards = [Card(r, s) for r in self.RANKS for s in self.SUITS]
    
    def __len__(self) -> int:
        return len(self.cards)
    
    def __getitem__(self, index: int) -> Card:
        return self.cards[index]
    
    def get_card(self, rank: int, suit: int) -> Card:
        """Retourne la carte avec le rang et la couleur spécifiés"""
        return Card(rank, suit)
    
    def get_card_by_int(self, card_int: int) -> Card:
        """Retourne la carte à partir de sa représentation entière"""
        return Card.from_int(card_int)
    
    def all_starting_combos(self) -> List[Tuple[Card, Card]]:
        """Retourne tous les combos de départ possibles (1326 combos)"""
        combos = []
        for i in range(52):
            for j in range(i+1, 52):
                combos.append((self.cards[i], self.cards[j]))
        return combos
    
    def filter_combos_excluding(self, combos: Iterable[Tuple[Card, Card]], 
                               blocked: Set[int]) -> List[Tuple[Card, Card]]:
        """Filtre les combos en excluant les cartes bloquées"""
        out = []
        for a, b in combos:
            if a.to_int() in blocked or b.to_int() in blocked:
                continue
            out.append((a, b))
        return out
    
    def combo_to_169(self, c1: Card, c2: Card) -> str:
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


# Instances globales pour compatibilité avec le code existant
DECK = [Card(r, s) for r in Card.RANKS for s in Card.SUITS]
ALL_COMBOS = [(c1, c2) for i, c1 in enumerate(DECK) for c2 in DECK[i+1:]]

# Fonctions utilitaires pour compatibilité
def card(rank: int, suit: int) -> int:
    """Compatibilité avec l'ancien code - retourne l'entier de la carte"""
    return Card(rank, suit).to_int()

def card_rank(c: int) -> int:
    """Compatibilité avec l'ancien code - retourne le rang de la carte"""
    return Card.from_int(c).rank

def card_suit(c: int) -> int:
    """Compatibilité avec l'ancien code - retourne la couleur de la carte"""
    return Card.from_int(c).suit

def all_starting_combos() -> List[Tuple[int, int]]:
    """Compatibilité avec l'ancien code - retourne les combos en entiers"""
    return [(c1.to_int(), c2.to_int()) for c1, c2 in ALL_COMBOS]

def combo_to_169(c1: int, c2: int) -> str:
    """Compatibilité avec l'ancien code - convertit un combo en notation 169"""
    card1 = Card.from_int(c1)
    card2 = Card.from_int(c2)
    deck = Deck()
    return deck.combo_to_169(card1, card2)

def filter_combos_excluding(combos: Iterable[Tuple[int, int]], blocked: Set[int]) -> List[Tuple[int, int]]:
    """Compatibilité avec l'ancien code - filtre les combos"""
    deck = Deck()
    card_combos = [(Card.from_int(c1), Card.from_int(c2)) for c1, c2 in combos]
    filtered = deck.filter_combos_excluding(card_combos, blocked)
    return [(c1.to_int(), c2.to_int()) for c1, c2 in filtered]
