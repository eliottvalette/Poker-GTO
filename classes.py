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
