import numpy as np

HEARTS   = '\033[91m' + chr(9829) + '\033[0m'
SPADES   = chr(9824)
DIAMONDS = '\033[91m' + chr(9830) + '\033[0m'
CLUBS    = chr(9827)

class Card(object):
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
        self.showing = True
        self.backside = False
    
    def to_num(self):
        return (self.rank - 2) * 4 + self.suit

    def to_vec(self):
        vec = np.zeros((4, 52), dtype=np.uint8)
        vec[self.suit, self.rank - 2] = 1
        return vec
    
    def __str__(self):
        suit, rank = self.suit, self.rank
        if rank == 1 or rank == 14:
            rank = 'A'
        elif rank == 10:
            rank = 'T'
        elif rank == 11:
            rank = 'J'
        elif rank == 12:
            rank = 'Q'
        elif rank == 13:
            rank = 'K'
        else:
            rank = str(rank)
        if suit == 0:
            suit = HEARTS
        elif suit == 1:
            suit = SPADES
        elif suit == 2:
            suit = DIAMONDS
        elif suit == 3:
            suit = CLUBS
        return rank + suit