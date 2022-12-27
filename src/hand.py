# color + symbol
HEARTS   = '\033[91m' + chr(9829) + '\033[0m'
SPADES   = '\033[0m' + chr(9824) + '\033[0m'
DIAMONDS = '\033[91m' + chr(9830) + '\033[0m'
CLUBS    = '\033[0m' + chr(9827) + '\033[0m'

class Hand(object):
    """A hand of playing cards.

    Attributes:
        cards: list of Card objects.
    """
    def __init__(self):
        self.cards = list()
        self.backside = True
        
    def add_card(self, card):
        """Adds a card to the hand."""
        self.cards.append(card)
    
    def remove_card(self, card):
        """Removes a card from the hand."""
        self.cards.remove(card)
        self.sort()
    
    def clear(self):
        """Removes all cards from the hand."""
        self.cards.clear()
    
    def sort(self):
        """Sorts the cards in the hand."""
        self.cards.sort(key=lambda card: card.rank, reverse=True)
    
    def __str__(self):
        """Returns a string representation of a hand."""
        rows = ['' for _ in range(4)]
        rows[0] = ' ___  ' * len(self.cards)
        for i, card in enumerate(self.cards):
            rank = str(card.rank)
            if rank == '10':
                rank = 'T'
            elif rank == '11':
                rank = 'J'
            elif rank == '12':
                rank = 'Q'
            elif rank == '13':
                rank = 'K'
            elif rank == '1' or rank == '14':
                rank = 'A'
            if card.suit == 0:
                suit = HEARTS
            elif card.suit == 1:
                suit = SPADES
            elif card.suit == 2:
                suit = DIAMONDS
            elif card.suit == 3:
                suit = CLUBS
            if self.backside:
                # Print the card's back:
                rows[1] += '|** | '
                rows[2] += '|***| '
                rows[3] += '|_**| '
            else:
                rows[1] += '|{} | '.format(rank.ljust(2))
                rows[2] += '| {} | '.format(suit)
                rows[3] += '|_{}| '.format(rank.rjust(2, '_'))
                
        # print each row on the screen:
        for row in rows:
            print(row)
        return ''
        
    def __repr__(self):
        """Returns a string representation of a hand."""
        return self.cards
    
    def get_card_list(self):
        """Returns a list of the cards in the hand."""
        return self.cards
    
    def basic_str(self):
        """Displays the cards in the hand."""
        str_hand = ""
        for card in sorted(self.cards, key=lambda card: card.rank, reverse=True):
            rank = str(card.rank)
            if rank == '10':
                rank = 'T'
            elif rank == '11':
                rank = 'J'
            elif rank == '12':
                rank = 'Q'
            elif rank == '13':
                rank = 'K'
            elif rank == '1' or rank == '14':
                rank = 'A'
            if card.suit == 0:
                suit = HEARTS
            elif card.suit == 1:
                suit = SPADES
            elif card.suit == 2:
                suit = DIAMONDS
            elif card.suit == 3:
                suit = CLUBS
            str_hand += rank + suit + ' '
        return '\033[0m<< ' + str_hand + '>>\033[0m'