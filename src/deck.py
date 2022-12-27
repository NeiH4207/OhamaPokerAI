from src.card import Card
import random
import numpy as np

class StandardDeck(list):
    def __init__(self):
        super().__init__()
        self.n_cards = 13
        self.n_suits= 4
        suits = list(range(self.n_suits))
        ranks = list(range(2, 2 + self.n_cards))
        for j in suits:
            for i in ranks:
                self.append(Card(i, j))

    def __repr__(self):
        return f"Standard deck of cards {len(self)} cards remaining"

    def shuffle(self):
        random.shuffle(self)
        # print("\n\n--deck shuffled--")

    def deal(self, location, times=1):
        for i in range(times):
            location.add_card(self.pop(0))

    def burn(self):
        self.pop(0)
        
    def check_straight_flush(self, hand):
        return self.check_straight(hand) and self.check_flush(hand)
        
    def check_straight(self, hand):
        rank_list = []
        for card in hand:
            rank_list.append(card.rank)
        rank_list = sorted(rank_list)
        if len(list(set(rank_list))) == 5 \
            and rank_list[-1] - rank_list[0] + 1 == 5:
            return True # 2 3 4 5 6 -> T J Q K A
        elif len(list(set(rank_list))) == 5 \
            and rank_list[0] == 2 and rank_list[-2] == 5 and rank_list[-1] == 14:
            return True # 10 J Q K A
        return False
    
    def check_flush(self, hand):
        suit_list = []
        for card in hand:
            suit_list.append(card.suit)
        return len(list(set(suit_list))) == 1
    
    def check_four_of_a_kind(self, hand):
        rank_list = []
        for card in hand:
            rank_list.append(card.rank)
        unique_list, counter = np.unique(rank_list, return_counts=True)
        return len(unique_list) == 2 and (counter[0] == 1 or counter[1] == 1)
        
    def check_full_house(self, hand):
        rank_list = []
        for card in hand:
            rank_list.append(card.rank)
        unique_list, counter = np.unique(rank_list, return_counts=True)
        return len(unique_list) == 2 and (counter[0] == 2 or counter[1] == 2)
    
    def check_three_of_a_kind(self, hand):
        rank_list = []
        for card in hand:
            rank_list.append(card.rank)
        unique_list, counter = np.unique(rank_list, return_counts=True)
        return len(unique_list) == 3 and \
            (counter[0] == 3 or counter[1] == 3 or counter[2] == 3)
    
    def check_two_pair(self, hand):
        rank_list = []
        for card in hand:
            rank_list.append(card.rank)
        unique_list, counter = np.unique(rank_list, return_counts=True)
        counter = sorted(counter)
        return len(unique_list) == 3 and \
            (counter[-1] == 2 or counter[-2] == 2)
            
    def check_pair(self, hand):
        rank_list = []
        for card in hand:
            rank_list.append(card.rank)
        unique_list = np.unique(rank_list)
        return len(unique_list) == 4
        
    def get_hand_score(self, hand):
        score = 0
        suit_list = []
        rank_list = []
        for card in hand:
            suit, rank = card.suit, card.rank
            if rank == 0:
                rank = 13
            suit_list.append(suit)
            rank_list.append(rank)
        sorted_rank_list = sorted((np.array(rank_list) + self.n_cards - 1) % self.n_cards)
        unique_list, counter = np.unique(sorted_rank_list, return_counts=True)
        if len(unique_list) < 5:
            for _ in range(5 - len(unique_list)):
                unique_list = np.append(unique_list, 0)
                counter = np.append(counter, 0)
        strongest_cards = unique_list[np.argsort(counter)[::-1]]
        for card in strongest_cards:
            score = score * self.n_cards + ((card - 1 + self.n_cards) % self.n_cards)
        if self.check_straight_flush(hand):
            return (8, score)
        elif self.check_four_of_a_kind(hand):
            return (7, score)
        elif self.check_full_house(hand):
            return (6, score)
        elif self.check_flush(hand):
            return (5, score)
        elif self.check_straight(hand):
            return (4, score)
        elif self.check_three_of_a_kind(hand):
            return (3, score)
        elif self.check_two_pair(hand):
            return (2, score)
        elif self.check_pair(hand):
            return (1, score)
        else:
            return (0, score)
    
    def compare_hands(self, hand1, hand2):
        score1 = self.get_hand_score(hand1)
        score2 = self.get_hand_score(hand2)
        if score1 > score2:
            return 1
        elif score1 < score2:
            return -1
        else:
            return 0
        
    def get_best_hand(self, hand_list):
        best_hand = hand_list[0]
        for i in range(1, len(hand_list)):
            if self.compare_hands(best_hand, hand_list[i]) == -1:
                best_hand = hand_list[i]
        
        return best_hand
    
    def get_hand_name(self, hand):
        hand_names = ["High Card", "Pair", "Two Pair", "Three of a Kind",
                        "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush"]
        return '\033[92m' + hand_names[self.get_hand_score(hand)[0]] + '\033[0m'
    
    def get_best_hand_score(self, hand_list):
        best_hand = hand_list[0]
        for i in range(1, len(hand_list)):
            if self.compare_hands(best_hand, hand_list[i]) == -1:
                best_hand = hand_list[i]
        return self.get_hand_score(best_hand)
    