import torch
from torch.autograd import Variable
from src.hand import Hand
from src.algorithms import Random


class Player(object):
    def __init__(self, id=None, name=None, chips=0, nnet=None):
        self.id = id
        self.name = name
        self.active = True
        self.chips = chips
        self.action = 0
        self.bet_value = 0
        self.raise_value = 0
        self.stake = 0
        self.stake_gap = 0
        self.hand = Hand()
        self.best_hand = Hand()
        self.score = []
        self.fold = False
        self.ready = False
        self.all_in = False
        self.list_of_special_attributes = []
        self.is_winner = False
        self.chip_in_pots = 0
        self.algorithm = Random()
        self.bankroll = 0
        self.last_win_chips = 0
        self.last_state = None
        self.last_action_log_prob = None
        self.last_value_pred = None
        self.nnet = nnet
        self.chip_history = []
        
    def set_bankroll(self, bankroll):
        self.bankroll = bankroll

    def __repr__(self):
        name = self.name
        return name
    
    def buy_in(self, buy_in_chip, show=True):
        if self.chips > buy_in_chip:
            if show:
                print("{} + {} chips to bankroll, buy-in chip | bankroll: {} | {}".format(
                    self.name, buy_in_chip, self.chips - buy_in_chip, self.bankroll + self.chips - buy_in_chip))
            self.bankroll += self.chips - buy_in_chip
            self.chips = buy_in_chip
        else:
            added_chip = min(buy_in_chip - self.chips, self.bankroll)
            self.bankroll -= added_chip
            self.chips += added_chip
            if show:
                print("{} buy {} chips from bankroll, buy-in chip | bankroll: {} | {}".format(
                    self.name, added_chip, self.chips, self.bankroll))
        
    def _call(self, x):
        self.chip_in_pots += x
        self.chips -= x
        
    def _check(self):
        pass
    
    def _bet(self, x):
        self.chip_in_pots += x
        self.chips -= x
        
    def _raise(self, x):
        self.chip_in_pots += x
        self.chips -= x
        
    def _all_in(self):
        self.chip_in_pots += self.chips
        self.chips = 0
        
    def _fold(self):
        self.fold = True
        
    def add_card(self, card):
        self.hand.add_card(card)
        
    def get_ohama_action(self, state):
        action = self.algorithm.get_action(state)
        return action

    def make_best_hand(self, cards):
        self.best_hand.clear()
        for card in cards:
            self.best_hand.add_card(card)
            
    def get_card_list(self):
        return self.hand.get_card_list()
    
    def get_action(self, state): # 0: fold, 1: all-in
        self.last_state = state
        action, action_log_prob, state_value_pred = self.nnet.select_action(state)
        self.last_action_log_prob = action_log_prob
        self.last_value_pred = state_value_pred
        return action
    
    def select_action(self, state):
        action = self.get_action(state)
        self.last_action = action
        return action