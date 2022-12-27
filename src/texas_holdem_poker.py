from src.player import Player
from src.deck import StandardDeck
import time
import itertools
import numpy as np

class Game(object):
    def __init__(self, n_playes=3, big_blind=10):
        self.n_playes = n_playes
        self.big_blind = big_blind
        self.small_blind = self.big_blind // 2
        self.buyin = self.big_blind * 100
        self.small_blind_index = 0
        self.cards = []
        self.players = [Player(id=i, chips=self.buyin) for i in range(self.n_playes)]
        self.deck = StandardDeck()
        self.action_list = []
        self.action_round_list = []
        self.total_pot = 0
        
    def get_raw_state(self, player_id=0):
        state = {}
        state['player-id'] = player_id
        state['big-blind'] = self.big_blind
        state['common-cards'] = self.cards
        state['hole-cards'] = self.players[player_id].cards
        state['player-chips'] = [player.chips for player in self.players]
        state['all-action'] = self.action_list
        state['total-pot'] = self.total_pot
        return state
    
    def action_embedding(self, act):
        player_id = act[0]
        action = act[1]
        value = act[2]
        player_vec = np.zeros(self.n_playes, dtype=np.float32)
        player_vec[player_id] = 1
        action_vec = np.zeros(5, dtype=np.float32)
        if action == 'check':
            action_vec[0] = 1
        elif action == 'bet':
            action_vec[1] = 1
        elif action == 'call':
            action_vec[2] = 1
        elif action == 'raise':
            action_vec[3] = 1
        elif action == 'all-in':
            action_vec[4] = 1
        feature = np.concatenate([player_vec, action_vec, [value]])
        return feature
        
    def get_embed_state(self, player_id=0):
        state = self.get_raw_state(player_id)
        _state = {}
        _state['timeseries-feature'] = [self.action_embedding(act) for act in state['all-action']]
        _state['static-feature'] = np.concatenate([[state['big-blind']],
                                                  state['player-chips'],
                                                  state['total-pot'],
                                                  np.flatten([card.to_vec() for card in state['common-cards']]),
                                                  np.flatten([card.to_vec() for card in state['hole-cards']])
                                                  ])
        return _state
        
    def print_round_info(self):
        print("\n")
        for player in self.list_of_players:
            print("\n")
            print(f"Name: {player.name}")
            print(f"Cards: {player.cards}")
            print(f"Player score: {player.score}")
            print(f"Chips: {player.chips}")
            print(f"Special Attributes: {player.list_of_special_attributes}")
            if player.fold:
                print(f"Folded")
            if player.all_in:
                print(f"All-in")
            print(f"Stake: {player.stake}")
            print(f"Stake-gap: {player.stake_gap}")
            print("\n")
        print(f"Pot: {self.pot}")
        print(f"Community cards: {self.cards}")
        print("\n")

    def deal_hole(self):
        for player in self.players:
            if player.active:
                self.deck.deal(player, 2)

    def deal_flop(self):
        self.deck.burn()
        self.deck.deal(self, 3)

    def deal_turn(self):
        self.deck.burn()
        print("\n--card burned--")
        self.deck.deal(self, 1)
        print(f"\nCommunity Cards: {self.cards}")

    def deal_river(self):
        self.deck.burn()
        print("\n--card burned--")
        self.deck.deal(self, 1)
        print(f"\n\nCommunity Cards: {self.cards}")
        
    def hand_scorer(self, player):
        seven_cards = player.cards + self.cards
        all_hand_combos = list(itertools.combinations(seven_cards, 5))
        list_of_all_score_possibilities = []
        for hand in all_hand_combos:
            score = self.deck.get_hand_score(hand)
            list_of_all_score_possibilities.append(score)
        player.score = max(list_of_all_score_possibilities)

    def find_winners(self):
        scores = []
        for player in self.players:
            if not player.fold:
                scores.append(player.score)
            else:
                scores.append(0)
                
        max_score = max(scores)
        winners = [player for player in self.players if player.score == max_score]
        winner_pot_chips = [player.chip_in_pots for player in winners]
        max_winner_pot_chips = max(winner_pot_chips)
        
        losers  = [player for player in self.players if player.score < max_score]
        loser_pot_chips = sorted([player.chip_in_pots for player in losers])
        for player in self.players:
            player.chips += max(0, player.chip_in_pots - max_winner_pot_chips)
        sharing_pot_winner_idx = 0
        while(sharing_pot_winner_idx < len(winners)):
            while(winner_pot_chips[sharing_pot_winner_idx].chip_in_pots == \
                winner_pot_chips[sharing_pot_winner_idx + 1].chip_in_pots):
                sharing_pot_winner_idx += 1
            sharing_chips = 0
            for i in range(len(loser_pot_chips)):
                collect_chips =  loser_pot_chips[i] - winner_pot_chips[sharing_pot_winner_idx]
                sharing_chips += collect_chips
                loser_pot_chips[i] -= collect_chips
            sharing_chips /= sharing_pot_winner_idx
            for i in range(sharing_pot_winner_idx):
                winners[i].chips += sharing_chips
            sharing_pot_winner_idx += 1
        

    def clear_board(self):
        self.possible_responses.clear()
        self.cards.clear()
        self.deck = StandardDeck()
        self.deck.shuffle()
        self.pot = 0
        self.pot_dict.clear()
        self.winners.clear()
        self.list_of_scores_from_eligible_winners.clear()
        self.action_counter = 0
        self.highest_stake = 0
        self.fold_list.clear()
        self.not_fold_list.clear()
        self.fold_out = False
        self.list_of_scores_eligible.clear()
        self.round_ended = False
        for player in self.list_of_players:
            player.score.clear()
            player.cards.clear()
            player.stake = 0
            player.stake_gap = 0
            player.ready = False
            player.all_in = False
            player.fold = False
            player.list_of_special_attributes.clear()
            player.win = False
            
    def show_hold_cards(self):
        for player in self.players:
            print("Player {} hole: {}, {}".format(player.id, player.cards[0], player.cards[1]))
    
    def play(self):
        orders = np.arange(self.n_playes) + self.small_blind_index
        value = 0
        for idx in orders:
            print("Action: check | bet | call | raise")
            inputs = input().split()
            action = inputs[0]
            if len(inputs) > 1:
                value = int(inputs[1])
            if action == 'check':
                continue
            elif action == 'bet':
                self.players[idx].bet(value)
            elif action == 'call':
                self.players[idx].bet(value)
    
    def _action_table(self):
        orders = np.array([x for x in range(self.small_blind_index, self.n_players+1)]
                           + [x for x in range(0, self.small_blind_index)])
        while True:
            for player_id in orders:
                action = self.players[player_id].get_action(self.get_embed_state())
     
    def run_game(self):
        self.deck.shuffle()
        self.deal_hole()
        self.show_hold_cards()
        self._action_table()
        # Preflop Round
        