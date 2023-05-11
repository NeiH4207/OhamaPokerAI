import os
import random
from src.hand import Hand as CommonCard
from src.player import Player
from src.deck import StandardDeck
from src.utils import *
import time
import itertools
import numpy as np
from models.PokerNet import PokerNet
from collections import deque
from matplotlib import pyplot as plt
import seaborn as sns

class Game(object):
    def __init__(self, n_players=4, big_blind=10, 
                 playerNames=list(), n_games=1000001,
                 model_path=None):
        self.n_players = n_players
        self.big_blind = big_blind
        self.small_blind = self.big_blind // 2
        self.small_blind_index = random.randint(0, self.n_players - 1)
        self.buyin_amount = 10 * self.big_blind
        self.cards = []
        self.input_dim = 52 * 4 * 2 + 16
        self.model = PokerNet(self.input_dim, 256, 2)
        self.playerNames = playerNames
        self.players = [Player(id=i, chips=0, name=playerNames[i],
                               nnet=self.model) for i in range(self.n_players)]
        self.dealer = Player(id=self.n_players, name='Dealer')
        self.dealer.hand = CommonCard()
        self.dealer.hand.backside = False
        self.deck = StandardDeck()
        self.action_list = []
        self.total_pot = 0
        self.num_cards_in_hold = 4
        self.n_games = n_games
        self.model_path = model_path
        for i in range(self.n_players):
            # random from 100 to 1000 big blind
            self.players[i].set_bankroll(random.randint(300, 300) * self.big_blind)
    
    def load_model(self, model_path):
        self.model.load(model_path)
    
    def reset(self):
        self.cards = []
        self.players = [Player(id=i, chips=0, name=self.playerNames[i],
                               nnet=self.model) for i in range(self.n_players)]
        self.dealer = Player(id=self.n_players, name='Dealer')
        self.dealer.hand = CommonCard()
        self.dealer.hand.backside = False
        self.deck = StandardDeck()
        self.action_list = []
        self.total_pot = 0
        self.num_cards_in_hold = 4
        for i in range(self.n_players):
            # random from 100 to 1000 big blind
            self.players[i].set_bankroll(random.randint(20, 40) * self.big_blind)
    
    def get_num_active_players(self):
        return len([player for player in self.players if player.active])
    
    def get_raw_state(self, player_id=0):
        state = {}
        state['player-id'] = player_id
        state['big-blind'] = self.big_blind
        state['common-cards'] = self.dealer.hand
        state['hole-cards'] = self.players[player_id].hand
        state['player-chip'] = self.players[player_id].chips
        state['all-action'] = self.action_list
        return state
    
    def action_embedding(self, act):
        player_id = act[0]
        action = act[1]
        value = act[2]
        player_vec = np.zeros(self.n_players, dtype=np.float32)
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
        
    def get_embeding_state(self, player_id=0):
        actions = []
        orders = (np.arange(self.n_players) + self.small_blind_index + 2) % self.n_players
        player_order = None
        is_actioned = True
        for i, idx in enumerate(orders):
            if idx == player_id:
                is_actioned = False
                player_order = i
            if self.players[idx].active:
                if self.players[idx].fold:
                    actions.extend([1, 0, is_actioned])
                else:
                    actions.extend([1, 1, is_actioned])
            else:
                actions.extend([0, 0, False])
        
        # make onehot
        action_state = np.array(actions, dtype=np.float32)
        player_order = np.eye(self.n_players)[player_order]
        action_state = np.concatenate([action_state, player_order])
        all_combo = []
        for pair in itertools.combinations(self.players[player_id].hand.get_card_list(), 2):
            combo = self.dealer.hand.get_card_list() + list(pair)
            all_combo.append(combo)
        embeding_cards = []
        embed_vec_hole_cards = np.zeros((4, 52), dtype=np.float32)
        for card in self.players[player_id].hand.get_card_list():
            embed_vec_hole_cards += card.to_vec()
        embeding_cards.append(embed_vec_hole_cards)
        
        embed_vec_common_cards = np.zeros((4, 52), dtype=np.float32)
        for card in self.dealer.hand.get_card_list():
            embed_vec_common_cards += card.to_vec()
        embeding_cards.append(embed_vec_common_cards)
        
        embeding_cards = np.array(embeding_cards, dtype=np.float32)
        state = [embeding_cards, action_state]
        return state

    def deal_hole(self, n_cards=4):
        for player in self.players:
            if player.active:
                self.deck.deal(player, n_cards)
                player.hand.sort()

    def deal_flop(self):
        self.deck.burn()
        self.deck.deal(self.dealer, 3)

    def deal_turn(self):
        self.deck.burn()
        # print("\n--card burned--")
        self.deck.deal(self.dealer, 1)
        # print(f"\nCommunity Cards: {self.cards}")

    def deal_river(self):
        self.deck.burn()
        # print("\n--card burned--")
        self.deck.deal(self.dealer, 1)
        # print(f"\n\nCommunity Cards: {self.cards}")
        
    def get_best_hand(self, player):
        all_hand_combos = list()
        all_pairs = list(itertools.combinations(player.hand.get_card_list(), 2))
        all_three_cards = list(itertools.combinations(self.dealer.get_card_list(), 3))
        for pair in all_pairs:
            pair = [card for card in pair]
            for three_cards in all_three_cards:
                three_cards = [card for card in three_cards]
                all_hand_combos.append(pair + three_cards)
        # for card in player.hand.get_card_list():
        #     all_hand_combos.extend(itertools.combinations([card] + self.dealer.get_card_list(), 5))
        best_hand = self.deck.get_best_hand(all_hand_combos)
        return best_hand

    def clear_board(self):
        # os.system('cls' if os.name == 'nt' else 'clear')
        self.deck = StandardDeck()
        self.deck.shuffle()
        self.dealer.hand.clear()
        self.action_list = []
        for player in self.players:
            player.stake = 0
            player.fold = False
            player.all_in = False
            player.is_winner = False
            player.best_hand.clear()
            player.hand.clear()

        
    def get_winners(self):
        winners = []    
        scores = []
        idx = []
        # Sort players by score
        for i, player in enumerate(self.players):
            if player.active:
                if player.fold:
                    score = (0, 0)
                else:
                    best_hand = self.get_best_hand(player)
                    player.make_best_hand(best_hand)
                    score = self.deck.get_hand_score(player.best_hand.get_card_list())
                scores.append(score)
                idx.append(i)
        
        # Sort tupple score = (score1, score2)
        sorted_scores = sorted(list(set(scores)), reverse=True)
        for i in range(len(sorted_scores)):
            winner_same_score = []
            score = sorted_scores[i]
            for j in range(len(scores)):
                if scores[j] == score:
                    winner_same_score.append(self.players[idx[j]])
            winners.append(winner_same_score)
        return winners
    
    def divide_pot(self, winners):
        
        for player in self.players:
            player.last_win_chips = 0
            
        if len(winners) == 1:
            for player in winners[0]:
                player.chips += player.stake
                player.stake = 0
                player.last_win_chips = 0
            self.total_pot = 0
            return
        
            
        for i, same_score_winners in enumerate(winners):
            for k, winner in enumerate(same_score_winners):
                clammied_chips = 0
                for j, losers in enumerate(winners[i+1:]):
                    for loser in losers:
                        added_chips = min(loser.stake, winner.stake) / (len(same_score_winners) - k)
                        clammied_chips += added_chips
                        loser.stake -= added_chips
                        loser.last_win_chips -= added_chips
                        winner.chips += added_chips
                        winner.last_win_chips += added_chips
                
                winner.chips += winner.stake
                winner.stake = 0
                
    def show_hold_cards(self):
        for player in self.players:
            if player.active and not player.fold:
                best_hand = self.get_best_hand(player)
                player.make_best_hand(best_hand)
                hand_name = self.deck.get_hand_name(best_hand)
                print('Player {} hole {} as a {}'.format(player.name, 
                            player.best_hand.basic_str(), hand_name))
                print(player.hand)
        print('-'*20)
        
    def show_infor_deck(self):
        # show community cards
        print("-----------------------------------")
        print(self.dealer.hand)
        print("-----------------------------------")
        # print total pot
        print("Total pot: {}".format(self.total_pot))
        # print small / big blind
        print("Blind: {} / {}".format(self.small_blind, self.big_blind))
        
    def play(self):
        self.model.eval()
        self.data = []
        ranking_orders = []
        while True:
            self.clear_board()
            for player in self.players:
                if player.active:
                    player.buy_in(self.buyin_amount)
                    if player.chips == 0:
                        player.active = False
                        ranking_orders.append(player.name)
                        print(bcolors.WARNING + "Player not enough chips to play" + bcolors.ENDC)
                player.chip_history.append(player.bankroll)
                # Visualize chip histories
            fig, ax = plt.subplots()
            for player in self.players:
                ax.plot(player.chip_history, label=player.name)
            ax.legend()
            ax.set_xlabel('Hands played')
            ax.set_ylabel('Chips')
            plt.savefig('chip_history.png')
                  
            if self.get_num_active_players() < 2:
                print("Not enough players to play")
                active_players = [player for player in self.players if player.active]
                ranking_orders.extend([player.name for player in active_players])
                return ranking_orders
            
            self.deal_hole(n_cards=self.num_cards_in_hold)
            self.deal_flop()
            # self.show_hold_cards()
            orders = (np.arange(self.n_players) + self.small_blind_index + 2) % self.n_players
            # drop inactive players 
            orders = [idx for idx in orders if self.players[idx].active]
            self.players[orders[-2]].stake = min(self.small_blind, self.players[orders[-2]].chips)
            self.players[orders[-2]].chips -= self.players[orders[-2]].stake
            self.players[orders[-1]].stake = min(self.big_blind, self.players[orders[-1]].chips)
            self.players[orders[-1]].chips -= self.players[orders[-1]].stake
            self.total_pot = self.small_blind + self.big_blind
            fold_count = 0
            all_of_fold = False
            
            for idx in orders:
                if fold_count == len(orders) - 1:
                    all_of_fold = True
                    break
                self.players[idx].hand.backside = False
                # clear console
                # os.system('cls' if os.name == 'nt' else 'clear')
                if idx == -1:
                    self.show_infor_deck()
                    # print chips on stake and chips left
                    print("{} / {} chips on stake, rest {} chips left in bankroll".format(
                        self.players[idx].stake, self.players[idx].chips + self.players[idx].stake, 
                        self.players[idx].bankroll))
                    # show hand
                    best_hand = self.get_best_hand(self.players[idx])
                    self.players[idx].make_best_hand(best_hand)
                    hand_name = self.deck.get_hand_name(best_hand)
                    print('You {} have {} as a \033[92m {} \033[0m'.format(self.players[idx].name, 
                                                                           self.players[idx].best_hand.basic_str(),
                                                                           hand_name))
                    print(self.players[idx].hand)
                    action = self.players[idx].select_action(self.get_embeding_state(idx))
                    print("Action recommend: {}".format(['fold', 'all_in'][action]))
                    print(bcolors.WARNING + "Do you want to all-in, {}? (y/n)".format(self.players[idx].name) + bcolors.ENDC)
                    inputs = input()
                    if inputs == 'y':
                        action = 'all_in'
                    else:
                        action = 'fold'
                else:
                    action = self.players[idx].select_action(self.get_embeding_state(idx))
                    action = ['fold', 'all_in'][action]
                    # print(bcolors.WARNING + "Player {} choose {}".format(self.players[idx].name, action) + bcolors.ENDC)
                if action == 'all_in':
                    self.players[idx].all_in = True
                    self.total_pot += self.players[idx].chips
                    self.players[idx].stake += self.players[idx].chips
                    self.players[idx].chips = 0
                else:
                    self.players[idx].all_in = False
                    self.players[idx].fold = True   
                    fold_count += 1
            
            if all_of_fold:
                winners = self.get_winners()
                print("All of players fold, player {} win!".format(winners[0][0].name))
                self.divide_pot(winners)
                self.small_blind_index = (self.small_blind_index + 1) % self.n_players
                # input("")
                continue
                
            self.deal_turn()
            self.deal_river()   
            # os.system('cls' if os.name == 'nt' else 'clear')  
            # self.show_hold_cards() 
            # print(self.dealer.hand)
            winners = self.get_winners()
            self.divide_pot(winners)
            for winner in winners[0]:
                print(bcolors.OKGREEN + "Player {} wins {} chips with {}".format(
                    winner.name, winner.last_win_chips, winner.best_hand.basic_str()) + bcolors.ENDC \
                        + " as a\033[0m {}".format(self.deck.get_hand_name(winner.best_hand.get_card_list())))
            self.small_blind_index = (self.small_blind_index + 1) % self.n_players
            # input("")
            
            
    
    def simulate(self):
        data = {
            'state': deque(maxlen=32768),
            'action_log_prob': deque(maxlen=32768),
            'reward': deque(maxlen=32768),
            'value_pred': deque(maxlen=32768),
        }
        fold_counter = 0
        all_in_counter = 0
        
        for idx, game in enumerate(range(self.n_games)):
            # Train model each 10 games
            if idx > 0 and (idx) % 100 == 0 and len(data['reward']) % 32 > 1:
                v = self.model._training(data)
        
                print("Game: {}, Expected value: {} | All-in rate: {}".\
                    format(idx, v, np.round(all_in_counter / (fold_counter + all_in_counter + 1e-8), 3)))
                fold_counter = 0
                all_in_counter = 0
                    
            if (idx) % 100 == 0:
                path = os.path.join(self.model_path, 'model_{}.pt'.format(game))
                self.model.save(path)
                    
            self.clear_board()
            
            if self.get_num_active_players() < 2:
                self.reset()
                
            for player in self.players:
                if player.active:
                    player.buy_in(self.buyin_amount, show=False)
                    if player.chips == 0:
                        player.active = False
                        
            if self.get_num_active_players() < 2:
                self.reset()
                for player in self.players:
                    player.buy_in(self.buyin_amount, show=False)
                
            self.deal_hole(n_cards=self.num_cards_in_hold)
            self.deal_flop()
            orders = (np.arange(self.n_players) + self.small_blind_index + 2) % self.n_players
            # drop inactive players 
            orders = [idx for idx in orders if self.players[idx].active]
            self.players[orders[-2]].stake = min(self.small_blind, self.players[orders[-2]].chips)
            self.players[orders[-2]].chips -= self.players[orders[-2]].stake
            self.players[orders[-1]].stake = min(self.big_blind, self.players[orders[-1]].chips)
            self.players[orders[-1]].chips -= self.players[orders[-1]].stake
            self.total_pot = self.small_blind + self.big_blind
            fold_count = 0
            all_of_fold = False
            
            for idx in orders:
                if fold_count == len(orders) - 1:
                    all_of_fold = True
                    break
                action = self.players[idx].get_action(self.get_embeding_state(idx))
                if action == 1:
                    self.players[idx].all_in = True
                    self.players[idx].fold = False
                    self.total_pot += self.players[idx].chips
                    self.players[idx].stake += self.players[idx].chips
                    self.players[idx].chips = 0
                    all_in_counter += 1
                else:
                    self.players[idx].all_in = False
                    self.players[idx].fold = True  
                    fold_count += 1
                    fold_counter += 1
            
            if all_of_fold:
                winners = self.get_winners()
                for winer in winners[0]:
                    winer.is_winner = True
                self.divide_pot(winners)
                self.small_blind_index = (self.small_blind_index + 1) % self.n_players
                
                for player in self.players:
                    if player.active:
                        if player.last_action_log_prob is not None:
                            data['state'].append(player.last_state)
                            data['action_log_prob'].append(player.last_action_log_prob)
                            data['reward'].append(player.last_win_chips / self.big_blind)
                            data['value_pred'].append(player.last_value_pred)
                            player.last_action_log_prob = None
                continue
                
            self.deal_turn()
            self.deal_river()   
            winners = self.get_winners()
            self.divide_pot(winners)
            for winner in winners[0]:
                winner.is_winner = True
            for player in self.players:
                if player.active:
                    data['state'].append(player.last_state)
                    data['action_log_prob'].append(player.last_action_log_prob)
                    data['reward'].append(player.last_win_chips / self.big_blind)
                    data['value_pred'].append(player.last_value_pred)
                    player.last_action_log_prob = None
                        
            self.small_blind_index = (self.small_blind_index + 1) % self.n_players
            
    
    def run_game(self):
        self.deck.shuffle()
        self.deal_hole(n_cards=self.num_cards_in_hold)
        self.show_hold_cards()
        self._action_table()
        # Preflop Round
        