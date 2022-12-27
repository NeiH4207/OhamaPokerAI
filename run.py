from src.texas_holdem_poker import Game as TexasHoldemGame
from src.ohama_poker import Game as OhamaGame

def main():
    playerNames = ['Alice', 'Bob', 'Cindy', 'David']
    saved_model_path = 'trained_models/ohama_poker/'
    poker = OhamaGame(n_players=4, big_blind=10, playerNames=playerNames, 
                      model_path=saved_model_path   )
    poker.model.load('trained_models/ohama_poker/model_{}.pt'.format(14999))
    poker.simulate()
    # each_model_path = ['trained_models/ohama_poker/model_{}.pt'.format(1999),
    #                     'trained_models/ohama_poker/model_{}.pt'.format(5999),
    #                     'trained_models/ohama_poker/model_{}.pt'.format(10999),
    #                     'trained_models/ohama_poker/model_{}.pt'.format(14999)]
    # # load model for all players
    # for i, player in enumerate(poker.players):
    #     player.nnet.load(each_model_path[i])
    # poker.play()
    
if __name__ == "__main__":
    main()