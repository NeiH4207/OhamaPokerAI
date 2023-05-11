from src.ohama_poker import Game as OhamaGame

def main():
    playerNames = ['Alice (lv0)', 'Bob (lv4)', 'Cindy (lv6)', 'David (lv24)']
    saved_model_path = 'trained_models/ohama_poker/'
    poker = OhamaGame(n_players=4, big_blind=10, playerNames=playerNames, 
                      model_path=saved_model_path   )
    # poker.model.load('trained_models/ohama_poker/model_{}.pt'.format(16300))
    poker.simulate()
    each_model_path = ['trained_models/ohama_poker/model_{}.pt'.format(100),
                        'trained_models/ohama_poker/model_{}.pt'.format(1000),
                        'trained_models/ohama_poker/model_{}.pt'.format(5000),
                        'trained_models/ohama_poker/model_{}.pt'.format(18500)]
    # load model for all players
    for i, player in enumerate(poker.players):
        player.nnet.load(each_model_path[i])
    ranking_orders = poker.play()
    print("Ranking:")
    for i, player_name in enumerate(reversed(ranking_orders)):
        print(player_name)
    
if __name__ == "__main__":
    main()