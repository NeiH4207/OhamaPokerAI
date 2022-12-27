import numpy as np

class Random(object):
    def __init__(self) -> None:
        pass
    
    def get_ohama_action(self, state):
        # action 0: all-in, action 1: fold
        action = np.random.randint(0, 2)
        return action
    
    def get_texas_action(self, state):
        # action 0: check, action 1: bet, action 2: call, action 3: raise, action 4: fold
        action = np.random.randint(0, 5)
        value = 0
        if action == 1:
            value = np.random.randint(1, state['player-chips'][state['player-id']])
        elif action == 3:
            value = np.random.randint(1, state['player-chips'][state['player-id']])
        return (action, value)