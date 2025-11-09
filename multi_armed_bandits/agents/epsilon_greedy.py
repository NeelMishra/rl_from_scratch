from agents.base import Agent
from collections import defaultdict
import numpy as np

class EpsilonGreedy(Agent):

    def __init__(self, num_states, num_actions, config):
        super().__init__()
        self.num_states = num_states # In k-armed bandit there is only one state the initial state, as taking any action will not do a state transition, because all the actions are still available to take with the same exact reward setup
        self.num_actions = num_actions
        self.config = config
        self.rng = np.random.default_rng(self.config.get('seed', 47)) # This is a practice that is taken from the coursera course, and although it is hard to truely avoid stochasticity, it helps to some extent

        self.q_values = np.array([[self.config.get('initial_value', 0) for __ in range(num_actions)] for _ in range(num_states)], dtype=float) # (state, action) -> return
        
    def start(self, state):
        self.time_step = 0

        selected_action = self.policy(state)

        self.last_state = state
        self.last_action = selected_action

        return selected_action

    def step(self, reward, state):

        # Update rule
        # Q_{t+1} = Q_{t} + alpha * (R_{t} - Q_{t})
        Q_t = self.q_values[self.last_state,self.last_action]
        self.q_values[self.last_state,self.last_action] =  (
            Q_t +
            self.config.get('lr', 1e-3) * (reward - Q_t)
        )

        selected_action = self.policy(state)

        self.last_state = state
        self.last_action = selected_action
        self.time_step += 1

        return selected_action

    def end(self, reward):
        
        # Just update the q-values
        Q_t = self.q_values[self.last_state,self.last_action]
        self.q_values[self.last_state,self.last_action] =  (
            Q_t +
            self.config['lr'] * (reward - Q_t)
        )
 
        self.episode += 1
        return 

        

    def policy(self, state):
        # Roll dice
        dice_roll = self.rng.random()

        ## if dice is in favor of exploration
        if dice_roll <= self.config.get('eps', 0.1):
            selected_action = self.rng.integers(0, self.num_actions)
        else:
            selected_action = np.argmax(self.q_values[state, :])

        # return the selected action
        return selected_action

if __name__ == '__main__':
    test_config = {
        'lr' : 1e-1,
        'eps' : 1e-1,
        'initial_value' : 3,
    }
    EG = EpsilonGreedy(1,10,test_config)