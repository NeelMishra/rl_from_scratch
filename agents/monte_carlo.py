from rl_from_scratch.agents.base import Agent
from collections import defaultdict
import numpy as np

class MonteCarlo(Agent):

    def __init__(self, num_states, num_actions, config):
        super().__init__()
        self.num_states = num_states # In k-armed bandit there is only one state the initial state, as taking any action will not do a state transition, because all the actions are still available to take with the same exact reward setup
        self.num_actions = num_actions
        self.config = config
        self.rng = np.random.default_rng(self.config.get('seed', 47)) # This is a practice that is taken from the coursera course, and although it is hard to truely avoid stochasticity, it helps to some extent

        self.returns = [[list() for __ in range(num_actions)] for _ in range(num_states)] # (state, action) -> return

        self.q_values = np.array([[self.config.get('initial_value', 0) for __ in range(num_actions)] for _ in range(num_states)], dtype=float) # (state, action) -> return
        self.num_of_times_action_taken = np.array([0 for _ in range(num_actions)])

    def step(self, episodes):

        # Update rule
        # Q_{t+1} = Q_{t} + alpha * (R_{t} - Q_{t})
        G = 0
        for episode in episodes[::-1]:
            state, action, reward = episode 
            
            G = self.config['discount'] * G + reward
            self.returns[state][action].append(G)

            self.q_values[state, action] =  np.mean(
                self.returns[state][action]
            )

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
    
    def start(self):
        pass

    def end(self):
        pass

