from rl_from_scratch.agents.base import Agent
import numpy as np

class SARSA(Agent):

    def __init__(self, num_states, num_actions, config):
        self.num_states = num_states
        self.num_actions = num_actions
        self.config = config
        self.rng = np.random.default_rng(
            self.config.get('seed', 47)
        )
        self.q_values = np.array([
            [
                self.config.get('initial_value', 0.0)
                for __ in range(self.num_actions)
            ] 
            for _ in range(self.num_states)
        ], dtype=float)

    def start(self, state):
        selected_action = self.policy(state)
        
        self.last_state = state
        self.last_action = selected_action

        return selected_action

    def step(self, state, reward):
        selected_action = self.policy(state)

        current_q_value = self.q_values[state][selected_action]
        past_q_value = self.q_values[self.last_state][self.last_action]

        self.q_values[self.last_state][self.last_action] = past_q_value + self.config.get('lr', 1e-3)  * (reward + self.config.get('discount', 0.9) * current_q_value - past_q_value)

        self.last_state = state
        self.last_action = selected_action

        return selected_action

    def end(self, reward):

        # Here we will not select any action or set any last state or last action
        # We will simply update the q-values

        past_q_value = self.q_values[self.last_state][self.last_action]
        self.q_values[self.last_state][self.last_action] = past_q_value + self.config.get('lr', 1e-3)  * (reward - past_q_value)

        

    def policy(self, state):
        
        dice = self.rng.random()
        if dice < self.config.get('eps', 5e-2):
            selected_action = self.rng.integers(0, self.num_actions)
        else:
            selected_action = np.argmax(self.q_values[state, :])
        
        return selected_action