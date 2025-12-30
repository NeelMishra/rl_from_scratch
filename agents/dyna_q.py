from rl_from_scratch.agents.base import Agent
from collections import defaultdict
import numpy as np

class DynaQ(Agent):

    def __init__(self, num_states, num_actions, config):
        self.num_states = num_states
        self.num_actions = num_actions
        self.config = config
        self.rng = np.random.default_rng(
            self.config.get('seed', 47)
        )
        self.planning_steps = config.get('planning_steps', 5)
        self.model = defaultdict(list)
        self.buffer_size = config.get('buffer_size_per_state_action', 5) # max number of episode to store per state action pair

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
    
    def planning_step_func(self):
        key = list(self.model.keys())
        for planning_step in range(self.planning_steps):
            random_state, random_action = key[self.rng.integers(len(self.model))]

            transitions = self.model[(random_state, random_action)]
            reward, next_state, terminal = transitions[self.rng.integers(len(transitions))]

            past_q_value = self.q_values[random_state][random_action]

            if not terminal:
                max_q_value = np.max(self.q_values[next_state])
                self.q_values[random_state][random_action] = past_q_value + self.config.get('lr', 1e-3)  * (reward + self.config.get('discount', 0.9) * max_q_value - past_q_value)
            else:
                self.q_values[random_state][random_action] = past_q_value + self.config.get('lr', 1e-3) * (reward - past_q_value)



    def step(self, state, reward):
        
        self.model[(self.last_state, self.last_action)].append((reward, state, False))
        if len(self.model[(self.last_state, self.last_action)]) > self.buffer_size:
            self.model[(self.last_state, self.last_action)].pop(0)

        max_q_value = np.max(self.q_values[state])
        past_q_value = self.q_values[self.last_state][self.last_action]

        self.q_values[self.last_state][self.last_action] = past_q_value + self.config.get('lr', 1e-3)  * (reward + self.config.get('discount', 0.9) * max_q_value - past_q_value)

        self.planning_step_func()

        selected_action = self.policy(state)

        self.last_state = state
        self.last_action = selected_action

        return selected_action

    def end(self, reward):

        # Here we will not select any action or set any last state or last action
        # We will simply update the q-values
        self.model[(self.last_state, self.last_action)].append((reward, None, True))
        if len(self.model[(self.last_state, self.last_action)]) > self.buffer_size:
            self.model[(self.last_state, self.last_action)].pop(0)
            
        past_q_value = self.q_values[self.last_state][self.last_action]
        self.q_values[self.last_state][self.last_action] = past_q_value + self.config.get('lr', 1e-3)  * (reward - past_q_value)

        self.planning_step_func()
        

    def policy(self, state):
        
        dice = self.rng.random()
        if dice < self.config.get('eps', 5e-2):
            selected_action = self.rng.integers(0, self.num_actions)
        else:
            selected_action = np.argmax(self.q_values[state, :])
        
        return selected_action
    
    def policy_deterministic(self, state):
        
        selected_action = np.argmax(self.q_values[state, :])
        
        return selected_action