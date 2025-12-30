from rl_from_scratch.agents.base import Agent
import numpy as np
import torch

class SemiGradientSARSA(Agent):
    # This implementation utilizes neural network, but I think it should be extendable to the linear functions?

    def __init__(self, q_net, optimizer, num_states, num_actions, config):
        self.num_states = num_states
        self.num_actions = num_actions
        self.config = config

        self.eps = config.get('eps', 1e-3)
        self.device = config.get('device', 'cpu')
        self.gamma = config.get('gamma', 1e-1)

        self.rng = torch.Generator(
            device=self.device
        )
        self.rng.manual_seed(config.get('seed', 47))

        self.q_net = q_net.to(self.device)
        self.optimizer = optimizer

    def start(self, state):
        state = torch.as_tensor(state, dtype=torch.float32)
        state = state.to(self.device)
        selected_action = self.policy(state)
        
        self.last_state = state
        self.last_action = selected_action

        return selected_action

    def step(self, state, reward):
        state = torch.as_tensor(state, dtype=torch.float32)
        state = state.to(self.device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)

        selected_action = self.policy(state)
        self.optimizer.zero_grad()

        past_q_value = self.q_net(self.last_state)[self.last_action]
        with torch.no_grad():
            curr_q_val = self.q_net(state)[selected_action]
            target = reward + self.gamma * curr_q_val
        
        loss = 0.5 * torch.pow((target-past_q_value), 2)
        loss.backward()
        self.optimizer.step()

        self.last_state = state
        self.last_action = selected_action

        return selected_action

    def end(self, reward):

        # Here we will not select any action or set any last state or last action
        # We will simply update the q-values
        self.optimizer.zero_grad()

        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            target = reward

        past_q_value = self.q_net(self.last_state)[self.last_action]
        
        loss = 0.5 * torch.pow((target-past_q_value), 2)
        loss.backward()
        self.optimizer.step()


    def policy(self, state):

        dice = torch.rand((), generator=self.rng, device=self.device)
        if dice < self.eps:
            selected_action = torch.randint(0, self.num_actions, (), generator=self.rng, device=self.device).item()
        else:
            with torch.no_grad():
                selected_action = torch.argmax(self.q_net(state), dim=-1).item()
        
        return selected_action