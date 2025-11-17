import gymnasium as gym
from rl_from_scratch.agents.monte_carlo import MonteCarlo
import numpy as np

'''
The code is plug and play, you can use from the below environment names
Blackjack-v1
CliffWalking-v1
FrozenLake-v1
Taxi-v3
'''
env_name = 'FrozenLake-v1'
env = gym.make(env_name)
num_actions = env.action_space.n
num_states = env.observation_space.n

config = {
    'initial_value': 0.0,
    'seed': 47,
    'eps': 1.0,          
    'lr': 1e-2,          
    'discount': 0.99        
}

agent = MonteCarlo(num_states=num_states,
              num_actions=num_actions,
              config=config)

def train(agent, env, epochs):
    
    reward_trajectory = []

    for epoch in range(epochs):
        # It's episodic
        state, prob_dict = env.reset()

        terminated = False
        truncated = False

        reward = 0
        rewards_this_epoch = []
        episodes = []
        
        while not terminated and not truncated:
            selected_action = agent.policy(state)
            next_state, reward, terminated, truncated, _ = env.step(selected_action)

            episodes.append((state, selected_action, reward)) 

            reward_trajectory.append(reward)
            rewards_this_epoch.append(reward)

            if terminated or truncated:
                break
            # simple epsilon decay
            agent.config['eps'] = max(0.05, agent.config['eps'] * 0.9995)
            state = next_state
        agent.step(episodes)
        if rewards_this_epoch:
            sum_reward = sum(rewards_this_epoch)
        else:
            sum_reward = 0
        if epoch % (epochs//100) == 0:
            print(f"epoch : {epoch}, reward obtained this epoch {sum_reward}")

        
    return reward_trajectory

def visualize(agent):
    env = gym.make(env_name, render_mode='human')
    start_state, prob_dict = env.reset()

    selected_action = agent.policy(start_state)
    reward_trajectory = []

    terminated, truncated = False, False

    while not terminated and not truncated:
        state, reward, terminated, truncated, _ = env.step(selected_action)
        reward_trajectory.append(reward)
        # render the new state
        env.render()
        if terminated or truncated:
            break
        selected_action = agent.policy(state)
        iterations += 1
    
    print(f"The total reward obtained in the visualization step is {sum(reward_trajectory)}")
    return reward_trajectory


epochs = 100000
train(agent, env, epochs)
visualize(agent)