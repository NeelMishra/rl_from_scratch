import gymnasium as gym
from rl_from_scratch.agents.sarsa import SARSA
import numpy as np

'''
The code is plug and play, you can use from the below environment names
Blackjack-v1
CliffWalking-v1
FrozenLake-v1
Taxi-v3
'''
env_name = 'CliffWalking-v1'
env = gym.make(env_name)
num_actions = env.action_space.n
num_states = env.observation_space.n

config = {
    'initial_value' : 1,
    'seed' : 47,
    'eps' : 1e-1,
    'lr' : 1e-3
}

agent = SARSA(num_states=num_states,
              num_actions=num_actions,
              config=config)

def train(agent, env, epochs):
    
    reward_trajectory = []

    for epoch in range(epochs):
        # It's episodic
        start_state, prob_dict = env.reset()
        selected_action = agent.start(start_state)

        terminated = False
        truncated = False

        reward = 0
        rewards_this_epoch = []
        while not terminated and not truncated:
            state, reward, terminated, truncated, _ = env.step(selected_action)
            reward_trajectory.append(reward)
            rewards_this_epoch.append(reward)
            if terminated or truncated:
                break
            selected_action = agent.step(state, reward)
            
        if rewards_this_epoch:
            sum_reward = sum(rewards_this_epoch)
        else:
            sum_reward = 0
        if epoch % (epochs//10) == 0:
            print(f"epoch : {epoch}, reward obtained this epoch {sum_reward}")

        agent.end(reward)
        
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
    
    print(f"The total reward obtained in the visualization step is {sum(reward_trajectory)}")
    return reward_trajectory


epochs = 200000
train(agent, env, epochs)
visualize(agent)