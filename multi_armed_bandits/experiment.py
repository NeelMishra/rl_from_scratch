from bandit_env import BanditEnv
import matplotlib.pyplot as plt
import numpy as np
from agents.epsilon_greedy import EpsilonGreedy


def train_loop(agent, env, epochs):
    
    # State will be 0 always for multi armed bandit as there is always 1 state
    selected_action = agent.start(0)
    reward = 0

    rewards_traj = [] # Array to store reward trajectory
    actions_traj = [selected_action] # Array to store action trajectory

    for epoch in range(epochs):
        
        # Receive feedback from the environment
        reward = env.pull_arm(selected_action)
        
        # Agent will take it's step based on the reward
        selected_action = agent.step(reward, 0)

        # logging steps
        rewards_traj.append(reward)
        actions_traj.append(selected_action)

    agent.end(reward)

    return rewards_traj, actions_traj

def plot_rewards(rewards_traj, label):
    plt.plot(np.arange(len(rewards_traj)), rewards_traj)

def init_env_and_agent(env_config, agent_config):
    agent_class = agent_config['agent_class']
    env = env_config['env_class']

    env_obj = env(env_config['k_arms'])
    agent_obj = agent_class(1, env_config['k_arms'], agent_config)

    return env_obj, agent_obj

env_config = {
    'env_class' : BanditEnv,
    'k_arms' : 10,
}

agent_config = {
    'lr' : 1e-2,
    'eps' : 0.05,
    'initial_value' : 0,
    'agent_class' : EpsilonGreedy
}

env, agent = init_env_and_agent(env_config=env_config, agent_config=agent_config)

rewards_traj, actions_traj = train_loop(agent=agent, env=env, epochs=10000)


# Plotting
plt.figure(figsize=(10, 6))
plot_rewards(rewards_traj, 'simple_epsilon_greedy_agent')
plt.xlabel("Timesteps/Iterations")
plt.ylabel("Rewards recieved")
plt.savefig("rewards-vs-iterations.png")

env.plot_all(10000)
print(f"Q-values : ", agent.q_values[0])
print(f"Max action discovered by agent is {1 + np.argmax(agent.q_values[0])}")