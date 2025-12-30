from rl_from_scratch.agents.base import Agent
import numpy as np
import torch

class ActorCritic(Agent):
    # This implementation utilizes neural network, but I think it should be extendable to the linear functions?
    
    # Assumes user provides distribution (probability) over the actions for the actor
    # Assumes actions are discrete

    def __init__(self, actor, critic, actor_optimizer, critic_optimizer, config):
        self.config = config

        self.eps = config.get('eps', 1e-3)
        self.device = config.get('device', 'cpu')
        self.gamma = config.get('gamma', 1e-1)

        self.rng = torch.Generator(
            device=self.device
        )
        self.rng.manual_seed(config.get('seed', 47))
        self.r_bar = torch.tensor(0.0, device=self.device)
        self.alpha_r_bar = config.get("alpha_r_bar", 1e-1)

        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

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

        
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        v_s = self.critic(self.last_state)
        with torch.no_grad():
            v_sp = self.critic(state)
        
        delta = reward - self.r_bar + self.gamma * (v_sp) - v_s
            
        self.r_bar += self.alpha_r_bar * delta.detach()

        critic_loss =  0.5 * torch.pow((delta), 2)
        actor_loss = -(delta.detach() * self.last_logp)

        critic_loss.backward()
        actor_loss.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        selected_action = self.policy(state)

        self.last_state = state
        self.last_action = selected_action

        return selected_action

    def end(self, reward):

        # Here we will not select any action or set any last state or last action
        # We will simply update the q-values
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        v_s = self.critic(self.last_state)
        delta = reward - self.r_bar - v_s


        critic_loss =  0.5 * torch.pow((delta), 2)
        actor_loss = -(delta.detach() * self.last_logp)

        critic_loss.backward()
        actor_loss.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()


    def policy(self, state):
        action_distribution = self.actor(state)
        action = action_distribution.sample()
        logp = action_distribution.log_prob(action)

        self.last_logp = logp
        
        return action
    
#### LLM Generated main block
if __name__ == "__main__":
    # Assumption: your ActorCritic class is defined ABOVE this block.

    import os
    import numpy as np
    import gymnasium as gym
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    from gymnasium.wrappers import RecordVideo

    # -----------------------------
    # Device (prefer MPS on Mac)
    # -----------------------------
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)

    # -----------------------------
    # Simple actor/critic networks
    # Actor returns a Distribution (Categorical)
    # Critic returns scalar V(s)
    # -----------------------------
    class MLPActor(nn.Module):
        def __init__(self, obs_dim: int, num_actions: int, hidden: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, num_actions),
            )

        def forward(self, s: torch.Tensor) -> Categorical:
            # Accept [obs_dim] or [1, obs_dim] and return Categorical over actions
            if s.dim() == 1:
                s = s.unsqueeze(0)
            logits = self.net(s).squeeze(0)  # [A]
            return Categorical(logits=logits)

    class MLPCritic(nn.Module):
        def __init__(self, obs_dim: int, hidden: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1),
            )

        def forward(self, s: torch.Tensor) -> torch.Tensor:
            # Accept [obs_dim] or [1, obs_dim] and return scalar ()
            if s.dim() == 1:
                s = s.unsqueeze(0)
            v = self.net(s).squeeze(0).squeeze(-1)  # ()
            return v

    # -----------------------------
    # Env
    # -----------------------------
    env_id = "CartPole-v1"
    env = gym.make(env_id)

    obs_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # -----------------------------
    # Seeds
    # -----------------------------
    seed = 47
    torch.manual_seed(seed)
    np.random.seed(seed)

    # -----------------------------
    # Build agent
    # -----------------------------
    actor = MLPActor(obs_dim, num_actions).to(device)
    critic = MLPCritic(obs_dim).to(device)

    actor_optim = optim.Adam(actor.parameters(), lr=1e-3)
    critic_optim = optim.Adam(critic.parameters(), lr=1e-3)

    config = {
        "device": device,
        "gamma": 0.99,
        "alpha_r_bar": 1e-2,
        "seed": seed,
    }

    agent = ActorCritic(actor, critic, actor_optim, critic_optim, config)

    # -----------------------------
    # Training loop
    # -----------------------------
    num_episodes = 400
    max_steps = 500
    returns = []

    actor.train()
    critic.train()

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)

        action = agent.start(obs)  # might be torch scalar tensor
        ep_return = 0.0

        for _ in range(max_steps):
            a_int = int(action) if not isinstance(action, (int, np.integer)) else action
            next_obs, reward, terminated, truncated, _ = env.step(a_int)
            ep_return += float(reward)

            if terminated or truncated:
                agent.end(reward)
                break
            else:
                action = agent.step(next_obs, reward)

        returns.append(ep_return)
        if (ep + 1) % 10 == 0:
            avg10 = sum(returns[-10:]) / 10.0
            print(f"Episode {ep+1:4d} | return={ep_return:6.1f} | avg10={avg10:6.1f}")

    env.close()

    # -----------------------------
    # Record ONE post-training episode video
    # -----------------------------
    video_dir = "videos_cartpole"
    os.makedirs(video_dir, exist_ok=True)

    eval_env = gym.make(env_id, render_mode="rgb_array")
    eval_env = RecordVideo(
        eval_env,
        video_folder=video_dir,
        episode_trigger=lambda episode_id: episode_id == 0,  # record only the first
        name_prefix="actor_critic_cartpole",
        disable_logger=True,
    )

    actor.eval()
    critic.eval()

    obs, _ = eval_env.reset(seed=seed + 999)
    done = False
    ep_ret = 0.0

    while not done:
        # Greedy action for nicer viewing (uses actor distribution probs)
        s = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            dist = actor(s)
            action = int(torch.argmax(dist.probs).item())

        obs, reward, terminated, truncated, _ = eval_env.step(action)
        ep_ret += float(reward)
        done = terminated or truncated

    eval_env.close()

    print(f"Recorded eval return = {ep_ret:.1f}")
    print(f"Video saved in folder: {video_dir}")
    print("Tip: if mp4 isn't created, install: pip install imageio imageio-ffmpeg")
