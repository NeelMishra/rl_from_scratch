## RL from Scratch

Repo for implementing reinforcement learning (RL) algorithms from scratch. The goal is to internalize the theory, build intuition, and leave behind a transparent reference that others can study or extend.

### Repository Charter

1. Implement multiple RL algorithms entirely from scratch (numpy, basic plotting, and lightweight utilities are fine; no prebuilt RL libraries).
2. Large Language Models may assist with high-level design discussions, README drafting, or plotting tips, but **never** with writing or editing the actual source code.
3. Every implemented algorithm must cite at least one authoritative resource (paper, lecture, or blog) so readers can trace the original idea.

### Roadmap

| Track | Candidate Algorithms | Focus |
| --- | --- | --- |
| Classic control | Tabular value iteration, policy iteration, Monte Carlo control, SARSA/Q-learning | Gymnasium-style toy problems for clarity. |
| Policy gradient | REINFORCE, baseline variants, natural policy gradient | Keep derivations alongside code comments to show gradient logic. |
| Deep RL | DQN (and extensions), Actor-Critic, A2C/A3C, PPO | Start from minimal MLPs; add replay buffers, target nets, and advantage estimates incrementally. |
| Exploration & planning | Dyna-Q, prioritized sweeping, curiosity-driven bonuses | Compare model-free vs. model-based behavior. |

Progress can happen in any order; correctness and clear explanations beat sheer volume.

### Development Workflow

- Prototype math on paper or in notebooks before translating into modules.
- Prefer small, composable utilities (buffers, schedulers) to avoid tangled scripts.
- Document each algorithm’s assumptions, limitations, and failure modes in-place.

### Recommended Resources

- Sutton & Barto, *Reinforcement Learning: An Introduction* – foundational coverage for dynamic programming and tabular control.
- David Silver’s RL Course (UCL/DeepMind) – concise lectures on value-based and policy gradient families.
- Spinning Up in Deep RL (OpenAI) – practical guide for actor-critic, PPO, and TRPO implementations.
- DQN Nature Paper (Mnih et al., 2015) – reference for replay buffers, target networks, and stability tricks.
- PPO Paper (Schulman et al., 2017) – canonical objective and clipping rationale for modern policy optimization.
- CleanRL project write-ups – sanity checks for learning curves and implementation gotchas.

Each algorithm directory should link back to the specific resource(s) it follows most closely.

### Contributing

1. Branch off `main`, naming the branch after the algorithm or experiment.
2. Verify adherence to the charter before opening a pull request (no LLM-generated code, cite resources, include evaluation notes).
3. Request human review that focuses on numerical correctness, readability, and reproducibility.

Happy hacking, and may your reward signals be informative!
