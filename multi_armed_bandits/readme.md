# Multi-Armed Bandits Design

## Goals
- Provide a flexible sandbox for investigating $k$-armed bandit strategies (ε-greedy, optimistic, UCB, Thompson sampling, gradient bandits, etc.).
- Support rapid experimentation across different reward distributions, agent hyperparameters, and horizons.
- Produce reproducible metrics and plots (reward, regret, optimal-action probability) for each experiment.

## Core Abstractions
| Component | Responsibilities |
| --- | --- |
| `BanditArm` | Encapsulates the true reward distribution (Bernoulli, Gaussian, etc.) and handles random sampling.|
| `BanditEnv` | Stores a list of `BanditArm` objects, exposes `pull(arm_index)` to return sampled reward, and tracks which arm is optimal for regret calculations.|
| `Agent` (abstract) | Defines `select_arm(timestep)` and `update(arm_index, reward)`; concrete subclasses implement specific strategies.|
| `Experiment` | Orchestrates multiple episodes/runs, mediates agent–environment interaction, and logs metrics.|
| `MetricsLogger` | Aggregates per-step data (reward, regret, action counts) and writes CSV/Parquet artifacts for later analysis.|
| `Plotter` | Loads saved metrics and renders standard figures with confidence intervals.|

## Configuration
- Represent experiment settings with a structured config object (dataclass or YAML): number of arms, reward distribution specs, horizon, number of runs, random seeds, agent parameters (e.g., ε initial/decay, optimism value, UCB confidence `c`, Beta priors).
- Allow batch sweeps via an iterable of configs so you can compare agents or hyperparameters in one CLI invocation.

## Experiment Flow
1. **Initialization**: Build the environment from config (instantiate each `BanditArm` with its parameters), instantiate an agent per config, reset seeds/loggers.
2. **Loop** (per run, per timestep):
   - `arm = agent.select_arm(t)`
   - `reward = env.pull(arm)`
   - `agent.update(arm, reward)`
   - `logger.record(t, arm, reward, optimal_arm_flag, regret)`
3. **Aggregation**: After `n_runs`, compute mean/median and std/CI for all tracked metrics; persist both raw traces and aggregates.
4. **Reporting**: Trigger plotting or produce a Markdown/HTML summary referencing generated figures.

## Metrics to Track
- Instantaneous reward and cumulative reward.
- Optimal-arm indicator (1 if chosen arm matches env optimum) to estimate probability of optimal action.
- Instantaneous regret (`μ* - reward`) and cumulative regret.
- Arm-selection counts/frequencies for diagnosing exploration.
- Agent-specific diagnostics (e.g., ε value after decay, UCB confidence bounds, Thompson samples) when helpful for debugging.

## Plotting Guidelines
- Use Matplotlib/Seaborn or Plotly; central helper should accept metric arrays, x-axis label, title, and output path.
- Standard plots:
  - Average reward vs. timestep with shaded 95% CI.
  - Cumulative regret vs. timestep.
  - Probability of optimal action vs. timestep.
  - Arm-selection histogram/bar plot per agent.
  - Hyperparameter sweep plots (e.g., different ε decay rates) with consistent color palette.
- Store figures under `figures/{agent}/{metric}.png` (or `.html` for interactive), ensuring directories auto-create.

## Testing & Validation
- **Deterministic tests**: Mock reward draws to validate agent update rules (e.g., UCB confidence formula, Thompson Beta posterior update).
- **Smoke tests**: Run a tiny experiment (k=2, horizon=50) during CI to confirm the pipeline runs end-to-end and produces expected files.
- **Plot verification**: Feed synthetic metric tensors into the Plotter to ensure figure generation works even without running full experiments.

## Future Extensions
- Contextual bandits by extending `BanditEnv` to emit feature vectors and conditioning agent policies.
- Dashboard/Notebook integration for interactive exploration of logged metrics.
- Parallel execution (multiprocessing or Ray) for large sweep grids.
- Reward distributions beyond stationary Bernoulli/Gaussian (e.g., non-stationary drift, heavy-tailed).
