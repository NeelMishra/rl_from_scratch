import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    '''
    This class is responsible to create the plots related to the rl experiments. 

    So far the below are supported.

    1. Raw line curves plot
    2. Aggregated avg curves (over a window)
    3. Average across multiple runs

    '''

    def __init__(self):
        pass

    def plot_regular(self, rewards, figsize, x_label, y_label, title, label, filename):
        plt.clf()
        plt.figure(figsize=figsize)
        plt.plot(np.arange(len(rewards)), rewards)

        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.title(title)
        plt.legend()
        plt.savefig(filename, dpi=300)
        
    def plot_rolling_avg(self, rewards, figsize, x_label, y_label, title, filename, label, window_size):
        plt.clf()
        plt.figure(figsize=figsize)
        
        kernel = np.ones(window_size)/window_size
        averaged_rewards = np.convolve(rewards, kernel, mode='valid')

        plt.plot(np.arange(len(averaged_rewards)), averaged_rewards, label=label)

        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.title(title)
        plt.legend()
        plt.savefig(filename, dpi=300)

    def plot_avg_across_multiple_runs(self, rewards, figsize, x_label, y_label, title, label, filename):
        # rewards => (runs, iterations)
        plt.clf()
        plt.figure(figsize=figsize)

        rewards_across_runs = np.array(rewards)
        mean_across_runs = rewards_across_runs.mean(axis=0)
        std_across_runs = rewards_across_runs.std(axis=0)

        x = np.arange(len(mean_across_runs))

        plt.plot(x, mean_across_runs, label=label)
        plt.fill_between(x, mean_across_runs - std_across_runs, mean_across_runs + std_across_runs, label='+/- 1 std')
        
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.title(title)
        plt.legend()
        plt.savefig(filename, dpi=300)