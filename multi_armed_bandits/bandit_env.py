import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class BanditEnv:

    def __init__(self, k):
        self.k = k
        self.arms = {}

        for i in range(k):
            self.arms[i] = self.sample_reward(0, 1)

    def sample_reward(self, mean, var):
        return np.random.normal(mean, var)

    def pull_arm(self, arm_index):
        return np.random.normal(self.sample_reward(self.arms[arm_index], 1))

    def plot_all(self, num_points = 10):
        data_points = [list() for _ in range(self.k)]
        
        for i in range(self.k):
            for point_num in range(num_points):
                data_points[i].append(self.pull_arm(i))


        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plots the raw distributions
        ax.violinplot(
            data_points,
            positions = np.arange(1, self.k+1),
            showmeans=False,
            showmedians=False,
            showextrema=False,
            widths=0.8
        )

        ## Now we need to beautify it
        
        # Rectify the x ticks => We need to have 1 to k ticks representing each arm
        plt.xticks(np.arange(1, self.k+1))

        # Label the x and y axes\
        plt.ylabel("Reward Distribution")
        plt.xlabel("Action/Arm Number (Index/Position)")

        plt.show()

if __name__ == '__main__':
    BE = BanditEnv(10)
    
    BE.plot_all(10000)
