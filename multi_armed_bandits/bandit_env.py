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
        return np.random.normal(self.arms[arm_index], 1)
    
    def get_distributions(self, num_points = 10):
        data_points = [list() for _ in range(self.k)]
        
        for i in range(self.k):
            for point_num in range(num_points):
                data_points[i].append(self.pull_arm(i))
        
        return data_points

    def plot_all(self, num_points = 10):
        
        data_points = self.get_distributions(num_points)

        fig, ax = plt.subplots(figsize=(10, 6))
        
        violin_width = 0.6

        # Plots the raw distributions
        ax.violinplot(
            data_points,
            positions = np.arange(1, self.k+1),
            showmeans=False,
            showmedians=False,
            showextrema=False,
            widths=violin_width
        )

        ## Now we need to beautify it
        
        # Rectify the x ticks => We need to have 1 to k ticks representing each arm
        plt.xticks(np.arange(1, self.k+1))

        # Label the x and y axes\
        plt.ylabel("Reward Distribution")
        plt.xlabel("Action/Arm Number (Index/Position)")

        # 0 reference line
        ax.axhline(y = 0, color='gray', linestyle='--', linewidth=1)

        # Showing information about variance in each of the distribution plot
        for key, val in self.arms.items():
            ax.hlines(val, key+1-violin_width/2, key+1+violin_width/2, colors='black', linewidth=1.2)


        # Saving the plot, if you want to show the plot change the below line to plt.show()
        plt.savefig(f"Reward-Distribution-{self.k}-arms.png")
if __name__ == '__main__':
    BE = BanditEnv(10)
    
    BE.plot_all(10000)
