from base import Agent
from collections import defaultdict

class EpsilonGreedy(Agent):

    def __init__(self):
        self.q_values = defaultdict(int) # (state, action) -> return
        

    def start(self):
        pass

    def step(self):
        pass

    def end(self):
        pass

    def policy(self):
        pass

if __name__ == '__main__':
    EG = EpsilonGreedy()