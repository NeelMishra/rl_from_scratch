from collections import defaultdict
import numpy
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional


class Agent(ABC):
    '''
    This is an base class implementation of an episodic agent.
    It is meant to be inherited by other agents.

    Typical loop in rl training

    start_state = env.reset() / or env.init()
    action = agent.start(start_state)
    terminal = False
    while not terminal:
        latest_reward, latest_state, terminal = env.step(action)
        action = agent.step(latest_reward, latest_state)
    agent.end(latest_reward [this is your final reward]) # No actions to be performed here

    '''
    def __init__(self):
        self.last_state: Optional[Any] = None
        self.last_action: Optional[Any] = None
        self.time_step: int = 0 # steps within current episode
        self.episode: int = 0 # current episode number

        # We will not have q-values here because lot of methods don't use Q values at all (Policy gradient/actor-critic, model-based, bandits with preferences, etc.)

    @abstractmethod
    def policy(self, state):
        '''
        Agent's policy, given a state what action it should pick.
        '''
        raise NotImplementedError
    
    @abstractmethod
    def start(self, state):
        '''
        Defines what the agent should do at the start state.
        And output and action according to its policy.

        There will be no update here, because the agent has not taken any action earlier
        
        It will also set the last state and last action
        '''
        raise NotImplementedError
    
    @abstractmethod
    def step(self, reward, state):
        '''
        Agent will update it's estimate given the recent reward and state
        and output an action according to its policy.

        It will also set the last state and last action.
        '''
        raise NotImplementedError

    @abstractmethod
    def end(self, reward):
        '''
        Agent will just update and not select any action here.
        '''
        raise NotImplementedError