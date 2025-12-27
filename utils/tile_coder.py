import numpy as np

class IHT:

    def __init__(self, size):

        if type(size) != int or size <= 0:
            raise ValueError("Size must be integer and > 0")

        self.size = size
        self._dict = {}
        self.overfull_count = 0

    def get_index(self, coords, readonly=False):
        key = tuple(coords)
        if key in self._dict:
            return self._dict[key]
        else:
            if readonly == True:
                return hash(key) % self.size
            else:
                if len(self._dict) < self.size:
                    next_id = len(self._dict)
                    self._dict[key] = next_id
                    return next_id
                else:
                    self.overfull_count += 1
                    return hash(key) % self.size
                



class TileCoder:

    def __init__(self, num_tilings, tiles_per_dim, state_low, state_high, hash_size, num_actions):
        self.num_tilings = num_tilings

        self.tiles_per_dim = np.array(tiles_per_dim, dtype=int)
        self.state_low = np.array(state_low, dtype=float)
        self.state_high = np.array(state_high, dtype=float)
        
        self.hash_size = hash_size
        self.num_actions = num_actions

        self.iht = IHT(hash_size)
        self.scale = self.tiles_per_dim/(self.state_high - self.state_low)
