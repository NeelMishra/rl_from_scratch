import numpy as np
import math
class IHT:

    def __init__(self, size):

        if not isinstance(size, int) or size <= 0:
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

    def _scale_state(self, state):

        state = np.array(state, dtype=float)
        scaled = (state - self.state_low) * self.scale

        return scaled
    
    def get_active_tiles(self, state, action):

        if not (0 <= action < self.num_actions and isinstance(action, (int, np.integer)) ):
            raise ValueError(f"Action must be between [0 to the {self.num_actions}) and must be of type int")

        scaled_state = self._scale_state(state)

        active_tiles = self._tiles(
                                  floats = scaled_state,
                                  ints=(action, ),
                                  readonly=False
                                  )

        return active_tiles
    
    def _tiles(self, floats, ints, readonly=False):

        q_floats = [math.floor(curr_float * self.num_tilings) for curr_float in floats]
        feature_idxs = []
        for tiling in range(self.num_tilings):
            
            b = tiling
            coords = [b]
            for q in q_floats:
                curr_bin = (q+b) // self.num_tilings
                coords.append(curr_bin)
                b += 2 * tiling
            coords.extend(ints)
            feature_idxs.append(self.iht.get_index(coords, readonly=readonly))
        return feature_idxs
            
