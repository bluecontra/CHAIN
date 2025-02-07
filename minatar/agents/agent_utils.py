import random
from collections import namedtuple

import numpy as np

from agents.sum_tree import SumTree

###########################################################################################################
# class replay_buffer
#
# A cyclic buffer of a fixed size containing the last N number of recent transitions.  A transition is a
# tuple of state, next_state, action, reward, is_terminal.  The boolean is_terminal is used to indicate
# whether if the next state is a terminal state or not.
#
###########################################################################################################
transition = namedtuple('transition', 'state, action, reward, next_state, is_terminal')

class replay_buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []

    def add(self, *args):
        # Append when the buffer is not full but overwrite when the buffer is full
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition(*args))
        else:
            self.buffer[self.location] = transition(*args)

        # Increment the buffer location
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        batch = transition(*zip(*samples))
        return batch


class prioritized_replay_buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = np.array([None for _ in range(buffer_size)])
        self.sum_tree = SumTree(capacity=buffer_size)

    def add(self, *args):
        # Append when the buffer is not full but overwrite when the buffer is full
        ptran = transition(*args)
        self.buffer[self.location] = ptran

        # Update the sum tree - set the newcoming sample to the highest priority
        self.sum_tree.set(node_index=self.location, value=self.sum_tree.max_recorded_priority)

        # Increment the buffer location
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        indices = self.sum_tree.stratified_sample(batch_size)

        # sample the transitions from buffer and sample the priority from sum_tree
        samples = self.buffer[np.array(indices)]
        batch = transition(*zip(*samples))
        priorities = self.sum_tree.get(indices)
        priority_probs = priorities / self.sum_tree.nodes[0][0]
        # return indices together for the handle for priority update
        return batch, priority_probs, indices

    def update_priority(self, indices, values):
        for index, v in zip(indices, values):
            self.sum_tree.set(node_index=index, value=v)

    def get_buffer_actual_size(self):
        return self.location if self.buffer[-1] is None else self.buffer_size