#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 19:25:44 2023

@author: oscar
"""

import random
import numpy as np
from Sumtree import SumTree

class PER:  # stored as ( s, a, r, s_ ) in SumTree
    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001


    def __init__(self, capacity, seed):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

    def __len__(self):
        return self.tree.n_entries

    def _get_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = np.empty((n, self.tree.data[0].size), dtype=float)
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(max(p, self.epsilon))
            batch[i, :] = data
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)