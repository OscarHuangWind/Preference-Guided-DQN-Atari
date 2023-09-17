#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 19:21:55 2023

@author: oscar
"""

import torch
import torch.nn as nn
import numpy as np
import random

if torch.cuda.is_available():
    device = torch.device("cuda", 0 if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print('Use:', device)

##############################################################################
class Net(nn.Module):

    def __init__(self, height, width, channel, num_outputs, dueling, preference, seed):
        super(Net, self).__init__()
        self.dueling = dueling
        self.preference = preference
        self.height = height
        self.width = width
        self.feature_dim = 512
        linear_input_size = self.linear_size_input()
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.features = nn.Sequential(
            nn.Conv2d(channel, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        if (self.dueling):
            self.advantage_func = nn.Sequential(
                nn.Linear(linear_input_size, self.feature_dim),
                nn.ReLU(),
                nn.Linear(self.feature_dim, num_outputs)
                )
            
            self.state_value_func = nn.Sequential(
                nn.Linear(linear_input_size, self.feature_dim ),
                nn.ReLU(),
                nn.Linear(self.feature_dim , 1)
                )
        elif (self.preference):
            self.actor_func = nn.Sequential(
                nn.Linear(linear_input_size, self.feature_dim),
                nn.ReLU(),
                nn.Linear(self.feature_dim, num_outputs),
                )
            
            self.q_func = nn.Sequential(
                nn.Linear(linear_input_size, self.feature_dim),
                nn.ReLU(),
                nn.Linear(self.feature_dim, num_outputs),
                )
        else:
            self.fc = nn.Sequential(
                nn.Linear(linear_input_size, self.feature_dim),
                nn.ReLU(),
                nn.Linear(self.feature_dim, num_outputs)
                )

    def linear_size_input(self):    
        convw = self.conv2d_size_3rd(self.conv2d_size_2nd(self.conv2d_size_1st(self.width)))
        convh = self.conv2d_size_3rd(self.conv2d_size_2nd(self.conv2d_size_1st(self.height)))
        return convw * convh * 64

    def conv2d_size_1st(self, size, kernel_size = 8, stride = 4):
        return (size - (kernel_size - 1) - 1) // stride  + 1
        
    def conv2d_size_2nd(self, size, kernel_size = 4, stride = 2):
        return (size - (kernel_size - 1) - 1) // stride  + 1
   
    def conv2d_size_3rd(self, size, kernel_size = 3, stride = 1):
        return (size - (kernel_size - 1) - 1) // stride  + 1

    def forward(self, x):
        x = x.to(device)
        x = self.features(x)
        x = x.contiguous().view(-1, self.linear_size_input())
        if (self.dueling):
            advantage_vec = self.advantage_func(x)
            value_scalar = self.state_value_func(x)
            x = value_scalar + advantage_vec - advantage_vec.mean()
            return x
        elif (self.preference):
            q_value = self.q_func(x)
            action_distribution = self.actor_func(x)
            normalize = nn.Softmax(dim=0)
            action_distribution = normalize(action_distribution.T).T
            return action_distribution, q_value
        else:
            x = self.fc(x)
            return x
        
##############################################################################
