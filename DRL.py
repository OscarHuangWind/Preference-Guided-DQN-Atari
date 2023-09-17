#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 19:20:43 2023

@author: oscar
"""

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_value_
from torch.distributions.categorical import Categorical

from Network import Net
from Network import device
from Prioritized_memory import PER

class ReplayMemory(object):

    def __init__(self, capacity, dimension, seed):
        
        self.capacity = capacity
        self.frame_buffer = None
        self.action_buffer = None
        self.reward_buffer = None
        self.memory_counter = 0
        self.seed = seed
        
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def store_frame(self, frame):
        if len(frame.shape) > 1:
            frame = frame.transpose(2, 0, 1)
        
        if self.frame_buffer is None:
            self.frame_buffer = np.zeros([self.capacity] + list(frame.shape), dtype = np.uint8)
            self.action_buffer = np.zeros([self.capacity], dtype = np.int64)
            self.reward_buffer = np.zeros([self.capacity], dtype = np.float32)

        index = self.memory_counter % self.capacity
        self.frame_buffer[index, :] = frame
        self.memory_counter += 1
        
        return index

    def store_state(self, idx, action, reward):
        self.action_buffer[idx] = action
        self.reward_buffer[idx] = reward

    def push(self, transition):
        """Save a transition"""
        index = self.memory_counter % self.capacity
        self.memory[index,:] = transition
        self.memory_counter += 1

    def sample(self, batch_size):
        up_bound = min(self.memory_counter - 1, self.capacity - 1)
        sample_index = np.random.choice(up_bound, batch_size)
        sample_index_next = sample_index + 1
        sample_index_next[sample_index_next > up_bound] = up_bound
        return self.frame_buffer[sample_index, :], self.action_buffer[sample_index],\
               self.reward_buffer[sample_index], self.frame_buffer[sample_index_next, :]

    def __len__(self):
        return self.memory_counter

class DQN():
    
    def __init__(self,
                 height,
                 width,
                 channel,
                 n_obs,
                 n_actions,
                 DOUBLE,
                 DUELING,
                 PRIORITY,
                 IMPORTANTSAMPLING,
                 PREFERENCE,
                 ENTROPY,
                 JOINT,
                 BATCH_SIZE,
                 GAMMA,
                 EPS_START,
                 EPS_END,
                 EPS_FRAME,
                 MEMORY_CAPACITY,
                 FRAME_HISTORY_LEN,
                 seed,
                 ):
        
        self.height = height
        self.width = width
        self.channel = channel
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.eps_start = EPS_START
        self.eps_end = EPS_END
        self.eps_frame = EPS_FRAME
        self.memory_capacity = MEMORY_CAPACITY
        self.frame_history_len = FRAME_HISTORY_LEN
        self.double = DOUBLE
        self.dueling = DUELING
        self.per = PRIORITY
        self.pref = PREFERENCE
        self.auto_entropy = ENTROPY
        self.imsamp = IMPORTANTSAMPLING
        self.joint = JOINT
        self.seed = seed
        
        self.lr = 0.00025
        self.lr_p = 0.0001
        self.lr_temp = 0.0001
        self.alpha = 0.95
        self.eps = 0.01
        self.threshold = 50000

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.policy_net = Net(height, width, channel, n_actions, DUELING, PREFERENCE, seed)#.to(device)
        self.target_net = Net(height, width, channel, n_actions, DUELING, PREFERENCE, seed)#.to(device)

        if torch.cuda.device_count() > 8:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.policy_net = nn.DataParallel(self.policy_net)
            self.target_net = nn.DataParallel(self.target_net)
            self.policy_net.to(device)
            self.target_net.to(device)
        else:
            self.policy_net.to(device)
            self.target_net.to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters(),lr=self.lr,
                                       alpha=self.alpha, eps=self.eps)
        self.optimizer_p = optim.RMSprop(self.policy_net.parameters(),lr=self.
                                         lr_p, alpha=self.alpha, eps=self.eps)

        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=1.0)
        self.scheduler_p = lr_scheduler.ExponentialLR(self.optimizer_p, gamma=1.0)
        
        self.steps_done = 0
        self.loss_critic = 0.0
        self.loss_actor = 0.0
        self.loss_entropy = 0.0
        self.eps_threshold = 0.0
        self.action_distribution = 0.0
        self.q_policy = np.zeros(self.n_actions)
        self.q_target = np.zeros(self.n_actions)
        
        self.target_entropy_ratio = 0.98
        self.temperature_copy = 0.0
        
        if self.auto_entropy:
            self.target_entropy = \
                -np.log(1.0 / self.n_actions) * self.target_entropy_ratio
            self.log_temp = torch.zeros(1, requires_grad=True, device=device)
            self.temperature = self.log_temp.exp()
            self.temp_optim = optim.Adam([self.log_temp], lr=self.lr_temp)
        else:
            self.temperature = 0.01 #1.0 / max(1, self.n_actions)
        
        if (PRIORITY):
            self.memory = PER(self.memory_capacity, self.seed)
        else:
            self.memory = ReplayMemory(self.memory_capacity, self.n_obs, self.seed)

    def select_action_deterministic(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        sample = random.random()
        eps_threshold = 0.0
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                if (self.pref):
                    action_distribution, action = self.policy_net.forward(x)
                    action_idx = action.max(1)[1].view(1, 1)
                    self.action_distribution = action_distribution.cpu().numpy()
                    return action_idx
                
                return self.policy_net.forward(x).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], 
                                device=device, dtype=torch.long) 

    def select_action_ucb(self, action, episode, number_selected_action, gain):
        return np.argmax(action + gain * np.sqrt(np.log(episode + 0.1)/(number_selected_action + 0.1)))

    def select_action(self, x, i_steps):
        sample = random.random()
        
        self.eps_threshold = self.calc_eps_threshold(i_steps)
        with torch.no_grad():
            x = torch.FloatTensor(x._force().transpose(2,0,1)[None]) / 255.0
            if sample > self.eps_threshold:
                if (self.pref):
                    action_distribution, q = self.policy_net.forward(x)
                    action_idx = q.max(1)[1].view(1, 1)
                    self.action_distribution = action_distribution.cpu().numpy()
                    return action_idx
                return self.policy_net.forward(x).max(1)[1].view(1, 1)
            else:
                if (self.pref):
                    action_distribution, q = self.policy_net.forward(x)
                    action_distribution = action_distribution.squeeze(0).cpu().numpy()
                    action_distribution /= action_distribution.sum().tolist()
                    return torch.tensor([[np.random.choice(np.arange(0, self.n_actions),
                                        p=action_distribution)]], device=device, dtype=torch.long)
                return torch.tensor([[random.randrange(self.n_actions)]], 
                                    device=device, dtype=torch.long)

    def calc_eps_threshold(self, i_steps):
        if (i_steps <= self.threshold):
            return self.eps_start
        else:
            fraction = min((i_steps - self.threshold) / self.eps_frame, 1.0)
            return self.eps_start + fraction * (self.eps_end - self.eps_start)

    def append_sample(self, state, action, reward, next_state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        
        if (self.pref):
            _, q_policy_temp = self.policy_net.forward(state_tensor)
            q_policy_temp = q_policy_temp[action]
            _, next_q_target_temp = self.target_net.forward(next_state_tensor)
        else:
            q_policy_temp = self.policy_net.forward(state_tensor).squeeze(0)[action]
            next_q_target_temp = self.target_net.forward(next_state_tensor)
            
        loss_temp = torch.abs(torch.from_numpy(np.array(reward)).to(device) + 
                              torch.max(next_q_target_temp) * self.gamma - q_policy_temp)
        state_roll, next_state_roll = self.preprocess(state, next_state)
        transition = np.hstack((state_roll, action, reward, next_state_roll))
        self.memory.add(loss_temp.detach().cpu().numpy(), transition)
        
        if (len(self.memory) > 50000):
            print(len(self.memory))
        if (len(self.memory) == self.memory.capacity):
            print('Memory pool is full filled.')

    def preprocess(self, state, next_state):
        state = state.reshape((-1))
        next_state = next_state.reshape((-1))
        return state, next_state

    def optimize_q_branch(self, state, action, reward, next_state):
        if (self.per):
            transitions, idxs, is_weight = self.memory.sample(self.batch_size)
        else:
            state_batch, action_batch, reward_batch, next_state_batch = self.memory.sample(self.batch_size)

        state_batch = torch.FloatTensor(state_batch).to(device)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(reward_batch).to(device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)

        state_batch = state_batch.view(-1, self.channel, self.height, self.width) /255
        next_state_batch = next_state_batch.view(-1, self.channel, self.height, self.width) /255

        if (not self.pref):
            q_policy = self.policy_net.forward(state_batch).gather(1, action_batch)
        else:
            action_distribution, q_policy = self.policy_net.forward(state_batch)
            q_policy = q_policy.gather(1, action_batch)

        next_q_target = torch.zeros(self.batch_size, device=device)
        if (self.double):
            if (self.pref):
                _, next_q_policy_temp = self.policy_net.forward(next_state_batch)
                _, next_q_target_temp = self.target_net.forward(next_state_batch)
                next_q_target_temp = next_q_target_temp.detach()
            else:
                next_q_policy_temp = self.policy_net.forward(next_state_batch)
                next_q_target_temp = self.target_net.forward(next_state_batch).detach()
                
            max_action_indices = torch.argmax(next_q_policy_temp, dim=1)
            indices_batch = torch.LongTensor(np.arange(self.batch_size))#torch.fromnumpy((np.arange(self.batch_size))
            next_q_target = next_q_target_temp[indices_batch, max_action_indices]
        elif (self.pref):
            _, next_q_target = self.target_net.forward(next_state_batch)
            next_q_target = next_q_target.max(1)[0].detach()
        else:
            next_q_target = self.target_net.forward(next_state_batch).max(1)[0].detach()

        # Compute the expected Q values
        q_target = (next_q_target * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()

        if (self.per):
            loss_batch = abs(q_policy.squeeze(1)  - q_target).detach().cpu().numpy()
            
            for i in range(self.batch_size):
                idx = idxs[i]
                self.memory.update(idx, loss_batch[i])

            if (self.imsamp):
                is_weight = torch.FloatTensor(is_weight.reshape(-1,1)).to(device)
            else:
                is_weight = 1.0

        else:
            is_weight = 1.0

        #critic loss
        loss_critic = criterion(q_policy * is_weight, q_target.unsqueeze(1) * is_weight)
        if torch.isnan(loss_critic):
            print('q loss is nan.')
        self.loss_critic = loss_critic.detach().cpu().numpy()

        self.optimizer.zero_grad()
        if (self.pref and self.joint):
            loss_policy = self.policy_gradient(state, action, reward, next_state)
            loss_total = loss_critic + loss_policy
            loss_total.backward()
        elif (not self.pref and self.joint):
            print('Mentor is false, but together is true.')
            loss_critic.backward()
        else:
            loss_critic.backward()

        clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()

    def policy_gradient(self, state, action, reward, next_state):
        state_tensor = torch.FloatTensor(state._force().transpose(2,0,1)[None]).to(device) / 255.0
        next_state_tensor = torch.FloatTensor(state._force().transpose(2,0,1)[None]).to(device) / 255.0
        action_distribution_policy, q_policy = self.policy_net.forward(state_tensor)
        action_distribution_target, _ = self.target_net.forward(state_tensor)
        _, next_q_target_temp = self.target_net.forward(next_state_tensor)

        q_target = torch.from_numpy(np.array(reward)).to(device) +\
                   next_q_target_temp * self.gamma

        self.ac_dis_policy = action_distribution_policy.squeeze(0).cpu().detach().numpy()
        self.ac_dis_target = action_distribution_target.squeeze(0).cpu().detach().numpy()
        self.q_policy = q_policy.squeeze(0).cpu().detach().numpy()
        self.q_target = q_target.squeeze(0).cpu().detach().numpy()

        action_distribution_policy = action_distribution_policy.squeeze(0)
        action_distribution_target = action_distribution_target.squeeze(0)
        action_prob_policy = Categorical(action_distribution_policy)

        q_policy = q_policy.squeeze(0)
        q_target = q_target.squeeze(0)

        state_value = torch.matmul(action_distribution_target, q_target)
        advantage_function = (q_target - state_value).detach()

        loss_policy = - torch.matmul(action_prob_policy.probs, advantage_function)
        if torch.isnan(loss_policy):
            print('policy loss is nan.')
        loss_entropy =  - action_prob_policy.entropy().mean()
        if torch.isnan(loss_entropy):
            print('entropy loss is nan.')

        self.loss_policy = loss_policy.detach().cpu().numpy()
        self.loss_entropy = loss_entropy.detach().cpu().numpy()

        if (self.temperature > 0):
            loss_policy = loss_policy + loss_entropy * self.temperature
        else:
            loss_policy = loss_policy
        
        if self.auto_entropy:
            self.optimize_entropy_parameter(loss_entropy.detach())
        
        if self.auto_entropy:
            self.temperature_copy = self.temperature.detach().squeeze().cpu().numpy()
        else:
            self.temperature_copy = self.temperature

        return loss_policy
        
    def optimize_preference_branch(self, state, action, reward, next_state):
        if (not self.pref or self.joint):
            return
        loss_policy = self.policy_gradient(state, action, reward, next_state)
        self.optimizer_p.zero_grad()
        loss_policy.backward()
        clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer_p.step()
        
    def optimize_entropy_parameter(self, entropy):
        temp_loss = -torch.mean(self.log_temp * (self.target_entropy + entropy))
        self.temp_optim.zero_grad()
        temp_loss.backward()
        self.temp_optim.step()
        self.temperature = self.log_temp.detach().exp()        
        
