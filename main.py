#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 18:10:36 2023

@author: oscar
"""
import os
import yaml
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt

import torch

from DRL import DQN
from env_wrapper.atari_wrapper import make_atari, wrap_deepmind

def plot_animation_figure():
    plt.figure()
    plt.clf() 
    plt.title(env_name + ' ' + algo + ' ' + str(agent.lr) + '_' + str(agent.lr_p) + '_' + str(agent.lr_temp))
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(train_durations, reward_list)
    plt.plot(train_durations, reward_mean_list)

    plt.pause(0.001) 

def smooth(train_duration_list, reward_list, smooth_horizon):
    mean1 = np.mean(train_duration_list[-min(len(train_duration_list), smooth_horizon):])
    train_durations_mean_list.append(mean1)
    mean3 = np.mean(reward_list[-min(len(reward_list), smooth_horizon):])
    reward_mean_list.append(mean3)

def preprocess(s, a, r, s_):
    state = np.squeeze(s._force().transpose(2,0,1)[None])
    state = state.reshape((-1))
    next_state = np.squeeze(s_._force().transpose(2,0,1)[None])
    next_state = next_state.reshape((-1))
    action = a.cpu().numpy().squeeze().astype(np.int32)
    reward = np.float32(r)
    
    return state, action, reward, next_state

def interaction():
    reward_total = 0.0 
    error = 0.0 
    num_steps = 0
    train_steps = 0
    epoc = 1
    cumulate_flag = False

    s = env.reset()

    if UCB:
        ucb_gain = 2.0
        number_selected_action = np.zeros(n_actions)

    for t in tqdm(range(1, MAX_NUM_STEPS+1), ascii=True):
        
        if VISUALIZATION:
            env.render()

        if num_steps > MAX_NUM_STEPS:
            print('Done')
            break

        # Select and perform an action
        if UCB:
            q = agent.policy_net.forward(torch.FloatTensor(s._force().transpose(2,0,1)[None]) / 255.0).cpu().detach().numpy()
            a = agent.select_action_ucb(q, epoc, number_selected_action, ucb_gain)
            number_selected_action[a] += 1
            a = torch.tensor(a).reshape(1,1)
        else:
            a = agent.select_action(s, num_steps)

        s_, r, done, _ = env.step(a.item())

        #preprocess the state
        state, action, reward, next_state = preprocess(s, a, r, s_)

        # Store the transition in memory
        if (PER):
            agent.append_sample(state, action, reward, next_state) 
        else:
            index = agent.memory.store_frame(state)
            agent.memory.store_state(index, action, reward)

        if (num_steps > THRESHOLD and num_steps % Q_FREQ == 0):
            agent.optimize_q_branch(s, a, reward, s_)
        if (agent.pref and not agent.joint and num_steps % PREFERENCE_FREQ == 0):
            agent.optimize_preference_branch(s, a, reward, s_)

        if (num_steps > THRESHOLD):
            reward_total += reward
            error += agent.loss_critic
            train_steps += 1

            if num_steps % TARGET_UPDATE == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

            if env.was_real_done:
                
                if (cumulate_flag):
                    reward_list.append(reward_total)
                    
                agent.scheduler.step()
                
                if (agent.pref and not agent.joint):
                    agent.scheduler_p.step()

                if num_steps % PRINT_INTERVAL == 0:
                    print('\n',
                          '\n',
                          'Epoc:', epoc,
                          'Step:', num_steps,
                          'ExpR:', agent.eps_threshold,
                          'R:', reward_total,
                          'Freq:', PREFERENCE_FREQ,
                          'Lr:', agent.scheduler.get_last_lr(),
                          'Lr_p:', agent.scheduler_p.get_last_lr(),
                          'Lr_temp:', agent.lr_temp,
                          'current_temp:', agent.temperature_copy,
                          'Algo:', algo,
                          'seed:', seed,
                          'Env:', env_name,
                          '\n')

                if (cumulate_flag):
                    reward_mean_list.append(np.mean(reward_list[-100:]))
                    train_durations.append(num_steps)
                    
                reward_total = 0
                error = 0
                epoc += 1
                cumulate_flag = True
                
        # Move to the next state and accumulate the reward along trajectory
        s = s_
        num_steps += 1

        if done:
            s = env.reset()

        if (num_steps % SAVE_INTERVAL == 0):
            q_policy_list.append(agent.q_policy)
            q_target_list.append(agent.q_target)
            state_list.append(s._force())

        if (num_steps % SAVE_INTERVAL == 0):
            np.save(os.path.join('data', 'reward_memo'+str(MEMORY_CAPACITY)+
                                 '_step'+str(MAX_NUM_STEPS)+'_seed'+ str(seed) +\
                                 '_' + str(agent.lr)+'_'+env_name+'_' + algo),
                                 [reward_mean_list], allow_pickle=True, fix_imports=True)
            
            np.save(os.path.join('data', 'steps_memo'+str(MEMORY_CAPACITY)+
                                 '_step'+str(MAX_NUM_STEPS)+'_seed'+ str(seed) +\
                                 '_' + str(agent.lr)+'_'+env_name+'_' + algo),
                                 [train_durations], allow_pickle=True, fix_imports=True)        
            
            torch.save(agent.policy_net.state_dict(), os.path.join('network',
                      algo +'_memo' + str(MEMORY_CAPACITY) + '_step' + str(MAX_NUM_STEPS)+
                      '_seed' + str(seed) + '_' + str(agent.lr) + '_' + env_name + '_policynet.pkl'))
            
            plot_animation_figure()


    print('Complete')
    env.close()
    plt.ioff()
    plt.show()

if __name__ == "__main__":

    plt.ion()
    
    # Create folders
    if not os.path.exists("./data"):
        os.makedirs("./data")
        
    if not os.path.exists("./network"):
        os.makedirs("./network")
    
    # Load the config
    path = os.getcwd()
    yaml_path = os.path.join(path, 'config.yaml')
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # DQN params
    algo = 'PGDQN' # pls check config.yaml file to check other algos
    algo_param = config[algo]
    DOUBLE = algo_param['DOUBLE']
    DUELING = algo_param['DUELING']
    UCB = algo_param['UCB']
    PER = algo_param['PER']
    IMPORTANT_SAMPLING = algo_param['IMPORTANT_SAMPLING'] 
    PREFERENCE = algo_param['PREFERENCE']
    ENTROPY = algo_param['ENTROPY']
    JOINT_OPT = algo_param['JOINT_OPT']
    
    # Training Params
    THRESHOLD = config['THRESHOLD']
    TARGET_UPDATE = config['TARGET_UPDATE']
    BATCH_SIZE = config['BATCH_SIZE']
    GAMMA = config['GAMMA']
    MEMORY_CAPACITY = config['MEMORY_CAPACITY']
    Q_FREQ = config['Q_FREQ']
    PREFERENCE_FREQ = config['PREFERENCE_FREQ']
    EPS_START = config['EPS_START']
    EPS_END = config['EPS_END']
    EPS_DECAY = config['EPS_DECAY']
    FRAME_HISTORY_LEN = config['FRAME_HISTORY_LEN']
    MAX_NUM_STEPS = config['MAX_NUM_STEPS']
    PRINT_INTERVAL = config['PRINT_INTERVAL']
    SAVE_INTERVAL = config['SAVE_INTERVAL']
    VISUALIZATION = config['VISUALIZATION']
    
    # Environment
    env = make_atari('PongNoFrameskip-v4')
    env = wrap_deepmind(env, clip_rewards=False, frame_stack=True)
    env_name = 'PongNoFrameskip-v4'
    
    # State and Action Space
    obs = env.reset()
    img_h, img_w, channel = env.observation_space.shape
    n_obs = img_h * img_w * channel
    n_actions = env.action_space.n

    train_durations_list = []
    step_list_list = []
    reward_list_list = []
    legend_bar = []
    
    for seed in range(525,526):
        env.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        legend_bar.append('seed'+str(seed))

        agent = DQN(img_h, img_w, channel, n_obs, n_actions,
                    DOUBLE, DUELING, PER, IMPORTANT_SAMPLING, PREFERENCE,
                    ENTROPY, JOINT_OPT, BATCH_SIZE, GAMMA, EPS_START, EPS_END,
                    EPS_DECAY, MEMORY_CAPACITY, FRAME_HISTORY_LEN, seed)
        
        train_durations = []
        train_durations_mean_list = []
        reward_list = []
        reward_mean_list = []
        q_policy_list = []
        q_target_list = []
        state_list = []
        
        print('The object is:', algo, '|Preference:', agent.pref,
              '|Auto Entropy:', agent.auto_entropy, '|Priority:', agent.per,
              '|Double:', agent.double, '|Dueling:', agent.dueling, '|UCB', UCB,
              '|Joint Optimization:', agent.joint, '|Seed:', seed)

        interaction()
        train_durations_list.append(train_durations)
        reward_list_list.append(reward_mean_list)

        np.save(os.path.join('data', 'reward_memo'+str(MEMORY_CAPACITY)+
                             '_step'+str(MAX_NUM_STEPS)+'_seed'+ str(seed) +\
                             '_' + str(agent.lr)+'_'+env_name+'_' + algo),
                             [reward_mean_list], allow_pickle=True, fix_imports=True)
        
        np.save(os.path.join('data', 'steps_memo'+str(MEMORY_CAPACITY)+
                             '_step'+str(MAX_NUM_STEPS)+'_seed'+ str(seed) +\
                             '_' + str(agent.lr)+'_'+env_name+'_' + algo),
                             [train_durations], allow_pickle=True, fix_imports=True)        
        
        torch.save(agent.policy_net.state_dict(), os.path.join('network',
                  algo +'_memo' + str(MEMORY_CAPACITY) + '_step' + str(MAX_NUM_STEPS)+
                  '_seed' + str(seed) + '_' + str(agent.lr) + '_' + env_name + '_policynet.pkl'))
