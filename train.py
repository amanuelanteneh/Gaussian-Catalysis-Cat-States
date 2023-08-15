import math, cmath, random, functools, scipy
import numpy as np
import gym
from gym import Env
from gym import spaces
import matplotlib.pyplot as plt
from matplotlib import cm
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from sb3_contrib import RecurrentPPO
import torch
import torch.optim as optim
import json
import os
import sys
from circuit import Circuit
from utils import *
from typing import Callable

    
def make_env(dim, intialSqz, maxSteps, exp, sqzMax, phi, target,\
                      reward, rank, seed=1, loss=0) -> Callable:
    """
    Utility function for multiprocessed env.
    
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env = Circuit(dim, intialSqz, maxSteps, exp, sqzMax, phi, target, \
                      reward, 1+rank, False, loss)
       
        return env

    return _init
    
    

if __name__ == '__main__': # needed for multi proc
    
    parameters = {}
    with open("parameters.txt") as f: # read training file
        for line in f:
           (key, val) = line.split()
           parameters[key] = val
 
    print("Parameters used for training:", flush=True)
    for key in parameters:
        print(key, ":", parameters[key], flush=True)
    
    # env parameters
    dim = int(parameters['hilbert-dimension'])
    maxSteps = int(parameters['max-steps'])
    intialSqz = float(parameters['initial-sqz'])
    sqzMax = float(parameters['sqz-max'])
    exp = float(sys.argv[4])
    reward = sys.argv[3]
    target = sys.argv[2]
    phi = parameters['phi'] == 't'
    loss = float(parameters['loss'])
    
    print("\nTarget state:", target, flush=True)
    
    # agent parameters
    gamma = float(parameters['gamma'])
    totalTimesteps = int(parameters['total-timesteps'])
    n_epochs = int(parameters['n_epochs'])
    n_steps = int(parameters['n_steps'])
    clip_range = float(parameters['clip_range'])
    lr = float(parameters['lr'])
    batchSize = int(parameters['batchSize'])
    hidden1 = int(parameters['hidden1'])
    hidden2 = int(parameters['hidden2'])
    hidden3 = int(parameters['hidden3'])
    activation = parameters['activation']
    useLSTM = parameters['lstm'] == 't'
    multiProc = parameters['multi-proc'] == 't'

    if activation == 'relu':
       act = torch.nn.ReLU
    else:
       act = torch.nn.Tanh


    modelName = 'dim_'+str(dim) \
                +'_T_'+str(maxSteps)+'_init_'+parameters['initial-sqz']\
                +'_exp_'+str(exp)\
                +'_rew_'+reward+'_rmax_'+parameters['sqz-max']\
                +'_tar_'+str(target)+'_lstm_'+parameters['lstm']\
                +'_phi_'+parameters['phi']\
                +'_loss_'+parameters['loss']\
                +'_buf_'+str(n_steps)+'_epoch_'+str(n_epochs)+'_batch_'+str(batchSize)

    os.makedirs('models/', exist_ok=True)

    os.makedirs('models/' + modelName, exist_ok=True) # create folder for agent with these parameters

    logDir = "logs/"

    os.makedirs(logDir, exist_ok=True)
    
    if multiProc:
    
        cpus = int(sys.argv[1])
        n_steps = n_steps // cpus
        print(f"\nUsing multi-proccessing with {cpus} cpu cores ({cpus} environments)\n")
        print(f"n_steps passed to PPO object was changed to be different than that given in training-parameters file, now is: {n_steps}\n")
        # create env vector for parallel training
        env = SubprocVecEnv([make_env(dim, intialSqz, maxSteps, exp, sqzMax, phi, target,\
                                  reward, i, 2023, loss) for i in range(cpus)])


        # create env to plot inital state and evaluate agent
        plotEnv = Circuit(dim, intialSqz, maxSteps, exp, sqzMax, phi, target, \
                                   reward, 1982, False, loss)

        # plot the initial state
        plotEnv.render(False, 'initial', 'models/'+modelName+"/start")
        # plot the target state
        plotEnv.render(True, 'target', 'models/'+modelName+"/target")
        
        checkpoint_callback = CheckpointCallback(save_freq=max(50000 // cpus, 1), save_path="models/"+modelName, name_prefix="rl_model")
        
        timestep_callback = TimestepCallbackMulti()
        eps_callback = EpisodeCallbackMulti()
        
    else:
        print("\nNot using multi-processing")
        timestep_callback = TimestepCallback()
        eps_callback = EpisodeCallback()
        
        checkpoint_callback = CheckpointCallback(save_freq=2000, 
                                                 save_path="models/"+modelName, name_prefix="rl_model")
        
        env = Circuit(dim, intialSqz, maxSteps, exp, sqzMax, phi, target,\
                                   reward, 2023, False, loss)
        
         # plot the initial state
        env.render(False, 'initial', 'models/'+modelName+"/start")
        # plot the target state
        env.render(True, 'target', 'models/'+modelName+"/target")


    if not useLSTM: # not using LSTM layer
        # pi is neural network arch of the actor and vf is arch for the critic
        policy_kwargs = dict(activation_fn = act,
                         net_arch=dict(pi=[hidden1, hidden2, hidden3], vf=[hidden1, hidden2, hidden3]), 
                         optimizer_class = optim.Adam)

        model = PPO("MlpPolicy",
                    env,
                    gamma=gamma,
                    n_epochs=n_epochs,
                    batch_size=batchSize,
                    clip_range=clip_range,
                    n_steps=n_steps,
                    learning_rate=lr,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=logDir)

    else: # if we want the first layer of the actor and critic networks to be LSTM layers
         policy_kwargs = dict(activation_fn = act,
                         net_arch=dict(pi=[hidden1, hidden2, hidden3], vf=[hidden1, hidden2, hidden3]), 
                         lstm_hidden_size = hidden1, 
                         n_lstm_layers = 1,
                         optimizer_class = optim.Adam)

         model = RecurrentPPO("MlpLstmPolicy",
                    env,
                    gamma=gamma,
                    n_epochs=n_epochs,
                    batch_size=batchSize,
                    clip_range=clip_range,
                    n_steps=n_steps,
                    learning_rate=lr,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=logDir) 

    print("\nNeural network architecture: \n\n", model.policy)

    print("\nStarting training.", flush=True)

    if not multiProc:
        model.learn(total_timesteps=totalTimesteps, tb_log_name=modelName, callback=[timestep_callback, eps_callback, checkpoint_callback])
    else:
        model.learn(total_timesteps=totalTimesteps, tb_log_name=modelName, callback=[timestep_callback, eps_callback, checkpoint_callback])

    print("\nTraining complete.")
