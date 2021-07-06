import numpy as np
import torch
import torch.nn as nn
import gym
import copy
import argparse
import gym_Aircraft
from dppo import DPPO, Memory, train_agent
from custom_models import FcLayer, FcLayerBn, WaveNET, WaveResNET, CombinedModel, init_weights, deactivate_batchnorm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# parsing user input option
parser = argparse.ArgumentParser(description='Train Implementation')
parser.add_argument('--value_tune_epi', type=int, default=5000, help='number of episodes for tuning value net')
parser.add_argument('--step_size', type=int, default=10000, help='step size for lr scheduler')
parser.add_argument('--update_timestep', type=int, default=2000, help='time step for update ppo model')
parser.add_argument('--max_episodes', type=int, default=50000, help='number of total episodes')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weights_initialize', type=lambda s: s.lower() in ["true", 1], default=False, help='weights initialize')
parser.add_argument('--plot_option', type=lambda s: s.lower() in ["true", 1], default=False, help='plot option')
args = parser.parse_args()

if __name__ == "__main__":
    ############## Hyperparameters ##############
    succeed_coef = 8000  # maximum reward when agent avoids collision
    collide_coef = -4000  # reward when agent doesn't avoid collision
    change_cmd_penalty = -100  # reward when agent changes command values
    cmd_penalty = -0.15  # coefficient of penaly on using command
    cmd_suit_coef = -100  # coefficient of suitable command
    start_cond_coef = 100  # coefficient of condition on begining

    value_tune_epi = args.value_tune_epi  # number of episodes for value net fine tuning

    solved_reward = 7000  # stop training if avg_reward > solved_reward
    log_interval = 50  # print avg reward in the interval
    max_episodes = args.max_episodes + value_tune_epi  # max training episodes
    max_timesteps = 200  # max timesteps in one episode
    update_timestep = args.update_timestep  # update policy every n timesteps
    lr = args.lr  # learning rate
    betas = (0.9, 0.999)  # betas for adam optimizer
    gamma = 0.999  # discount factor
    K_epochs = 4  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    step_size = args.step_size  # lr scheduling step size
    random_seed = 0  # random seed
    experiment_version = "val_net"  # experiment version
    weights_initialize = args.weights_initialize  # choose weight initialize or not
    plot_option = args.plot_option
    ##############################################

    # load learned model
    model_cmd = WaveNET(FcLayer, [2, 2, 2], [40, 20, 60])
    model_radar = WaveResNET(FcLayerBn, [3, 3, 3], [300, 300, 300], out_nodes=5, in_nodes=100)
    mean_cmd = np.load("mean_cmd.npy")
    std_cmd = np.load("std_cmd.npy")
    model_cb = CombinedModel(model_radar, model_cmd, mean_cmd, std_cmd)
    model_cb.load_state_dict(torch.load("Custom_model_Combined_fin.pth"))
    model_cb.to(device)
    model_cb.apply(deactivate_batchnorm)
    actor_model = copy.deepcopy(model_cb)
    critic_model = copy.deepcopy(model_cb)
    test_model = copy.deepcopy(model_cb)
    # initialize weights if it should
    if weights_initialize:
        actor_model.apply(init_weights)
        critic_model.apply(init_weights)
    # load mean and std of trained data
    mean = np.load('mean_noised.npy')
    std = np.load('std_noised.npy')
    # set final nodes for each model(final node of critic to one and add softmax to actor)
    num_final_nodes = critic_model.model_cmd.fin_fc.out_features
    critic_model.soft_max = nn.Linear(num_final_nodes, 1)
    test_model = test_model.to(device)

    # set requires grad to false for fixing until several episodes
    for param in actor_model.parameters():
        param.requires_grad = False

    for param in test_model.parameters():
        param.requires_grad = False

    # creating environment
    env_name = "acav-v0"
    env = gym.make(env_name)
    env.env.__init__(succeed_coef, collide_coef, change_cmd_penalty,
                     cmd_penalty, start_cond_coef, cmd_suit_coef)
    # set random seed of environment
    torch.manual_seed(random_seed)
    env.seed(random_seed)

    # set replay buffer and dppo model
    memory = Memory()
    dppo = DPPO(actor_model, critic_model, lr, betas,
                gamma, K_epochs, eps_clip, step_size)
    # train agent
    rewards, total_res, test_rewards, test_total_res = train_agent(env, dppo, test_model, mean, std, memory,
                                                                   max_episodes, max_timesteps, value_tune_epi,
                                                                   update_timestep, log_interval, solved_reward,
                                                                   experiment_version, plot_option)
    # save reward
    np.savetxt("{}.csv".format(experiment_version),
               np.array(rewards), delimiter=",")
