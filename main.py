import copy
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np
import time
import logging
import argparse
from tqdm import tqdm
import torch.multiprocessing as mp

import gym
import ptan

from wrappers import *
from memory import ReplayMemory
from models import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

Transition = namedtuple('Transion',
                        ('state', 'action', 'next_state', 'reward'))


def select_action(state):
    # policy_net.eval()

    global steps_done
    global epsilon
    sample = random.random()
    if steps_done > EPS_OFFSET:
        epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * (steps_done-EPS_OFFSET) / (EPS_DECAY+EPS_OFFSET))
    else:
        epsilon = EPS_START
    steps_done += 1
    if sample > epsilon:
        with torch.no_grad():
            return policy_net(state.to(device)).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(N_ACTIONS)]], device=device, dtype=torch.long)


def optimize_model(double_dqn):
    """
    zip(*transitions) unzips the transitions into
    Transition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor)
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)    
    """

    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)


    batch = Transition(*zip(*transitions))

    actions = tuple((map(lambda a: torch.tensor([[a]], device=device), batch.action)))
    rewards = tuple((map(lambda r: torch.tensor([r], device=device), batch.reward)))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                  dtype=torch.uint8)

    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)

    # policy_net.train()
    # target_net.eval()

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    # double DQN
    if double_dqn:
        argmax_next_state_values = policy_net(non_final_next_states).argmax(1).detach().unsqueeze(1)
        next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1,
                                                                                     argmax_next_state_values).detach().squeeze()

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    else:
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def get_state(obs):
    state = np.array(obs)
    state = torch.from_numpy(state).permute(2,0,1)
    return state.unsqueeze(0)


def train(env, n_episodes, render=False, double_dqn=False):
    for episode in tqdm(range(n_episodes)):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        elaps = time.time()
        frames = 0
        total_opt_time = 0.
        for t in count():

            frames += 1
            action = select_action(state)

            if render:
                env.render()

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            reward = torch.tensor([reward], device=device)

            memory.push(state, action.to('cpu'), next_state, reward.to('cpu'))
            state = next_state

            if steps_done > INITIAL_MEMORY:
                if t % play_steps == 0:
                    opt_time = time.time()
                    optimize_model(double_dqn)
                    total_opt_time += time.time() - opt_time
                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        if episode % 1 == 0:
            elaps = (time.time() - elaps)
            frames_seconds = frames / elaps
            logger.info(
                f'Total steps: {steps_done}   Episode: {episode}/{t}   Epsilon {epsilon:.3f}   Fps {frames_seconds:.3f}   Time ({elaps:.3f} / {total_opt_time:.3f})   Total reward: {total_reward:.0f}')
    env.close()
    return


def test(env, n_episodes, policy, logdir, render=True):
    # policy.eval()

    env = gym.wrappers.Monitor(env, logdir, force=True)
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            action = policy(state.to(device)).max(1)[1].view(1, 1)

            if render:
                env.render()
                time.sleep(0.02)

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            state = next_state

            if done:
                logger.info("Finished Episode {} with reward {}".format(episode, total_reward))
                break

    env.close()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default='4', help='version')
    parser.add_argument('--env', default="PongNoFrameskip-v4", help='gym env')
    parser.add_argument('--episodes', type=int, default='1000', help='number of episodes')
    parser.add_argument('--batch_size', type=int, default='32', help='batch size')
    parser.add_argument('--memory_size', type=int, default='100000', help='memory size')
    parser.add_argument('--initial_memory_size', type=int, default='10000', help='initial memory size')
    parser.add_argument('--epsilon_decay', type=int, default='100000', help='number of steps to decrease epsilon')
    parser.add_argument('--min_epsilon', type=float, default=0.02, help='minimum epsilon')
    parser.add_argument('--play_steps', type=int, default=1, help='number of playing steps without optimization')
    parser.add_argument('--target_update_freq', type=int, default='1000',
                        help='number of episodes before each target update')
    parser.add_argument('--double', action='store_true', help='use double DQN')
    parser.add_argument('--gamma', type=float, default=0.99, help='gamma parameter')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    opt = parser.parse_args()

    run_id = f"{'DoubleDQN' if opt.double else 'DQN'}-{opt.env}-version-{opt.version}-{opt.batch_size}-{opt.gamma}-{opt.lr}-{opt.play_steps}-{opt.memory_size}"

    # logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-4s %(levelname)-4s %(message)s',
                        datefmt='%m-%d %H:%M',
                        handlers=[logging.FileHandler(f"logs/{run_id}.log"),
                                  logging.StreamHandler()])
    logger = logging.getLogger(run_id)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #torch.manual_seed(42)
    #np.random.seed(42)

    # hyperparameters
    BATCH_SIZE = opt.batch_size * opt.play_steps
    GAMMA = opt.gamma
    EPS_START = 1
    EPS_END = opt.min_epsilon
    EPS_DECAY = opt.epsilon_decay
    EPS_OFFSET = opt.initial_memory_size
    TARGET_UPDATE = opt.target_update_freq
    RENDER = False
    lr = opt.lr
    INITIAL_MEMORY = opt.initial_memory_size
    MEMORY_SIZE = opt.memory_size
    play_steps = opt.play_steps

    # create environment
    env = gym.make(opt.env)
    env = make_env(env)

    N_ACTIONS = env.action_space.n

    # create networks
    policy_net = DQNbn(n_actions=N_ACTIONS).to(device)
    target_net = DQNbn(n_actions=N_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    steps_done = 0
    epsilon = 1.0


    # initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)

    # train model
    train(env, opt.episodes, double_dqn=opt.double)
    torch.save(policy_net, f"logs/{run_id}.pt")
    policy_net = torch.load(f"logs/{run_id}.pt")
    logdir = f".videos/{run_id}/"
    test(env, 1, policy_net, logdir, render=False)

