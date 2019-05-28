import argparse
import json
import logging
import os
import numpy as np
import random
import time
from collections import namedtuple
from itertools import count

import gym
import ptan
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from lunar_lander import LunarLander
from memory import ReplayMemory
from models import *
from visualize import *

Transition = namedtuple('Transion', ('state', 'action', 'next_state', 'reward'))


def select_action(policy, state, epsilon):
    """
    select action folowing an epsilon greedy policy
    :param policy: policy model
    :param state: current state
    :param epsilon: prob of sampling the model
    :return: action as a long tensor
    """
    policy.eval()
    sample = random.random()
    if sample > epsilon:
        with torch.no_grad():
            return policy(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(N_ACTIONS)]], device=device, dtype=torch.long)


def update_epsilon(steps_done):
    """exponential decay"""
    if steps_done > EPS_OFFSET:
        epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * (steps_done - EPS_OFFSET) / (EPS_DECAY + EPS_OFFSET))
    else:
        epsilon = EPS_START

    return epsilon


def sample_memory(memory, device, non_blocking=False):
    # sample replay buffer
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # convert to tensors and create batches
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                  dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device,
                                                                                         non_blocking=non_blocking)
    state_batch = torch.cat(batch.state).to(device, non_blocking=non_blocking)
    action_batch = torch.cat(batch.action).to(device, non_blocking=non_blocking)
    reward_batch = torch.cat(batch.reward).to(device, non_blocking=non_blocking)

    return state_batch, action_batch, reward_batch, non_final_mask, non_final_next_states


def optimize_model(policy_net, target_net, memory, batch=None, use_double_dqn=False):
    """
    Optimize policy following the DQN objective or Double DQN objective.

    :param policy_net: policy model
    :param target_net: target policy model updated every
    :param memory: Replay memory
    :param batch: batch to use (sample memory if None)
    :param use_double_dqn: use double DQN
    :return: None
    """

    policy_net.train()
    target_net.eval()

    if len(memory) < BATCH_SIZE:
        return

    if batch is None:
        batch = sample_memory(memory, device)

    state_batch, action_batch, reward_batch, non_final_mask, non_final_next_states = batch

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    # Compute expected state action value
    if use_double_dqn:
        argmax_next_state_values = policy_net(non_final_next_states).argmax(1).detach().unsqueeze(1)
        next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1,
                                                                                     argmax_next_state_values).detach().squeeze()

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    else:
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # compute loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # optimize
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 10.)
    optimizer.step()


def preprocess_obs(obs):
    """
    create state tensor from observation
    :param obs: observation from the environment
    :return: state as torch tensor of shape [1 x *]
    """
    if RAM:
        state = torch.tensor(obs, dtype=torch.float)
    else:
        state = torch.tensor(obs.__array__())
    return state.unsqueeze(0).contiguous()


def train(env, policy_net, target_net, memory, max_frames, render=False, double_dqn=False):
    """
    train the policy model for `max_frames`frames using a replay memory and the target model.
    Update the target model every `TARGET_SYNC`frames.
    Log every `LOG_FREQ`steps.


    :param env: gym environment
    :param max_frames: number of frames
    :param render: if True, renders the environment
    :param double_dqn: use double DQN policy
    :return: None
    """
    steps_done = 0
    rewards = []
    iters = 0
    with tqdm(total=max_frames) as pbar:
        for episode in count():
            obs = env.reset()
            state = preprocess_obs(obs).to(device, non_blocking=True)
            total_reward = 0.
            elaps = time.time()
            frames = 0
            total_opt_time = 0.
            batch = None

            for t in count():

                epsilon = update_epsilon(steps_done)
                steps_done += 1
                frames += 1
                iters += 1

                action = select_action(policy_net, state, epsilon)

                if render:
                    env.render()

                obs, reward, done, info = env.step(action)

                total_reward += reward

                if not done:
                    next_state = preprocess_obs(obs).to(device, non_blocking=True)
                else:
                    next_state = None

                reward = torch.tensor([reward], device=device, dtype=torch.float)

                memory.push(state, action, next_state, reward)

                state = next_state

                if steps_done > INITIAL_MEMORY:

                    if t % PLAY_STEPS == 0:
                        opt_time = time.time()
                        optimize_model(policy_net, target_net, memory, batch=batch, use_double_dqn=double_dqn)
                        total_opt_time += time.time() - opt_time
                        batch = sample_memory(memory, device, non_blocking=True)

                    if steps_done % TARGET_SYNC == 0:
                        target_net.load_state_dict(policy_net.state_dict())

                if iters > 500:
                    pbar.update(iters)
                    iters = 0

                if done:
                    break

            rewards += [total_reward]

            if episode % LOG_FREQ == 0:
                elaps = (time.time() - elaps)
                frames_seconds = frames / elaps
                logger.info(
                    f'Total steps: {steps_done}   Episode: {episode}/{t}   Epsilon {epsilon:.3f}   Fps {frames_seconds:.3f}   '
                    f'Time ({elaps:.3f} / {total_opt_time:.3f})   Total reward: {np.mean(rewards):.2f} ({np.std(rewards):.2f})')
                rewards = []

            if steps_done > max_frames:
                break

    env.close()


def test(env, n_episodes, policy, logdir, render=True):
    """
    Test the policy for `n_episodes``
    Render saliency if `visualize_saliency` and `render`.

    :param env: gym environement
    :param n_episodes: numer of episodes to play
    :param policy: policy model
    :param logdir: directory where to log video
    :param render: render frames
    """

    policy.eval()

    # env = gym.wrappers.Monitor(env, logdir, force=True)
    for episode in range(n_episodes):
        obs = env.reset()
        state = preprocess_obs(obs)
        total_reward = 0.0
        for t in count():
            action = policy(state.to(device)).max(1)[1].view(1, 1)

            if HUMAN:
                if not _human_sets_pause:
                    action = _human_agent_action
                else:
                    pass

            if render:
                _wait_time = 0.1 if not _human_fast_forward else 0.01
                if _human_saliency:
                    t = time.time()
                    render_saliency(env, state, policy)
                    run_time = time.time() - t
                    if run_time < _wait_time:
                        time.sleep(_wait_time - run_time)
                else:
                    env.render()
                    time.sleep(_wait_time)

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = preprocess_obs(obs)
            else:
                next_state = None

            state = next_state

            if done:
                logger.info("Finished Episode {} with reward {}".format(episode, total_reward))
                break

    env.close()


def render_saliency(env, state, policy):
    """
    Visualize saliency on top of the environement frame.

    :param env:
    :param state:
    :param policy:
    """

    # get frame
    I = env.ale.getScreenRGB2().astype(np.float32)

    # compute saliency
    saliency, sign = saliency_map(state, policy, I.shape[:2])

    # red and blue masks
    blue_mask = np.ones_like(I)
    blue_mask[:, :, :] = (0, 0, 255)
    red_mask = np.ones_like(I)
    red_mask[:, :, :] = (255, 0, 0)

    # post processing + normalize
    saliency = (saliency.squeeze().data).numpy()[:, :, None]
    sign = (sign.squeeze().data).numpy()[:, :, None]

    global _saliencies
    _saliencies += [np.max(saliency)]
    if len(_saliencies) > 1000:
        _saliencies.pop(0)

    saliency = np.sqrt(saliency / np.percentile(_saliencies, 75))

    thresh = 0.0
    saliency *= 3.0
    saliency = 0.7 * (saliency.clip(thresh, 1.0) - thresh) / (1 - thresh)

    # apply masks
    I = np.where(sign > 0, saliency * red_mask + (1. - saliency) * I, saliency * blue_mask + (1. - saliency) * I)

    # render
    I = I.clip(1, 255).astype('uint8')
    env.viewer.imshow(I)


if __name__ == '__main__':
    """
    Train and/or evaluate an agen on a Gym environment. 
    
    You can play Atari yourself using the flags --render --human 
    (Use SPACE to take control, use keys A,S,D,Q,W,D to control the agent and press ENTER to display saliency) 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--root', type=str, default='logs/', help='log directory')
    parser.add_argument('--version', default='final4', help='version')
    parser.add_argument('--env', default="Pong", help='gym env')
    parser.add_argument('--env_version', default="NoFrameskip-v4", help='gym env')
    parser.add_argument('--frames', type=int, default='10000000', help='number of episodes')
    parser.add_argument('--log_freq', type=int, default='1', help='log frequency')
    parser.add_argument('--batch_size', type=int, default='32', help='batch size')
    parser.add_argument('--memory_size', type=int, default='300000', help='memory size')
    parser.add_argument('--initial_memory_size', type=int, default='100000', help='initial memory size')
    parser.add_argument('--epsilon_decay', type=int, default='1000000', help='number of steps to decrease epsilon')
    parser.add_argument('--min_epsilon', type=float, default=0.1, help='minimum epsilon')
    parser.add_argument('--play_steps', type=int, default=4, help='number of playing steps without optimization')
    parser.add_argument('--sync_freq', type=int, default='10000', help='number of episodes before each target update')
    parser.add_argument('--double', action='store_true', help='use double DQN')
    parser.add_argument('--dueling', action='store_true', help='use dueling DQN')
    parser.add_argument('--gamma', type=float, default=0.99, help='gamma parameter')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--render', action='store_true', help='render environment')
    parser.add_argument('--evaluate', action='store_true', help='evalaute trained model')
    parser.add_argument('--human', action='store_true',
                        help='play yourself (press SPACE bar to give control to the agent, press S to display saliency)')

    opt = parser.parse_args()

    env_id = f"{opt.env}{opt.env_version}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(opt.root):
        os.makedirs(opt.root)

    with open(f"{opt.root}/{env_id}.json", "w") as f:
        json = json.dumps(vars(opt))
        f.write(json)

    # define unique id
    model_id = 'DDQN' if opt.dueling else 'DQN'
    if opt.double:
        model_id = 'Double' + model_id
    run_id = f"{model_id}-{env_id}-version-{opt.version}"

    # logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-4s %(levelname)-4s %(message)s',
                        datefmt='%m-%d %H:%M',
                        handlers=[logging.FileHandler(f"{opt.root}/{run_id}.log"),
                                  logging.StreamHandler()])
    logger = logging.getLogger(run_id)

    # parameters
    seed = opt.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    BATCH_SIZE = opt.batch_size * opt.play_steps
    GAMMA = opt.gamma
    EPS_START = 1
    EPS_END = opt.min_epsilon
    EPS_DECAY = opt.epsilon_decay
    EPS_OFFSET = opt.initial_memory_size
    TARGET_SYNC = opt.sync_freq
    LOG_FREQ = opt.log_freq
    RENDER = opt.render
    MAX_FRAMES = opt.frames
    LR = opt.lr
    INITIAL_MEMORY = opt.initial_memory_size
    MEMORY_SIZE = opt.memory_size
    PLAY_STEPS = opt.play_steps
    HUMAN = opt.human

    # create environment
    if 'lunar' in opt.env:
        env = LunarLander()
        RAM = True
    else:
        env = gym.make(env_id)
        if not opt.evaluate:
            env = ptan.common.wrappers.wrap_dqn(env)
        else:
            env = ptan.common.wrappers.wrap_dqn(env, episodic_life=False, reward_clipping=False)
        RAM = False

    N_ACTIONS = env.action_space.n

    # human control mode and saliency rendering
    if HUMAN:
        RENDER = True

        _saliencies = []

        ACTIONS = env.get_action_meanings()

        print(ACTIONS)

        from pyglet.window import key as KEY

        # flags used to control human inputs.
        _human_agent_action = 0
        _human_wants_restart = False
        _human_sets_pause = True
        _human_fast_forward = False
        _human_saliency = False

        # store saliency maxes for normalization
        saliency_average = None


        def key_press(key, mod):
            global _human_agent_action, _human_wants_restart, _human_sets_pause, _human_saliency, _human_fast_forward
            if key == 0xff0d: _human_wants_restart = True
            if key == 32: _human_sets_pause = not _human_sets_pause
            if key == KEY.ENTER: _human_saliency = not _human_saliency
            if key == KEY.F: _human_fast_forward = not _human_fast_forward

            if key == KEY.A:  _human_agent_action = ACTIONS.index('LEFT')
            if key == KEY.D:  _human_agent_action = ACTIONS.index('RIGHT')
            if key == KEY.W:  _human_agent_action = ACTIONS.index('FIRE')
            if key == KEY.Q:  _human_agent_action = ACTIONS.index('RIGHTFIRE')
            if key == KEY.E:  _human_agent_action = ACTIONS.index('LEFTFIRE')


        def key_release(key, mod):
            global _human_agent_action
            _human_agent_action = ACTIONS.index('NOOP')


        env.render()
        env.unwrapped.viewer.window.on_key_press = key_press
        env.unwrapped.viewer.window.on_key_release = key_release

    # create models
    if RAM:
        a = random.randrange(env.action_space.n)
        s, r, done, info = env.step(a)
        N_STATE = len(s)
        MODEL = LanderDQN if 'lunar' in opt.env else RamDQN
        policy_net = MODEL(N_STATE, N_ACTIONS).to(device)
        target_net = MODEL(N_STATE, N_ACTIONS).to(device)
    else:
        MODEL = DDQN if opt.dueling else DQN
        policy_net = MODEL(n_actions=N_ACTIONS).to(device)
        target_net = MODEL(n_actions=N_ACTIONS).to(device)

    # init target model
    target_net.load_state_dict(policy_net.state_dict())

    # setup optimizer
    optimizer = optim.RMSprop(policy_net.parameters(), lr=LR)

    # initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)

    # train model and evaluate
    if not opt.evaluate:
        train(env, policy_net, target_net, memory, MAX_FRAMES, double_dqn=opt.double, render=RENDER)
        torch.save(policy_net.state_dict(), f"{opt.root}/{run_id}.pt")
    policy_net.load_state_dict(torch.load(f"{opt.root}/{run_id}.pt", map_location=device))
    logdir = f"{opt.root}/.videos/{run_id}/"
    with torch.no_grad():
        test(env, 10, policy_net, logdir, render=RENDER)
