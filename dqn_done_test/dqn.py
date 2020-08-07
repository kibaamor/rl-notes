#!/usr/bin/python
# coding=utf-8
import ast
import os
import argparse
import pprint
import random
from collections import namedtuple, deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory:

    def __init__(self, cap):
        self.buffer = deque(maxlen=cap)

    def __len__(self):
        return len(self.buffer)

    def put(self, trans):
        self.buffer.append(trans)

    def sample(self, num):
        batch = random.sample(self.buffer, num)
        return Transition(*(zip(*batch)))


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes)-1):
        act = activation if i < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*layers)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str)
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--epochs', type=int, default=10000)

    parser.add_argument('--hidden-size', type=int, default=32)
    parser.add_argument('--layer-num', type=int, default=2)

    parser.add_argument('--buffer-size', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=128)

    parser.add_argument('--update-after', type=int, default=200)
    parser.add_argument('--update-every', type=int, default=8)
    parser.add_argument('--sync-every', type=int, default=20)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)

    parser.add_argument('--e-greed', type=float, default=0.1)
    parser.add_argument('--e-greed-dec', type=float, default=1e-6)
    parser.add_argument('--e-greed-min', type=float, default=0.01)
    parser.add_argument('--use-dbqn', type=ast.literal_eval, default=False)
    parser.add_argument('--use-done', type=ast.literal_eval, default=True)

    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_known_args()[0]
    return args


class Agent:
    def __init__(self, env_fn, hidden_size=128, layer_num=1,
                 buffer_size=int(1e6), batch_size=100,
                 update_after=200, update_every=8, sync_every=20,
                 lr=1e-3, gamma=0.99, e_greed=0.1, e_greed_dec=1e-6, e_greed_min=0.01,
                 use_dbqn=False, use_done=True, device='cpu'):

        env = env_fn()
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n

        self.model = self.build_model(hidden_size, layer_num).to(device)
        self.target_model = self.build_model(hidden_size, layer_num).to(device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

        self.gamma = gamma
        self.e_greed = e_greed
        self.e_greed_dec = e_greed_dec
        self.e_greed_min = e_greed_min

        self.rpm = ReplayMemory(buffer_size)
        self.batch_size = batch_size
        self.update_after = update_after
        self.update_every = update_every
        self.update_count = 0
        self.sync_every = sync_every
        self.sync_count = 0

        self.use_dbqn = use_dbqn
        self.use_done = use_done
        self.device = device

    def build_model(self, hidden_size, layer_num):
        hidden_sizes = [self.obs_dim] + \
            [hidden_size] * layer_num + [self.act_dim]
        return mlp(hidden_sizes, nn.ReLU)

    def update_egreed(self):
        self.e_greed = max(self.e_greed_min, self.e_greed - self.e_greed_dec)

    def predict(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        q_val = self.model(obs).cpu().detach().numpy()
        q_max = np.max(q_val)
        choice_list = np.where(q_val == q_max)[0]
        return np.random.choice(choice_list)

    def sample(self, obs):
        if np.random.rand() < self.e_greed:
            return np.random.choice(self.act_dim)
        return self.predict(obs)

    def store_transition(self, trans):
        self.rpm.put(trans)

    def learn(self):
        assert self.update_after >= self.batch_size
        assert self.update_every > 0

        if len(self.rpm) < self.update_after:
            return None, None

        self.update_count += 1
        if (self.update_count - 1) % self.update_every != 0:
            return None, None

        batch = self.rpm.sample(self.batch_size)
        s0 = torch.FloatTensor(batch.state).to(self.device)
        a0 = torch.LongTensor(batch.action).to(self.device).unsqueeze(1)
        r1 = torch.FloatTensor(batch.reward).to(self.device)
        s1 = torch.FloatTensor(batch.next_state).to(self.device)
        d1 = torch.LongTensor(batch.done).to(self.device)

        q_pred = self.model(s0).gather(1, a0).squeeze()
        with torch.no_grad():
            if self.use_dbqn:
                acts = self.model(s1).max(1)[1].unsqueeze(1)
                q_target = self.target_model(s1).gather(1, acts).squeeze(1)
            else:
                q_target = self.target_model(s1).max(1)[0]

            if self.use_done:
                q_target = r1 + self.gamma * (1 - d1) * q_target
            else:
                q_target = r1 + self.gamma * q_target
        loss = self.loss_func(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.sync_count += 1
        if (self.sync_count - 1) % self.sync_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item(), self.update_count

    def save(self, pt_file):
        torch.save(self.model.state_dict(), pt_file)
        print(pt_file + ' saved.')

    def load(self, pt_file):
        self.model.load_state_dict(torch.load(pt_file))
        self.target_model.load_state_dict(self.model.state_dict())
        print(pt_file + ' loaded.')


def train(env, agent, episode, writer):
    agent.update_egreed()
    obs = env.reset()
    total_reward = 0
    for t in range(10000):
        act = agent.sample(obs)
        next_obs, reward, done, _ = env.step(act)

        agent.store_transition(Transition(obs, act, reward, next_obs, done))

        loss, count = agent.learn()
        if loss is not None:
            writer.add_scalar('train/value_loss', loss, count)

        obs = next_obs
        total_reward += reward
        if done or t >= 9999:
            writer.add_scalar('train/finish_step', t + 1, global_step=episode)
            writer.add_scalar('train/total_reward',
                              total_reward, global_step=episode)
            break


def evaluate(env, agent, episode, writer, render=True):
    total_reward = 0
    obs = env.reset()
    for t in range(10000):
        act = agent.predict(obs)
        obs, reward, done, _ = env.step(act)
        total_reward += reward

        if render:
            env.render()
        if done:
            writer.add_scalar('evaluate/finish_step',
                              t + 1, global_step=episode)
            writer.add_scalar('evaluate/total_reward',
                              total_reward, global_step=episode)
            break


def main():
    args = get_args()
    pprint.pprint(args)

    def env_fn():
        return gym.make(args.env)

    agent = Agent(
        env_fn,
        hidden_size=args.hidden_size,
        layer_num=args.layer_num,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        update_after=args.update_after,
        update_every=args.update_every,
        sync_every=args.sync_every,
        lr=args.lr,
        gamma=args.gamma,
        e_greed=args.e_greed,
        e_greed_dec=args.e_greed_dec,
        e_greed_min=args.e_greed_min,
        use_dbqn=args.use_dbqn,
        use_done=args.use_done,
        device=args.device
    )

    pt_file = args.exp_name + '.pt'
    if os.path.exists(pt_file):
        agent.load(pt_file)

    env = env_fn()
    writer = SummaryWriter('./DQN/' + args.exp_name)

    for episode in range(args.epochs):
        print(f'episode: {episode}')
        train(env, agent, episode, writer)
        if episode % 10 == 9:
            evaluate(env, agent, episode, writer, False)
        # if episode % 50 == 49:
        #     agent.save(pt_file)


if __name__ == '__main__':
    main()
