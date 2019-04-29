import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Parameters:
    def __init__(self):
        self.BUFFER_SIZE = int(1e6)
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.TAU = 2e-1
        self.LR_ACTOR = 1e-4
        self.LR_CRITIC = 1e-3
        self.WEIGHT_DECAY = 0
        self.LEARN_EVERY = 1
        self.LEARN_NUM = 1
        self.OU_SIGMA = 0.2
        self.OU_THETA = 0.15
        self.ENABLE_EPSILON = False
        self.EPSILON = 1.0
        self.EPSILON_DECAY = 1e-6


class OUNoise:
    def __init__(self, params, size, seed, mu=0.):
        self.mu = mu * np.ones(size)
        self.theta = params.OU_THETA
        self.sigma = params.OU_SIGMA
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx

        return self.state


class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)

    return (-lim, lim)


class ActorNN(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=512, fc2_units=256):
        super(ActorNN, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.fc2(x))

        return F.tanh(self.fc3(x))


class CriticNN(nn.Module):
    def __init__(self, state_size, action_size, seed, fcs1_units=512, fc2_units=256):
        super(CriticNN, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        xs = F.relu(self.bn1(self.fcs1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))

        return self.fc3(x)


class Actor(object):
    def __init__(self, params, state_size, action_size, memory, random_seed):
        self.params = params
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.epsilon = self.params.EPSILON

        # actor networks
        self.local = ActorNN(state_size, action_size, random_seed).to(device)
        self.target = ActorNN(state_size, action_size, random_seed).to(device)
        self.optimizer = optim.Adam(self.local.parameters(), lr=self.params.LR_ACTOR)

        self.noise = OUNoise(self.params, action_size, random_seed)

        self.memory = memory

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.local.eval()
        with torch.no_grad():
            action = self.local(state).cpu().data.numpy()
        self.local.train()
        if add_noise:
            #action += self.epsilon * self.noise.sample()
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        if self.params.ENABLE_EPSILON:
            self.epsilon -= self.params.EPSILON_DECAY

        self.noise.reset()


class Critic(object):
    def __init__(self, params, state_size, action_size, memory, random_seed):
        self.params = params
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.epsilon = self.params.EPSILON

        # critic networks
        self.local = CriticNN(state_size, action_size, random_seed).to(device)
        self.target = CriticNN(state_size, action_size, random_seed).to(device)
        self.optimizer = optim.Adam(self.local.parameters(), lr=self.params.LR_CRITIC, weight_decay=self.params.WEIGHT_DECAY)

        self.memory = memory

    def learn(self, actor, experiences):
        states, actions, rewards, next_states, dones = experiences

        # critic update
        # step 1:
        actions_next = actor.target(next_states)
        Q_targets_next = self.target(next_states, actions_next)
        # step 2:
        Q_targets = rewards + (self.params.GAMMA * Q_targets_next * (1 - dones))
        # step 3:
        Q_expected = self.local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # step 4:
        self.optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.optimizer.step()

        # actor update
        # step 5:
        actions_pred = actor.local(states)
        actor_loss = -self.local(states, actions_pred).mean()
        # step 6:
        actor.optimizer.zero_grad()
        actor_loss.backward()
        actor.optimizer.step()

        # step 7:
        self.soft_update(self.local, self.target, self.params.TAU)
        self.soft_update(actor.local, actor.target, self.params.TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class AgentManager:
    def __init__(self, params, state_size, action_size, num_agents, random_seed):
        self.params = params
        self.num_agents = num_agents

        self.memory = ReplayBuffer(action_size, self.params.BUFFER_SIZE, self.params.BATCH_SIZE, random_seed)

        self.actors = [Actor(self.params, state_size, action_size, self.memory, random_seed) for _ in range(num_agents)]
        self.critic = Critic(self.params, state_size, action_size, self.memory, random_seed)

    def load_models(self):
        for i, actor in enumerate(self.actors):
            actor.local.load_state_dict(torch.load("actor_model_{}.pth".format(i+1)))

    def reset(self):
        for actor in self.actors:
            actor.reset()

    def act(self, states, add_noise=True):
        actions = [actor.act(np.expand_dims(states, axis=0), add_noise) for actor, states in zip(self.actors, states)]

        return actions

    def step(self, states, actions, rewards, next_states, dones, timestep):
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        for actor in self.actors:
            if len(self.memory) > self.params.BATCH_SIZE and timestep % self.params.LEARN_EVERY == 0:
                for _ in range(self.params.LEARN_NUM):
                    experiences = self.memory.sample()

                    self.critic.learn(actor, experiences)