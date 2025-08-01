# ml_agent_final.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size, self.ptr, self.size = max_size, 0, 0
        self.state = np.zeros((max_size, state_dim)); self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim)); self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state; self.action[self.ptr] = action; self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward; self.not_done[self.ptr] = 1. - done
        self.ptr = (self.ptr + 1) % self.max_size; self.size = min(self.size + 1, self.max_size)
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (torch.FloatTensor(self.state[ind]).to(device), torch.FloatTensor(self.action[ind]).to(device),
                torch.FloatTensor(self.next_state[ind]).to(device), torch.FloatTensor(self.reward[ind]).to(device),
                torch.FloatTensor(self.not_done[ind]).to(device))

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__(); self.l1 = nn.Linear(state_dim, 256); self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim); self.max_action = max_action
    def forward(self, state):
        a = F.relu(self.l1(state)); a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__(); self.l1 = nn.Linear(state_dim + action_dim, 256); self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1); self.l4 = nn.Linear(state_dim + action_dim, 256); self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)
    def forward(self, state, action):
        sa = torch.cat([state, action], 1); q1 = F.relu(self.l1(sa)); q1 = F.relu(self.l2(q1)); q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa)); q2 = F.relu(self.l5(q2)); q2 = self.l6(q2)
        return q1, q2
    def Q1(self, state, action):
        sa = torch.cat([state, action], 1); q1 = F.relu(self.l1(sa)); q1 = F.relu(self.l2(q1)); q1 = self.l3(q1)
        return q1

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.max_action = max_action
        self.total_it = 0
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    def train(self, replay_buffer, batch_size=256, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.total_it += 1; state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        with torch.no_grad():
            noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * discount * target_Q
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad(); critic_loss.backward(); self.critic_optimizer.step()
        if self.total_it % policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.actor.state_dict(), filename + "_actor")
    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic")); self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor")); self.actor_target = copy.deepcopy(self.actor)