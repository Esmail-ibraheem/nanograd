from typing import Tuple
import time
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import trange
import numpy as np

ENVIRONMENT_NAME = 'CartPole-v1'

# *** hyperparameters ***

BATCH_SIZE = 256
ENTROPY_SCALE = 0.0005
REPLAY_BUFFER_SIZE = 2000
PPO_EPSILON = 0.2
HIDDEN_UNITS = 32
LEARNING_RATE = 1e-2
TRAIN_STEPS = 5
EPISODES = 40
DISCOUNT_FACTOR = 0.99

class ActorCritic(nn.Module):
    def __init__(self, in_features, out_features, hidden_state=HIDDEN_UNITS):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(in_features, hidden_state),
            nn.Tanh(),
            nn.Linear(hidden_state, out_features),
            nn.LogSoftmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(in_features, hidden_state),
            nn.ReLU(),
            nn.Linear(hidden_state, 1)
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        act = self.actor(obs)
        value = self.critic(obs)
        return act, value

def evaluate(model: ActorCritic, test_env: gym.Env) -> float:
    obs, _ = test_env.reset()
    terminated, truncated = False, False
    total_rew = 0.0
    while not terminated and not truncated:
        act = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))[0].argmax().item()
        obs, rew, terminated, truncated, _ = test_env.step(act)
        total_rew += float(rew)
    return total_rew

def run():
    env = gym.make(ENVIRONMENT_NAME, render_mode='human')
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def train_step(x: torch.Tensor, selected_action: torch.Tensor, reward: torch.Tensor, old_log_dist: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model.train()
        log_dist, value = model(x)
        action_mask = (selected_action.unsqueeze(1) == torch.arange(log_dist.shape[1]).unsqueeze(0)).float()

        advantage = reward.unsqueeze(1) - value
        masked_advantage = action_mask * advantage.detach()

        ratios = (log_dist - old_log_dist).exp()
        unclipped_ratio = masked_advantage * ratios
        clipped_ratio = masked_advantage * torch.clamp(ratios, 1-PPO_EPSILON, 1+PPO_EPSILON)
        action_loss = -torch.min(unclipped_ratio, clipped_ratio).sum(-1).mean()
  
        entropy_loss = (log_dist.exp() * log_dist).sum(-1).mean()
        critic_loss = advantage.pow(2).mean()

        optimizer.zero_grad()
        (action_loss + entropy_loss * ENTROPY_SCALE + critic_loss).backward()
        optimizer.step()

        return action_loss.item(), entropy_loss.item(), critic_loss.item()

    def get_action(obs: torch.Tensor) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            log_dist, _ = model(obs)
            return Categorical(logits=log_dist).sample()

    st, steps = time.perf_counter(), 0
    Xn, An, Rn = [], [], []
    for episode_number in (t := trange(EPISODES)):
        obs, _ = env.reset()
        rews, terminated, truncated = [], False, False
        while not terminated and not truncated:
            env.render()
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            act = get_action(obs_tensor).item()

            Xn.append(np.copy(obs))
            An.append(act)

            obs, rew, terminated, truncated, _ = env.step(act)
            rews.append(float(rew))
        steps += len(rews)

        discounts = np.power(DISCOUNT_FACTOR, np.arange(len(rews)))
        Rn += [np.sum(rews[i:] * discounts[:len(rews) - i]) for i in range(len(rews))]

        Xn, An, Rn = Xn[-REPLAY_BUFFER_SIZE:], An[-REPLAY_BUFFER_SIZE:], Rn[-REPLAY_BUFFER_SIZE:]
        X = torch.tensor(Xn, dtype=torch.float32)
        A = torch.tensor(An, dtype=torch.int64)
        R = torch.tensor(Rn, dtype=torch.float32)

        old_log_dist = model(X)[0].detach()
        for i in range(TRAIN_STEPS):
            indices = torch.randint(0, X.shape[0], (BATCH_SIZE,))
            action_loss, entropy_loss, critic_loss = train_step(X[indices], A[indices], R[indices], old_log_dist[indices])
        t.set_description(f"sz: {len(Xn):5d} steps/s: {steps / (time.perf_counter() - st):7.2f} action_loss: {action_loss:7.3f} entropy_loss: {entropy_loss:7.3f} critic_loss: {critic_loss:8.3f} reward: {sum(rews):6.2f}")

    test_rew = evaluate(model, gym.make(ENVIRONMENT_NAME, render_mode='human'))
    print(f"test reward: {test_rew}")

if __name__ == "__main__":
    run()
