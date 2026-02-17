import torch
import torch.nn as nn
import numpy as np


# --- PID Controller ---
class PIDController:
    """
    Discrete-time PID with Anti-Windup.
    Standard industrial baseline.
    """

    def __init__(self, kp, ki, kd, output_limits=(-10.0, 10.0), dt=0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_out, self.max_out = output_limits
        self.dt = dt
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, target, current):
        error = target - current

        # Proportional
        P = self.kp * error

        # Integral
        self.integral += error * self.dt
        # Simple clamping for anti-windup
        self.integral = np.clip(self.integral, self.min_out, self.max_out)
        I = self.ki * self.integral

        # Derivative
        derivative = (error - self.prev_error) / self.dt
        D = self.kd * derivative

        # Output Clamping
        output = P + I + D
        output = np.clip(output, self.min_out, self.max_out)

        self.prev_error = error
        return output


# --- PPO (Pure RL) Helper Classes ---
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.action_var = torch.full((action_dim,), 0.5)

    def act(self, state, memory):
        # state: [state_dim] -> [1, state_dim]
        state = torch.from_numpy(state).float().unsqueeze(0)

        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(0)
        dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach().numpy().flatten()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)
        dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)

        # action shape must match action_mean shape [Batch, Action_Dim]
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPOAgent:
    """
    Proximal Policy Optimization Agent.
    """

    def __init__(
        self, state_dim, action_dim, lr=0.002, gamma=0.99, K_epochs=4, eps_clip=0.2
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.memory = Memory()

    def select_action(self, state):
        return self.policy.act(state, self.memory)

    def update(self):
        # Monte Carlo estimate of rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.memory.rewards), reversed(self.memory.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # --- CRITICAL FIX START ---
        # Convert list to tensor with explicit squeezing

        # memory.states: list of [1, S]. Stack -> [N, 1, S]. Squeeze(1) -> [N, S]
        old_states = torch.stack(self.memory.states, dim=0).detach().squeeze(1)

        # memory.actions: list of [1, A]. Stack -> [N, 1, A]. Squeeze(1) -> [N, A]
        # DO NOT use .squeeze() without dim, or it will flatten [N, 1] to [N] if A=1
        old_actions = torch.stack(self.memory.actions, dim=0).detach().squeeze(1)

        # memory.logprobs: list of [1]. Stack -> [N, 1]. Squeeze() -> [N]
        old_logprobs = torch.stack(self.memory.logprobs, dim=0).detach().squeeze()
        # --- CRITICAL FIX END ---

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )
            state_values = torch.squeeze(state_values)

            ratios = torch.exp(logprobs - old_logprobs)
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards)
                - 0.01 * dist_entropy
            )

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory.clear_memory()
