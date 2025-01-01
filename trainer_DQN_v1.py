import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tianshou.policy import DQNPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

from training_environment import ForexTradingEnv
from data import get_forex_data

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),  # Add batch normalization
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.value_layer = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.advantage_layer = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )
    def forward(self, obs, state=None, info=None):
        """Forward pass for Dueling DQN with optional state and info."""
        if isinstance(obs, np.ndarray):  # Check if obs is a numpy array
            obs = torch.tensor(obs, dtype=torch.float32)  # Convert to tensor
            if torch.cuda.is_available():
                obs = obs.cuda()  # Move to GPU if available
        features = self.feature_layer(obs)
        value = self.value_layer(features)
        advantages = self.advantage_layer(features)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        # Return q_values and state (state remains unchanged here)
        return q_values, state

if __name__ == "__main__":
    # Step 1: Prepare the Dataset and Environment
    data = get_forex_data()
    split_index = int(0.8 * len(data))
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    train_envs = DummyVectorEnv([lambda: ForexTradingEnv(train_data)] * 5)  # Multiple envs for parallel training
    test_envs = DummyVectorEnv([lambda: ForexTradingEnv(test_data)] * 2)
    sample_env =ForexTradingEnv(test_data)
    # Step 2: Setup DQNPolicy with DuelingDQN
    input_dim = sample_env.observation_space.shape[0]
    output_dim = sample_env.action_space.n
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = DuelingDQN(input_dim, output_dim).to(device)
    optim = optim.Adam(net.parameters(), lr=1e-4)
    policy = DQNPolicy(
        model=net,
        optim=optim,
        action_space=sample_env.action_space,
        discount_factor=0.99,
        estimation_step=3,
        target_update_freq=500,
    )
    # restore the saved policy state if there is any
    policy_path = "policies/dueling_dqn_v1.pth"
    try:
        policy.load_state_dict(torch.load(policy_path))
        print(f"Policy loaded from {policy_path}")
    except FileNotFoundError:
        print(f"No policy found at {policy_path}")

    # Step 3: Setup Replay Buffer and Collectors
    buffer = VectorReplayBuffer(100000, buffer_num=5)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=False)

    # Step 4: TensorBoard Logger
    writer = SummaryWriter("./logs")
    logger = TensorboardLogger(writer)

    # Step 5: Define Training and Testing Functions
    def train_fn(epoch, env_step):
        # Linear decay of epsilon
        eps = max(0.1, 1.0 - epoch / 100)
        policy.set_eps(eps)

    def test_fn(epoch, env_step):
        policy.set_eps(0.05)

    def stop_fn(mean_reward):
        # Early stopping condition
        return mean_reward >= 1000

    # Step 6: Train the Model
    trainer = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=100,  #  Increase epochs for meaningful learning
        step_per_epoch=10000,
        step_per_collect=10,
        episode_per_test=10,
        batch_size=64,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        update_per_step=0.1,
        logger=logger,
        verbose=True,
    )

    # Train the model and log results
    result = trainer.run()
    print("Training finished!", result)

    # Step 7: Save the Policy
    torch.save(policy.state_dict(), policy_path)
    print(f"Policy saved to {policy_path}")

    # Step 8: Evaluate the Model
    policy.eval()
    test_collector.reset()
    result = test_collector.collect(n_episode=10)
    print("Evaluation Results ",result)

    # Step 9: Close TensorBoard Writer
    writer.close()