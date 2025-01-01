import torch
import torch.optim as optim
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils.net.common import ActorCritic,Net

from tianshou.utils.net.discrete import Actor, Critic
from training_environment import ForexTradingEnv
from data import get_forex_data

# Step 1: Prepare the Dataset and Environment
data = get_forex_data()
split_index = int(0.8 * len(data))
train_data = data.iloc[:split_index]
test_data = data.iloc[split_index:]

train_envs = DummyVectorEnv([lambda: ForexTradingEnv(train_data)] * 5)  # Multiple envs for parallel training
test_envs = DummyVectorEnv([lambda: ForexTradingEnv(test_data)] * 2)
sample_env = ForexTradingEnv(test_data)

# Step 2: Setup PPOPolicy with Actor-Critic Model
input_dim = sample_env.observation_space.shape[0]
output_dim = sample_env.action_space.n
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define Actor-Critic Network
net = Net(state_shape=sample_env.observation_space.shape, hidden_sizes=[64, 64], device=device)
actor = Actor(preprocess_net=net, action_shape=sample_env.action_space.n, device=device).to(device)
critic = Critic(preprocess_net=net, device=device).to(device)
actor_critic = ActorCritic(actor, critic)

# Define Optimizer
optim = torch.optim.Adam(actor_critic.parameters(), lr=3e-4)

# PPO Policy
dist_fn = torch.distributions.Categorical
policy = PPOPolicy(
    actor=actor,
    critic=critic,
    optim=optim,
    dist_fn=dist_fn,
    action_space=sample_env.action_space,
    action_scaling=False,
    discount_factor=0.99,
    max_grad_norm=0.5,
    eps_clip=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    gae_lambda=0.95,
    reward_normalization=True,
)

# restore the saved policy state if there is any
policy_path = "policies/ppo_v1.pth"
try:
    policy.load_state_dict(torch.load(policy_path))
    print(f"Policy loaded from {policy_path}")
except FileNotFoundError:
    print(f"No policy found at {policy_path}")
# Step 3: Setup Replay Buffer and Collectors
train_collector = Collector(policy, train_envs, VectorReplayBuffer(100000, buffer_num=5))
test_collector = Collector(policy, test_envs)

# Step 4: TensorBoard Logger
writer = SummaryWriter("./logs")
logger = TensorboardLogger(writer)

# Step 5: Define Training and Testing Functions
def stop_fn(mean_reward):
    # Early stopping condition
    return mean_reward >= 1000

# Step 6: Train the Model
trainer = OnpolicyTrainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=100,  # Increase epochs for meaningful learning
    step_per_epoch=10000,
    repeat_per_collect=10,
    episode_per_test=10,
    batch_size=256,
    step_per_collect=2000,
    stop_fn=stop_fn,
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
print("Evaluation Results ", result)

# Step 9: Close TensorBoard Writer
writer.close()
