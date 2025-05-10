import numpy as np
import torch
import torch.optim as optim
from data import get_forex_data
from tianshou.data import Batch
from training_environment import ForexTradingEnv
from tianshou.policy import DQNPolicy
from trainer_DQN_v1 import DuelingDQN


def evaluate_policy(policy, env, num_episodes=10):
    policy.eval()
    total_rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = policy(Batch(obs=np.array([obs]),info=Batch())).act[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        total_rewards.append(episode_reward)    
    print(f"Evaluation Rewards: {total_rewards}")
    print(f"Mean Reward: {np.mean(total_rewards)}, Std: {np.std(total_rewards)}")

if __name__ == "__main__":
    policy_path = "policies/dueling_dqn_forex.pth"
    data = get_forex_data()
    split_index = int(0.8 * len(data))
    test_data = data.iloc[split_index:]
    sample_env =ForexTradingEnv(test_data)
    input_dim = sample_env.observation_space.shape[0]
    output_dim = sample_env.action_space.n
    net = DuelingDQN(input_dim, output_dim)
    optim = optim.Adam(net.parameters(), lr=1e-4)
    policy = DQNPolicy(
        model=net,
        optim=optim,
        action_space=sample_env.action_space,
        discount_factor=0.99,
        estimation_step=3,
        target_update_freq=500,
    )
    policy.load_state_dict(torch.load(policy_path,weights_only=True))
    evaluate_policy(policy,sample_env,10)