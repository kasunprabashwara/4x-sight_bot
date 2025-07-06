import os
import torch
import warnings
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from src.data import get_forex_data
from src.environment import ForexTradingEnv
from src.model import LSTMFeatureExtractor
from src.config import (
    MODEL_SAVE_PATH, 
    LOG_DIR, 
    TOTAL_TIMESTEPS, 
    N_ENVS, 
    N_STEPS, 
    FEATURES_DIM, 
    LSTM_HIDDEN_SIZE
)

warnings.simplefilter(action="ignore", category=FutureWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    """Train the PPO model and save it."""
    data = get_forex_data()
    split_index = int(0.08 * len(data))
    train_data = data.iloc[:split_index]

    kwargs = {"df": train_data}
    train_envs = make_vec_env(ForexTradingEnv, n_envs=N_ENVS, env_kwargs=kwargs)
    train_envs = VecNormalize(train_envs, norm_obs=True, norm_reward=True, clip_obs=10.0)

    policy_kwargs = dict(
        features_extractor_class=LSTMFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=FEATURES_DIM, lstm_hidden_size=LSTM_HIDDEN_SIZE),
        lstm_hidden_size=256,
        n_lstm_layers=1,
        shared_lstm=False,
        enable_critic_lstm=True,
    )

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading existing model from {MODEL_SAVE_PATH}...")
        model = RecurrentPPO.load(MODEL_SAVE_PATH, env=train_envs, device=device, tensorboard_log=LOG_DIR)
    else:
        print("No existing model found. Training from scratch...")
        model = RecurrentPPO(
            "MlpLstmPolicy", 
            train_envs, 
            verbose=1, 
            device=device, 
            policy_kwargs=policy_kwargs,
            n_steps=N_STEPS
        )

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
    except KeyboardInterrupt:
        model.save(MODEL_SAVE_PATH)
        print(f"Training interrupted. Model saved to {MODEL_SAVE_PATH}")

    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
