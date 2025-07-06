import os
import time
import torch
import schedule
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.environment import ForexTradingEnv
from src.data import get_forex_data
from src.mt5_wrapper import MT5Wrapper
from src.config import MODEL_SAVE_PATH, CURRENCIES, BASE_CURRENCY, INITIAL_BALANCE
from train import train_model

def main():
    """Main function to run the trading bot."""
    if not os.path.exists(MODEL_SAVE_PATH):
        print("No model found, starting training...")
        train_model()

    data = get_forex_data()
    env_kwargs = {"df": data, "currencies": CURRENCIES, "base_currency": BASE_CURRENCY, "initial_balance": INITIAL_BALANCE}
    env = DummyVecEnv([lambda: ForexTradingEnv(**env_kwargs)])

    model = RecurrentPPO.load(MODEL_SAVE_PATH, env=env)

    mt5_wrapper = MT5Wrapper()

    def job():
        obs = env.reset()
        done = False
        lstm_states = None
        episode_starts = [True]
        while not done:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            obs, _, done, _ = env.step(action)
            
            # Get the pairs from the environment
            pairs = env.envs[0].pairs
            
            # Execute trades based on the action
            for i, (A, B) in enumerate(pairs):
                if action[0][i] > 0:
                    # Buy A with B
                    mt5_wrapper.check_and_trade(A, B, 100) # Example trade amount
                elif action[0][i] < 0:
                    # Sell A for B
                    mt5_wrapper.check_and_trade(B, A, 100) # Example trade amount

    schedule.every(5).seconds.do(job)

    print("Bot started, running job every 5 seconds...")
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
