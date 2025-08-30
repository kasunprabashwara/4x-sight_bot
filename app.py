import os
import time
import schedule
import numpy as np
from sb3_contrib import RecurrentPPO
from src.live_environment import LiveForexEnv
from src.config import MODEL_SAVE_PATH
from train import train_model

class TradingBot:
    def __init__(self):
        print("Initializing trading bot...")
        self.env = LiveForexEnv()
        
        if not os.path.exists(MODEL_SAVE_PATH):
            print("No model found. Please train the model first by running train.py")
            self.env.close()
            exit()
            
        print(f"Loading model from {MODEL_SAVE_PATH}")
        self.model = RecurrentPPO.load(MODEL_SAVE_PATH, env=None)
        
        # State for the recurrent (LSTM) policy
        self.lstm_states = None
        self.num_envs = 1  # We are running a single live environment
        self.episode_starts = np.ones((self.num_envs,), dtype=bool)
        
        # Get the initial observation from the market
        self.obs, _ = self.env.reset()
        print("Bot initialized successfully.")

    def run_trading_cycle(self):
        print("\n--- New Trading Cycle ---")
        try:
            # Predict the action using the AI model and its previous internal state (lstm_states)
            action, self.lstm_states = self.model.predict(
                self.obs,
                state=self.lstm_states,
                episode_start=self.episode_starts,
                deterministic=True
            )
            
            # The 'episode_starts' flag is set to False after the first run
            self.episode_starts = np.zeros((self.num_envs,), dtype=bool)

            # The environment's step function handles all trading logic
            # and returns the next observation of the market.
            new_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Update the observation for the next trading cycle
            self.obs = new_obs
            
            print(f"Cycle complete. Info: {info}")

        except Exception as e:
            print(f"An error occurred during the trading cycle: {e}")
            # Reset LSTM state on error to start fresh
            self.lstm_states = None
            self.episode_starts = np.ones((self.num_envs,), dtype=bool)

    def start(self, interval_seconds=300):
        print(f"Bot starting. Trading cycle will run every {interval_seconds} seconds.")
        # Run once immediately at the start
        self.run_trading_cycle()
        
        schedule.every(interval_seconds).seconds.do(self.run_trading_cycle)

        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nBot shutting down...")
            self.env.close()
            print("Shutdown complete.")

def main():
    if not os.path.exists(MODEL_SAVE_PATH):
        print("No trained model found. Starting training process...")
        train_model()
    
    bot = TradingBot()
    # Using a 5-minute interval. For H1 data, a 1-hour interval (3600s) might be more appropriate.
    bot.start(interval_seconds=20)

if __name__ == "__main__":
    main()