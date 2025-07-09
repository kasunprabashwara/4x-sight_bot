# src/training_environment.py
import numpy as np
import pandas as pd
from gymnasium import Env
from gymnasium.spaces import Box, Dict
from itertools import combinations
from src.utils import get_pair_price_from_row # <-- CORRECTED IMPORT
from src.config import (
    BARS_COUNT, 
    MAX_STEPS_TRAINING, 
    BASE_CURRENCY, 
    CURRENCIES, 
    TRADE_MAX_PERCENTAGE,
    LEVERAGE
)

class TrainingForexEnv(Env):
    # __init__ method is unchanged
    def __init__(self, df, currencies=CURRENCIES, base_currency=BASE_CURRENCY, initial_balance=10000,
                 bars_count=BARS_COUNT, max_steps=MAX_STEPS_TRAINING, verbose=False):
        super().__init__()
        self.df = df.copy()
        self.initial_balance = initial_balance
        self.bars_count = bars_count
        self.max_steps = max_steps
        self.verbose = verbose
        
        self.base_currency = base_currency
        self.portfolio_currencies = sorted(list(set([base_currency] + currencies)))
        self.pairs = sorted([tuple(sorted(c)) for c in combinations(self.portfolio_currencies, 2)])

        self.action_space = Box(low=-1, high=1, shape=(len(self.pairs),), dtype=np.float32)
        self.observation_space = Dict({
            "past_prices": Box(low=-np.inf, high=np.inf, shape=(bars_count, len(self.pairs)), dtype=np.float32),
            "portfolio": Box(low=0, high=1, shape=(len(self.portfolio_currencies),), dtype=np.float32)
        })

    # reset method is unchanged
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.steps_done = 0
        self.balance = self.initial_balance
        self.portfolio = {curr: (self.initial_balance if curr == self.base_currency else 0) 
                          for curr in self.portfolio_currencies}
        # Corrected offset calculation to prevent IndexError
        max_offset = len(self.df) - self.max_steps - 1
        min_offset = self.bars_count
        if min_offset > max_offset:
            raise ValueError("Not enough data in the DataFrame for even one full episode.")
        self.current_offset = self.np_random.integers(min_offset, max_offset)
        return self._get_observation(), {}


    def _get_observation(self):
        historical_data = self.df.iloc[self.current_offset - self.bars_count + 1 : self.current_offset + 1]
        past_prices = np.zeros((self.bars_count, len(self.pairs)), dtype=np.float32)
        for i, (_, row) in enumerate(historical_data.iterrows()):
            for j, pair in enumerate(self.pairs):
                # This function call is now robust
                past_prices[i, j] = get_pair_price_from_row(row, pair, self.base_currency)
        
        # This logic remains the same
        current_row = historical_data.iloc[-1]
        total_value = 0.0
        for curr, amt in self.portfolio.items():
            val = amt * get_pair_price_from_row(current_row, (curr, self.base_currency), self.base_currency)
            total_value += val
        
        portfolio_frac = np.zeros(len(self.portfolio_currencies), dtype=np.float32)
        if total_value > 0:
            for i, curr in enumerate(self.portfolio_currencies):
                portfolio_frac[i] = (self.portfolio[curr] * get_pair_price_from_row(current_row, (curr, self.base_currency), self.base_currency)) / total_value

        return {"past_prices": past_prices, "portfolio": portfolio_frac}
    
    # step method is unchanged
    def step(self, action):
        current_row = self.df.iloc[self.current_offset]
        
        for i, (A_orig, B_orig) in enumerate(self.pairs):
            trade_direction = action[i]
            if trade_direction == 0: continue
            
            A, B = (A_orig, B_orig) if trade_direction > 0 else (B_orig, A_orig)
            
            amount_B_to_spend = (self.portfolio[B]) * abs(trade_direction)
            price_A_in_B = get_pair_price_from_row(current_row, (A, B), self.base_currency)
            if price_A_in_B == 0: continue

            amount_A_to_acquire = amount_B_to_spend / price_A_in_B
            self.portfolio[B] -= amount_B_to_spend
            self.portfolio[A] += amount_A_to_acquire

        self.current_offset += 1
        self.steps_done += 1
        
        next_row = self.df.iloc[self.current_offset]
        new_balance = sum(amt * get_pair_price_from_row(next_row, (curr, self.base_currency), self.base_currency) for curr, amt in self.portfolio.items())
        
        reward = np.log(new_balance / self.balance) if self.balance > 0 and new_balance > 0 else 0
        self.balance = new_balance

        terminated = self.balance <= (self.initial_balance * 0.5) or self.steps_done >= self.max_steps
        truncated = self.current_offset >= len(self.df) - 1
        
        observation = self._get_observation()
        info = {"balance": self.balance}
        
        return observation, reward, terminated, truncated, info