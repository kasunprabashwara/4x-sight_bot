import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Dict
from itertools import combinations
from src.state import State
from src.config import (
    BARS_COUNT, 
    MAX_STEPS, 
    BASE_CURRENCY, 
    CURRENCIES, 
    INITIAL_BALANCE, 
    TRADE_MAX_PERCENTAGE
)

class ForexTradingEnv(Env):
    def __init__(self, df, currencies=CURRENCIES, base_currency=BASE_CURRENCY, initial_balance=INITIAL_BALANCE, verbose=False,
                 bars_count=BARS_COUNT, max_steps=MAX_STEPS):
        super(ForexTradingEnv, self).__init__()
        self.df = df.copy()
        self.initial_balance = initial_balance
        self.bars_count = bars_count
        self.max_steps = max_steps
        self.steps = 0
        self.base_currency = base_currency
        self.currencies = currencies
        self.portfolio_currencies = sorted(list(set([base_currency] + currencies)))

        pairs = []
        for comb in combinations(self.portfolio_currencies, 2):
            pair = tuple(sorted(comb))
            pairs.append(pair)
        pairs = sorted(pairs)
        self.pairs = pairs

        self.action_space = Box(low=-1, high=1, shape=(len(self.pairs),), dtype=np.float32)

        self.observation_space = Dict({
            "past_prices": Box(low=0, high=np.inf, shape=(bars_count, len(self.pairs)), dtype=np.float32),
            "portfolio": Box(low=0, high=1, shape=(len(self.portfolio_currencies),), dtype=np.float32)
        })

        self.state = State(bars_count=self.bars_count, verbose=verbose)

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.steps = 0
        rng = np.random.default_rng(seed)
        offset = rng.integers(self.bars_count - 1, len(self.df) - 1)
        self.state.reset(prices=self.df, offset=offset,
                         initial_balance=self.initial_balance,
                         trade_max_percentage=TRADE_MAX_PERCENTAGE,
                         pairs=self.pairs,
                         base_currency=self.base_currency,
                         portfolio_currencies=self.portfolio_currencies)
        return self.state.encode(), {}

    def step(self, action):
        reward, terminated, info = self.state.step(action)
        truncated = self.state._offset >= len(self.df) - 1
        observation = self.state.encode()
        self.steps += 1
        if self.steps >= self.max_steps:
            terminated = True
        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        if mode != 'human':
            raise NotImplementedError("Only 'human' rendering mode is supported.")
        print(f"Step: {self.state._offset}")
        print(f"Portfolio: {self.state.portfolio}")
        print(f"Balance: {self.state.balance}")
