import math
import time

import torch as th
import numpy as np
# import pandas as pd
from itertools import combinations
from gymnasium import Env, spaces
from gymnasium.spaces import Box, Dict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from mt5_wrapper import MT5Wrapper

LOT_SIZE = 100000

def get_pair_price_from_row(row, pair, base_currency):
    """
    Given a row (a pandas Series) and a pair (tuple of two currencies),
    return the exchange rate defined as:
      - If the second currency equals the base, then the price is assumed to be available
        in the column f"{non_base}{base_currency}".
      - If the first currency equals the base, then the price is 1/(price from f"{other}{base_currency}").
      - Otherwise, compute the cross rate as (price of A in base)/(price of B in base).
    """
    A, B = pair
    if A == B:
        return 1.0
    if B == base_currency:
        return row[f"{A}{base_currency}"]
    elif A == base_currency:
        return 1.0 / row[f"{B}{base_currency}"]
    else:
        return row[f"{A}{base_currency}"] / row[f"{B}{base_currency}"]

# ///////////////////////////////////////////////////////////////////////////////////////////

class State:
    def __init__(self, bars_count=30, verbose=False):
        self.mt5_wrapper = MT5Wrapper(env_file=".env")
        self.bars_count = bars_count
        self._prices = None
        self.balance = None
        self.trade_max_percentage = None
        self.verbose = verbose
        # These will be set in reset:
        self.portfolio = None        # dict mapping currency -> amount (in native units)
        self.base_currency = None
        self.pairs = None            # list of tuples of currency pairs (sorted)
        self.portfolio_currencies = None  # list of all currencies in portfolio (including base)
        self.leverage = 60  # default leverage ratio

    def reset(self, trade_max_percentage,
              pairs, base_currency, portfolio_currencies):
        
        self.trade_max_percentage = trade_max_percentage
        self.base_currency = base_currency
        self.pairs = pairs
        self.portfolio_currencies = portfolio_currencies
        self.prices = self.mt5_wrapper.get_intial_data(self.pairs, self.bars_count)
        self.balance = self.mt5_wrapper.get_account_info().equity

        print(f"Initial balance: {self.balance} {self.base_currency}")
        print(f"Initial prices: {self.prices}")

        # Initialize portfolio: all funds in base_currency; zero in others.
        self.portfolio = {curr: (self.balance if curr == base_currency else 0) 
                          for curr in portfolio_currencies}
        self.ammortized_values = {curr: 0 for curr in portfolio_currencies}
        
    def get_pair_price(self, row, pair):
        return get_pair_price_from_row(row, pair, self.base_currency)

    def encode(self):
        """
        Build an observation dictionary that includes:
        - 'past_prices': a float32 array of shape (bars_count, num_pairs) computed dynamically
        - 'portfolio': a float32 array of portfolio fractions (computed in base currency)
        """
        # Use the entire `prices` array (already contains `bars_count` rows).
        historical = self.prices
        bars = self.bars_count
        num_pairs = len(self.pairs)
        past_prices = np.zeros((bars, num_pairs), dtype=np.float32)

        # Compute price for each pair for every historical row.
        for i, row in enumerate(historical):
            for j, pair in enumerate(self.pairs):
                past_prices[i, j] = self.get_pair_price(row, pair)

        # Compute current portfolio value in base currency using the latest row.
        current_row = historical[-1]
        total_value = 0.0
        portfolio_values = {}
        for curr, amt in self.portfolio.items():
            if curr == self.base_currency:
                val = amt
            else:
                val = amt / current_row[f"{curr}{self.base_currency}"]
            portfolio_values[curr] = val
            total_value += val

        # Compute portfolio fractions in a fixed (sorted) order.
        sorted_curr = sorted(self.portfolio.keys())
        portfolio_frac = np.array([portfolio_values[c] for c in sorted_curr], dtype=np.float32)
        portfolio_frac = portfolio_frac / (total_value + 1e-8)

        return {"past_prices": past_prices, "portfolio": portfolio_frac}

    @property
    def shape(self):
        """
        Returns a dict mapping observation keys to their shapes.
        """
        return {
            "past_prices": (self.bars_count, len(self.pairs)),
            "portfolio": (len(self.portfolio),)
        }

    def step(self, action, reward_type="Direct"):
        """
        Process a trade action vector (one action per pair). For each pair (A,B)
        in self.pairs (sorted lexicographically), interpret a positive action as
        "buy A using B" and a negative action as "sell A for B." Trades are capped
        by a maximum trade amount (based on current balance, leverage, and trade_max_percentage).
        Returns the reward.
        """
        reward = 0
        current_row = self.prices[-1]  # Use the latest row in `prices`.

        # Fetch the next row of live data and update `prices`.
        next_row = {}
        for currency in self.portfolio_currencies:
            if currency != self.base_currency:
                symbol = f"{currency}{self.base_currency}"
                price = self.mt5_wrapper.get_symbol_price(symbol)
                if price is not None:
                    next_row[symbol] = price
                else:
                    print(f"Failed to fetch price for {symbol}.")
                    return 0, True, {"error": f"Failed to fetch price for {symbol}"}

        
        self.prices = np.vstack([self.prices[1:], next_row])  # Remove the oldest row and append the new one.

        num_pairs = len(self.pairs)
        # For each pair, compute a max trade amount (dividing available leverage across pairs)
        per_pair_trade = self.balance * (self.leverage / num_pairs) * self.trade_max_percentage

        # Loop over each pair and perform the trade.
        for i, (A, B) in enumerate(self.pairs):
            # if the action is negative, we swap the currencies
            if action[i] < 0:
                A, B = B, A
            a_base_price = self.get_pair_price(current_row, (A, self.base_currency))
            b_base_price = self.get_pair_price(current_row, (B, self.base_currency))
            # Buy currency A using currency B.
            trade_amount = per_pair_trade * abs(action[i])
            trade_amount = min(trade_amount, trade_amount + (self.portfolio[B]/ b_base_price))
            if trade_amount > 0:
                # You spend trade_amount of B to get (trade_amount/price) of A.
                self.portfolio[A] += trade_amount * a_base_price
                self.portfolio[B] -= trade_amount * b_base_price
                if reward_type =="InDirect":
                    if A == self.base_currency:
                        reward += trade_amount - (self.ammortized_values[B]/self.portfolio[B]) * trade_amount * b_base_price
                    self.ammortized_values[A] += trade_amount
                    self.ammortized_values[B] -= trade_amount
                if self.verbose:
                    print(f"{current_row['Timestamp']}: Spent {trade_amount* b_base_price:.2f} {B} to buy {trade_amount* a_base_price:.2f} {A}")
        # Compute new total portfolio value in base currency using next_row prices.
        new_value = 0.0
        for curr, amt in self.portfolio.items():
            if curr == self.base_currency:
                val = amt
            else:
                val = amt / next_row[f"{curr}{self.base_currency}"]
            new_value += val
        if(reward_type == "Direct"):
            # Compute direct reward as the log return.
            if new_value > 0 and self.balance > 0:
                reward = math.log(new_value / self.balance)
            else:
                reward = 0

        self.balance = new_value
        done = False
        info = {"balance": self.balance, "portfolio": self.portfolio}
        return reward, done, info
    
class LSTMFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: dict, features_dim=128, lstm_hidden_size=64):
        """
        observation_space: expects a Dict with keys:
          - "past_prices": Tensor of shape (bars_count, num_pairs)
          - "portfolio": Tensor of shape (portfolio_dim,)
        features_dim: dimension of the final feature vector.
        lstm_hidden_size: hidden state size for the LSTM processing the price sequence.
        """
        super(LSTMFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Get shapes from the observation space.
        self.bars_count, self.num_pairs = observation_space.spaces["past_prices"].shape
        self.portfolio_dim = observation_space.spaces["portfolio"].shape[0]

        # LSTM to process the price sequence.
        self.lstm = nn.LSTM(
            input_size=self.num_pairs,
            hidden_size=lstm_hidden_size,
            batch_first=True
        )

        # After computing indicators (RSI & MACD) for each pair, we have 2*num_pairs values.
        self.indicator_dim = 2 * self.num_pairs

        # Fully-connected layer to process portfolio info.
        self.portfolio_fc = nn.Sequential(
            nn.Linear(self.portfolio_dim, 32),
            nn.ReLU()
        )
        
        # Final fully-connected layer to combine all features.
        # The input dimension is: lstm_hidden_size + indicator_dim + 32.
        self.combined_fc = nn.Sequential(
            nn.Linear(lstm_hidden_size + self.indicator_dim + 32, features_dim),
            nn.ReLU()
        )
    def compute_ema(self, prices: th.Tensor, span: int) -> th.Tensor:
        """
        Compute the exponential moving average (EMA) for the given prices.
        prices: Tensor of shape (batch_size, bars_count, num_pairs)
        span: period span for the EMA.
        Returns:
            A tensor of shape (batch_size, num_pairs) containing the final EMA value.
        """
        alpha = 2.0 / (span + 1)
        batch_size, T, num_pairs = prices.shape
        # Initialize with the first time-step.
        ema = prices[:, 0, :]  # shape: (batch_size, num_pairs)
        # Iteratively update the EMA over time.
        for t in range(1, T):
            ema = alpha * prices[:, t, :] + (1 - alpha) * ema
        return ema  # shape: (batch_size, num_pairs)

    def compute_macd(self, prices: th.Tensor) -> th.Tensor:
        """
        Compute the MACD (Moving Average Convergence Divergence) indicator.
        Uses a short-term EMA (span 12) and a long-term EMA (span 26).
        Returns:
            A tensor of shape (batch_size, num_pairs) representing the MACD.
        """
        ema_short = self.compute_ema(prices, span=12)
        ema_long = self.compute_ema(prices, span=26)
        macd = ema_short - ema_long
        return macd

    def compute_rsi(self, prices: th.Tensor, period: int = 14) -> th.Tensor:
        """
        Compute the Relative Strength Index (RSI) using Wilder's smoothing method.
        prices: Tensor of shape (batch_size, bars_count, num_pairs)
        period: period for computing RSI (default: 14)
        Returns:
            A tensor of shape (batch_size, num_pairs) representing the RSI.
        """
        # Compute differences along the time dimension.
        diff = prices[:, 1:, :] - prices[:, :-1, :]  # shape: (batch_size, bars_count-1, num_pairs)
        # Separate gains and losses.
        gain = th.clamp(diff, min=0)
        loss = -th.clamp(diff, max=0)
        batch_size, T_minus_1, num_pairs = gain.shape
        # Adjust period if the sequence is too short.
        period = min(period, T_minus_1)
        # Initialize the average gain and loss using the first 'period' values.
        avg_gain = gain[:, :period, :].mean(dim=1)
        avg_loss = loss[:, :period, :].mean(dim=1)
        # Update the averages using Wilder's smoothing method.
        for t in range(period, T_minus_1):
            current_gain = gain[:, t, :]
            current_loss = loss[:, t, :]
            avg_gain = (avg_gain * (period - 1) + current_gain) / period
            avg_loss = (avg_loss * (period - 1) + current_loss) / period
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def forward(self, observations: dict) -> th.Tensor:
        """
        observations: dict with keys "past_prices" and "portfolio"
          - past_prices: Tensor of shape (batch_size, bars_count, num_pairs)
          - portfolio: Tensor of shape (batch_size, portfolio_dim)
        """
        past_prices = observations["past_prices"]
        # we pass price differences through the LSTM
        price_diffs = past_prices[:, 1:, :] - past_prices[:, :-1, :]  # Shape: (batch_size, bars_count-1, num_pairs)
        portfolio = observations["portfolio"]

        # Process the price sequence through the LSTM.
        _, (h_n, _) = self.lstm(price_diffs)
        lstm_last = h_n.squeeze(0)  # shape: (batch_size, lstm_hidden_size)

        # Compute the technical indicators.
        rsi = self.compute_rsi(past_prices, period=14)   # shape: (batch_size, num_pairs)
        macd = self.compute_macd(past_prices)              # shape: (batch_size, num_pairs)
        # Directly concatenate the raw indicator outputs.
        indicators = th.cat([rsi, macd], dim=1)            # shape: (batch_size, 2*num_pairs)

        # Process portfolio information.
        portfolio_features = self.portfolio_fc(portfolio)

        # Combine LSTM output, raw indicators, and portfolio features.
        combined = th.cat([lstm_last, indicators, portfolio_features], dim=1)
        return self.combined_fc(combined)

class ForexTradingEnv(Env):
    def __init__(self, currencies=["EUR","GBP"], base_currency="USD",verbose=False,
                 bars_count=30, max_steps=5000):
        """
        df: pandas DataFrame containing price data. It is assumed to have columns like "EURUSD", "GBPUSD", etc.
        currencies: list of non-base currencies (e.g. ["EUR","GBP"]).
        base_currency: the base currency (e.g. "USD").
        """
        super(ForexTradingEnv, self).__init__()
        # self.initial_balance = initial_balance
        self.bars_count = bars_count
        self.max_steps = max_steps
        self.steps = 0
        self.base_currency = base_currency
        self.currencies = currencies  # non-base currencies

        # The portfolio will include the base and the other currencies.
        self.portfolio_currencies = sorted(list(set([base_currency] + currencies)))

        # Create all unique pairs from portfolio currencies.
        # (For example, if portfolio currencies are ["EUR", "GBP", "USD"],
        # the pairs will be: ("EUR","GBP"), ("EUR","USD"), ("GBP","USD"))
        pairs = []
        for comb in combinations(self.portfolio_currencies, 2):
            pair = tuple(sorted(comb))
            pairs.append(pair)
        pairs = sorted(pairs)
        self.pairs = pairs

        # Define the dynamic action space: one continuous action per pair.
        self.action_space = Box(low=-1, high=1, shape=(len(self.pairs),), dtype=np.float32)

        # Define the observation space as a Dict with two keys.
        self.observation_space = Dict({
            "past_prices": Box(low=0, high=np.inf, shape=(bars_count, len(self.pairs)), dtype=np.float32),
            "portfolio": Box(low=0, high=1, shape=(len(self.portfolio_currencies),), dtype=np.float32)
        })

        self.state = State(bars_count=self.bars_count,verbose=verbose)

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.steps = 0
        rng = np.random.default_rng(seed)

        self.state.reset(
                         trade_max_percentage=0.2,
                         pairs=self.pairs,
                         base_currency=self.base_currency,
                         portfolio_currencies=self.portfolio_currencies)
        return self.state.encode(), {}

    def step(self, action):
        reward, terminated, info = self.state.step(action)
        truncated = False
        observation = self.state.encode()
        self.steps += 1
        if self.steps >= self.max_steps:
            terminated = True
        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        if mode != 'human':
            raise NotImplementedError("Only 'human' rendering mode is supported.")

        print(f"Portfolio: {self.state.portfolio}")
        print(f"Balance: {self.state.balance}")




if __name__ == "__main__":
    import time

    # Initialize the ForexTradingEnv
    env = ForexTradingEnv(
        currencies=["EUR", "GBP"],  # Example currencies
        base_currency="USD",
        verbose=True,
        bars_count=30,
        max_steps=5000
    )

    # Reset the environment to get the initial observation
    observation, _ = env.reset()

    print("Environment initialized. Starting random actions...")

    # Loop to generate and execute random actions
    for i in range(10):
        print(f"\nStep {i + 1}:")
        
        # Generate a random action based on the action space
        action = env.action_space.sample()
        print(f"Generated action: {action}")

        # Execute the action in the environment
        observation, reward, terminated, truncated, info = env.step(action)

        # Print the results of the step
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print(f"Info: {info}")

        # Check if the environment is terminated
        if terminated:
            print("Environment terminated. Resetting...")
            observation, _ = env.reset()

        # Wait for 20 seconds before the next action
        time.sleep(20)

    print("Random action loop completed.")
