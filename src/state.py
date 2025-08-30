import math
import numpy as np
from src.data import get_pair_price_from_row

class State:
    def __init__(self, bars_count=30, verbose=False):
        self.bars_count = bars_count
        self._prices = None
        self._offset = None
        self.balance = None
        self.trade_max_percentage = None
        self.verbose = verbose
        self.portfolio = None
        self.base_currency = None
        self.pairs = None
        self.portfolio_currencies = None
        self.leverage = 60

    def reset(self, prices, offset, initial_balance, trade_max_percentage,
              pairs, base_currency, portfolio_currencies):
        assert offset >= self.bars_count - 1, "Offset must allow for sufficient historical data"
        self._prices = prices.copy()
        self._offset = offset
        self.balance = initial_balance
        self.trade_max_percentage = trade_max_percentage
        self.base_currency = base_currency
        self.pairs = pairs
        self.portfolio_currencies = portfolio_currencies
        self.portfolio = {curr: (initial_balance if curr == base_currency else 0) 
                          for curr in portfolio_currencies}
        self.ammortized_values = {curr: 0 for curr in portfolio_currencies}

    def get_pair_price(self, row, pair):
        return get_pair_price_from_row(row, pair, self.base_currency)

    def encode(self):
        historical = self._prices.iloc[self._offset - self.bars_count + 1: self._offset + 1]
        bars = self.bars_count
        num_pairs = len(self.pairs)
        past_prices = np.zeros((bars, num_pairs), dtype=np.float32)
        for i, (_, row) in enumerate(historical.iterrows()):
            for j, pair in enumerate(self.pairs):
                past_prices[i, j] = self.get_pair_price(row, pair)
        
        current_row = historical.iloc[-1]
        total_value = 0.0
        portfolio_values = {}
        for curr, amt in self.portfolio.items():
            if curr == self.base_currency:
                val = amt
            else:
                val = amt / current_row[f"{curr}{self.base_currency}"]
            portfolio_values[curr] = val
            total_value += val
        
        sorted_curr = sorted(self.portfolio.keys())
        portfolio_frac = np.array([portfolio_values[c] for c in sorted_curr], dtype=np.float32)
        portfolio_frac = portfolio_frac / (total_value + 1e-8)
        
        return {"past_prices": past_prices, "portfolio": portfolio_frac}

    @property
    def shape(self):
        return {
            "past_prices": (self.bars_count, len(self.pairs)),
            "portfolio": (len(self.portfolio),)
        }

    def step(self, action, reward_type="Direct"):
        reward = 0.0
        current_row = self._prices.iloc[self._offset]
        next_row    = self._prices.iloc[self._offset + 1]
        num_pairs   = len(self.pairs)
    
        per_pair_trade = self.balance * (self.leverage / num_pairs) * self.trade_max_percentage
    
        for i, (A_orig, B_orig) in enumerate(self.pairs):
            A, B = A_orig, B_orig
            if action[i] < 0:
                A, B = B_orig, A_orig
    
            a_base_price = self.get_pair_price(current_row, (A, self.base_currency))
            b_base_price = self.get_pair_price(current_row, (B, self.base_currency))
    
            trade_amount = per_pair_trade * abs(action[i])
            trade_amount = min(trade_amount, self.portfolio[B] / b_base_price)
    
            if trade_amount <= 0:
                continue
    
            spent_B = trade_amount * b_base_price
            acquired_A = trade_amount * a_base_price
            self.portfolio[A] += acquired_A
            self.portfolio[B] -= spent_B
    
            if reward_type == "InDirect":
                if self.ammortized_values.get(B, 0) <= 0:
                    self.ammortized_values[B] = b_base_price
    
                ratio = spent_B / self.ammortized_values[B]
                if ratio > 0:
                    reward += math.log(ratio)
                else:
                    reward += 0.0
    
                self.ammortized_values[A] = self.ammortized_values.get(A, 0) + spent_B
                self.ammortized_values[B] = self.ammortized_values.get(B, 0) - spent_B
    
            if self.verbose:
                print(f"{current_row['Timestamp']}: Spent {spent_B:.2f} {B} to buy {acquired_A:.2f} {A}")
    
        new_value = 0.0
        for curr, amt in self.portfolio.items():
            if curr == self.base_currency:
                new_value += amt
            else:
                price = next_row[f"{curr}{self.base_currency}"]
                new_value += amt / price
    
        if reward_type == "Direct":
            if new_value > 0 and self.balance > 0:
                reward = math.log(new_value / self.balance)
            else:
                reward = 0.0
    
        self.balance = new_value
        self._offset += 1
        done = (self._offset >= len(self._prices) - 2)
        info = {"balance": self.balance, "portfolio": self.portfolio}
    
        return reward, done, info
