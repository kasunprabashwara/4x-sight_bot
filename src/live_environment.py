import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Dict
from itertools import combinations
from src.mt5_wrapper import MT5Wrapper
from src.utils import get_pair_price_from_row
from src.config import BARS_COUNT, BASE_CURRENCY, CURRENCIES, LEVERAGE, TRADE_MAX_PERCENTAGE

class LiveForexEnv(Env):
    # __init__ and _get_observation are unchanged from the previous correct version
    def __init__(self, currencies=CURRENCIES, base_currency=BASE_CURRENCY, bars_count=BARS_COUNT, verbose=False):
        super().__init__()
        self.mt5 = MT5Wrapper()
        self.bars_count, self.verbose = bars_count, verbose
        self.base_currency = base_currency
        self.portfolio_currencies = sorted(list(set([base_currency] + currencies)))
        self.pairs = sorted([tuple(sorted(c)) for c in combinations(self.portfolio_currencies, 2)])
        self.price_symbols = []
        for currency in currencies:
            direct, inverse = f"{currency}{base_currency}", f"{base_currency}{currency}"
            if self.mt5.is_symbol_available(direct): self.price_symbols.append(direct)
            elif self.mt5.is_symbol_available(inverse): self.price_symbols.append(inverse)
            else: self.mt5.shutdown(); raise ValueError(f"FATAL: Symbol for '{currency}' not found.")
        print(f"Required price symbols found: {self.price_symbols}")
        self.action_space = Box(low=-1, high=1, shape=(len(self.pairs),), dtype=np.float32)
        self.observation_space = Dict({"past_prices": Box(low=-np.inf, high=np.inf, shape=(bars_count, len(self.pairs))), "portfolio": Box(low=0, high=1, shape=(len(self.portfolio_currencies),))})

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.account_info = self.mt5.get_account_info()
        self.balance = self.account_info.equity
        return self._get_observation(), {}

    def _get_observation(self):
        price_data = self.mt5.get_initial_data(self.price_symbols, self.bars_count)
        if price_data is None: raise RuntimeError("Could not fetch initial price data from MT5.")
        past_prices = np.zeros((self.bars_count, len(self.pairs)))
        for i in range(len(price_data)):
            row = price_data.iloc[i]
            for j, pair in enumerate(self.pairs):
                past_prices[i, j] = get_pair_price_from_row(row, pair, self.base_currency)
        portfolio_frac = np.array([1.0 if c == self.base_currency else 0.0 for c in self.portfolio_currencies])
        return {"past_prices": past_prices, "portfolio": portfolio_frac}
    
    def step(self, action):
        """Executes trades based on the model's action and returns the new market state."""
        print("\n[Live Environment Step]")
        current_equity = self.mt5.get_account_info().equity
        print(f"Current Equity: {current_equity:.2f} {self.base_currency}")

        # Define the maximum capital to risk per trade signal
        per_pair_trade_limit_base = current_equity * (LEVERAGE / len(self.pairs)) * TRADE_MAX_PERCENTAGE
        
        flat_action = action.flatten()

        for i, (A_orig, B_orig) in enumerate(self.pairs):
            trade_direction = flat_action[i]
            # Determine direction and calculate the value of the trade in base currency
            A, B = (A_orig, B_orig) if trade_direction > 0 else (B_orig, A_orig)
            trade_value_in_base = per_pair_trade_limit_base * abs(trade_direction)

            print(f"  - Pair ({A_orig}/{B_orig}): Action value = {trade_direction:.4f}")
            print(f"    -> Signal: BUY '{A}' with '{B}'.")
            print(f"    -> Calculated Trade Value: {trade_value_in_base:.2f} {self.base_currency}")
            
            # The wrapper handles all complex execution logic
            self.mt5.check_and_trade(
                currency_A=A, 
                currency_B=B, 
                trade_value_in_base_currency=trade_value_in_base
            )

        print("[Live Environment] Fetching new market state post-trade...")
        new_obs = self._get_observation()
        new_equity = self.mt5.get_account_info().equity
        reward = new_equity - current_equity
        info = {"equity": f"{new_equity:.2f}"}
        print(f"[Live Environment Step Complete] Info: {info}")
        
        # Live environments are continuous; they don't terminate on their own.
        return new_obs, reward, False, False, info

    def close(self):
        self.mt5.shutdown()