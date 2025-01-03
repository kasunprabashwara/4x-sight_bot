import numpy as np
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from prediction import LSTMForexPredictor


class State:
    def __init__(self, bars_count=30, prediction_count=4):
        self.bars_count = bars_count
        self.prediction_count = prediction_count
        self._prices = None
        self._offset = None
        self._predictor : LSTMForexPredictor = LSTMForexPredictor()
        self._predictor.load_model("models/forex_lstm.keras")
        self.balance = None
        self.portfolio = None
        self.euro_buy_value = None
        self.yen_buy_value = None
        self.predictions = None

    def reset(self, prices, offset, initial_balance):
        assert offset >= self.bars_count - 1, "Offset must allow for sufficient historical data"
        self._prices = prices
        self._offset = offset
        self.balance = initial_balance
        self.portfolio = {'USD': initial_balance/3, 'EUR': initial_balance/3, 'JPY': initial_balance/3}
        self.euro_buy_value = 0
        self.yen_buy_value = 0

    def step(self, action, trade_percentage, reward_type = "Direct"):
        reward = None
        historical_prices = self._prices.iloc[self._offset - self.bars_count + 1: self._offset + 1]
        historical_prices_array = np.array(historical_prices[['EURO AREA - EURO/US$', 'JAPAN - YEN/US$', 'YEN/EURO']])
        euro_usd_column = historical_prices['EURO AREA - EURO/US$'].values

        # Ensure the past sequence length matches the model's look_back
        past_sequence = euro_usd_column[-self._predictor.look_back:]

        # Predict future values
        # print("past sequence", past_sequence)
        predictions = self._predictor.predict_future(past_sequence, self.prediction_count)
        self.predictions = predictions
        # print("predictions", predictions)

        current_price = self._prices.iloc[self._offset][['EURO AREA - EURO/US$', 'JAPAN - YEN/US$', 'YEN/EURO']].values
        eur_usd, jpy_usd, jpy_eur = current_price

        self.portfolio['USD'] = float(self.portfolio['USD'])
        self.portfolio['EUR'] = float(self.portfolio['EUR'])
        self.portfolio['JPY'] = float(self.portfolio['JPY'])

        trade_amount = self.balance * trade_percentage / 100

        if action == 0:  # Buy EUR with USD
            trade_volume = min(self.portfolio['USD'], trade_amount)

            reward = 0 

            self.portfolio['EUR'] += trade_volume * eur_usd
            self.portfolio['USD'] -= trade_volume
            self.euro_buy_value += trade_volume

        elif action == 1:  # Sell EUR for USD
            trade_volume = min(self.portfolio['EUR'], trade_amount * eur_usd)

            if trade_volume > 0:
                reward = trade_volume * (1/eur_usd-(self.euro_buy_value/self.portfolio['EUR']))

            self.portfolio['USD'] += trade_volume / eur_usd
            self.portfolio['EUR'] -= trade_volume
            self.euro_buy_value -= trade_volume / eur_usd

        elif action == 2:  # Buy JPY with USD
            trade_volume = min(self.portfolio['USD'], trade_amount)

            reward = 0

            self.portfolio['JPY'] += trade_volume * jpy_usd
            self.portfolio['USD'] -= trade_volume
            self.yen_buy_value += trade_volume

        elif action == 3:  # Sell JPY for USD
            trade_volume = min(self.portfolio['JPY'], trade_amount * jpy_usd)

            if trade_volume > 0:
                reward = trade_volume * (1/jpy_usd - self.yen_buy_value/self.portfolio['JPY'])

            self.portfolio['USD'] += trade_volume / jpy_usd
            self.portfolio['JPY'] -= trade_volume
            self.yen_buy_value -= trade_volume / jpy_usd

        elif action == 4:  # Buy JPY with EUR
            trade_volume = min(self.portfolio['EUR'], trade_amount * eur_usd)

            if trade_volume > 0:
                reward = trade_volume * (1/eur_usd - self.euro_buy_value/self.portfolio['EUR'])

            self.portfolio['JPY'] += trade_volume * jpy_eur
            self.portfolio['EUR'] -= trade_volume
            self.euro_buy_value -= trade_volume / eur_usd
            self.yen_buy_value += trade_volume / eur_usd

        elif action == 5:  # Sell JPY for EUR
            trade_volume = min(self.portfolio['JPY'], trade_amount * jpy_usd)

            if trade_volume > 0:
                reward = trade_volume * (1/jpy_usd - self.yen_buy_value/self.portfolio['JPY'])

            self.portfolio['EUR'] += trade_volume / jpy_eur
            self.portfolio['JPY'] -= trade_volume
            self.euro_buy_value += trade_volume / jpy_usd
            self.yen_buy_value -= trade_volume / jpy_usd

        if reward is not None:
            print("reward_indirect:", reward)

        portfolio_value = (self.portfolio['USD'] + self.portfolio['EUR'] / eur_usd + self.portfolio['JPY'] / jpy_usd)
        if reward_type == "Direct":
            reward = portfolio_value - self.balance
            print("reward_direct", reward)
        self.balance = portfolio_value

        self._offset += 1
        done = self._offset >= len(self._prices) - 1

        return reward, done

    def encode(self):
        # Extract historical prices
        historical_prices = self._prices.iloc[self._offset - self.bars_count + 1: self._offset + 1]
        historical_prices_array = np.array(historical_prices[['EURO AREA - EURO/US$', 'JAPAN - YEN/US$', 'YEN/EURO']])
        euro_usd_column = historical_prices['EURO AREA - EURO/US$'].values

        # Ensure the past sequence length matches the model's look_back
        past_sequence = euro_usd_column[-self._predictor.look_back:]

        # Predict future values
        # print("past sequence", past_sequence)
        # predictions = self._predictor.predict_future(past_sequence, self.prediction_count)
        # print("predictions", predictions)

        encoded_historical = historical_prices_array.flatten()  # Flatten historical prices
        # encoded_predictions = predictions.flatten()  
        # encoded = np.concatenate([encoded_historical, encoded_predictions])
        return encoded_historical
    @property
    def shape(self):
        # shape is bars_count * 3 + predictions count
        return (self.bars_count * 3 ,)

class ForexTradingEnv(Env):
    def __init__(self, df, initial_balance=1000, bars_count=30):
        super(ForexTradingEnv, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.bars_count = bars_count
        self.state = State(bars_count=self.bars_count)
        self.action_space = Discrete(5)
        self.observation_space = Box(
            low=0, high=np.inf, shape=self.state.shape, dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        offset = np.random.randint(self.bars_count - 1, len(self.df) - 1)
        self.state.reset(prices=self.df, offset=offset, initial_balance=self.initial_balance)
        return self.state.encode(), {}

    def step(self, action):
        reward, terminated = self.state.step(action, trade_percentage=10)
        truncated = self.state._offset >= len(self.df) - 1  
        observation = self.state.encode()
        return observation, reward, terminated, truncated, {}

    def render(self, mode='human'):
        if mode != 'human':
            raise NotImplementedError("Only 'human' rendering mode is supported.")
        print(f"Step: {self.state._offset}")
        print(f"Portfolio: {self.state.portfolio}")
        print(f"Balance: {self.state.balance}")
gym.register(
    id="gymnasium_env/ForexTrading-v0",
    entry_point=ForexTradingEnv,
)


# data = get_forex_data()
# env = ForexTradingEnv(data)

# # Test the environment
# state = env.reset()
# done = False

# # action = env.action_space.sample()  # Randomly sample an action
# # observation, reward, terminated, truncated, info = env.step(action)

# # env.render()

# # # action = env.action_space.sample()  # Randomly sample an action
# # # state, reward, done, info = env.step(action)
# # # env.render()

# # # action = env.action_space.sample()  # Randomly sample an action
# # # state, reward, done, info = env.step(action)
# # # env.render()

# i=0
# while (i<200) and (not done):
#     i+=1
#     print("i", i)
#     action = env.action_space.sample()  # Randomly sample an action
#     print("action", action)
#     observation, reward, terminated, truncated, info = env.step(action)
#     # env.encode()
#     env.render()