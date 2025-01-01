import gymnasium as gym
from gymnasium import spaces
import numpy as np
from mt5_wrapper import MT5Wrapper
import time

LOT_SIZE = 100000

class State:
    """
    State representation for the Forex Trading environment.
    Encodes the current portfolio and exchange rates.
    """

    def __init__(self, bars_count=4):
        self.bars_count = bars_count  # Number of historical bars to include in state
        self._offset = None  # Current position in the dataset
        self.balance = None  # Portfolio balance in base currency
        self.portfolio = None  # Portfolio allocation
        self.euro_buy_value = None
        self.yen_buy_value = None
        self.self.mt5 = MT5Wrapper(".env")

    def reset(self, initial_balance):
        """
        Reset the state for a new episode.
        :param prices: DataFrame containing price data (EUR/USD, JPY/USD, etc.)
        :param offset: Initial offset in the price data
        :param initial_balance: Starting balance for the portfolio
        """
        self.balance = initial_balance
        self.portfolio = {'USD': initial_balance, 'EUR': 0, 'JPY': 0}
        self.euro_buy_value = 0
        self.yen_buy_value = 0

    @property
    def shape(self):
        #NOT DONE YET
        """
        Return the shape of the encoded state.
        The state includes historical exchange rates and portfolio allocations.
        """
        return (self.bars_count * 3 + 3,)

    def encode(self):
        """
        Encode the current state as a numpy array for the agent.
        :return: Encoded state array
        """
        state = np.zeros(self.shape, dtype=np.float32)

        # Include historical prices
        # for i in range(self.bars_count):
        #     # bar_offset = self._offset - self.bars_count + i + 1
        #     state[i * 3] = self._prices.iloc[bar_offset]['EURO AREA - EURO/US$']
        #     state[i * 3 + 1] = self._prices.iloc[bar_offset]['JAPAN - YEN/US$']
        #     state[i * 3 + 2] = self._prices.iloc[bar_offset]['YEN/EURO']

        # Include portfolio state
        state[-3] = self.portfolio['USD']
        state[-2] = self.portfolio['EUR']
        state[-1] = self.portfolio['JPY']

        return state

    def step(self, action, trade_percentage=0.5):
        """
        Perform an action and update the portfolio state.
        :param action: The action to perform
        :param current_price: Current price data (EUR/USD, JPY/USD, JPY/EUR)
        :return: Reward and done flag
        """
        eur_usd = self.self.mt5.symbol_info_tick("EURUSD").ask
        jpy_usd = self.self.mt5.symbol_info_tick("USDJPY").ask
        jpy_eur = self.self.mt5.symbol_info_tick("EURJPY").ask

        print("eur_usd", eur_usd, "jpy_usd", jpy_usd, "jpy_eur", jpy_eur)


        print("action", action) 

        if self.portfolio['EUR'] !=0:
            print("euro buy value", self.euro_buy_value/self.portfolio['EUR'])
        
        if self.portfolio['JPY'] !=0:
            print("yen buy value", self.yen_buy_value/self.portfolio['JPY'])

        trade_amount = self.balance*trade_percentage/100
        volume=0.01

        # Update portfolio based on action
        if action == 1:  # Buy EUR with USD

            trade_volume = min(self.portfolio['USD'], trade_amount)
            print("trade volume", trade_volume * eur_usd/LOT_SIZE)
            self.mt5.place_buy_order("EURUSD", round(trade_volume * eur_usd/LOT_SIZE,2))
            # place_buy_order("EURUSD", 0.005)

            self.portfolio['EUR'] += trade_volume * eur_usd
            self.portfolio['USD'] -= trade_volume

            self.euro_buy_value += trade_volume

        elif action == 2:  # Sell EUR for USD

            trade_volume = min(self.portfolio['EUR'], trade_amount*eur_usd)
            print("trade volume", trade_volume/LOT_SIZE)
            self.mt5.place_sell_order("EURUSD", round(trade_volume / LOT_SIZE,2))
            # place_sell_order("EURUSD", 0.005)

            self.portfolio['USD'] += trade_volume / eur_usd
            self.portfolio['EUR'] -= trade_volume

            self.euro_buy_value = self.euro_buy_value - trade_volume/eur_usd

        elif action == 3:  # Buy JPY with USD

            trade_volume = min(self.portfolio['USD'], trade_amount)
            print("trade volume", trade_volume/LOT_SIZE)
            self.mt5.place_sell_order("USDJPY", round(trade_volume/LOT_SIZE,2))
            # place_sell_order("USDJPY", 0.005)


            self.portfolio['JPY'] += trade_volume * jpy_usd
            self.portfolio['USD'] -= trade_volume

            self.yen_buy_value += trade_volume
            
        elif action == 4:  # Sell JPY for USD

            trade_volume = min(self.portfolio['JPY'], trade_amount*jpy_usd)
            print("trade volume",(trade_volume/jpy_usd)/LOT_SIZE)
            self.mt5.place_buy_order("USDJPY", round((trade_volume/jpy_usd)/LOT_SIZE,2))
            # place_buy_order("USDJPY", 0.005)

            self.portfolio['USD'] += trade_volume / jpy_usd
            self.portfolio['JPY'] -= trade_volume

            self.yen_buy_value -= trade_volume/jpy_usd
        elif action == 5:  # Buy JPY with EUR

            trade_volume = min(self.portfolio['EUR'], trade_amount*eur_usd)
            self.mt5.place_sell_order("EURJPY", round(trade_volume/LOT_SIZE,2))
            # place_sell_order("EURJPY", 0.005)


            self.portfolio['JPY'] += trade_volume * jpy_eur
            self.portfolio['EUR'] -= trade_volume

            self.euro_buy_value -= trade_volume/eur_usd
            self.yen_buy_value += trade_volume/eur_usd   #sus
            
        elif action == 6:  # Sell JPY for EUR

            trade_volume = min(self.portfolio['JPY'], trade_amount*jpy_usd)
            self.mt5.place_buy_order("EURJPY", round((trade_volume/jpy_eur)/LOT_SIZE,2))
            # place_buy_order("EURJPY", 0.005)

            self.portfolio['EUR'] += trade_volume / jpy_eur
            self.portfolio['JPY'] -= trade_volume

            self.euro_buy_value += trade_volume/jpy_usd
            self.yen_buy_value -= trade_volume/jpy_usd   #sus

        # Calculate portfolio value in USD
        portfolio_value = (
            self.portfolio['USD']
            + self.portfolio['EUR'] * eur_usd
            + self.portfolio['JPY'] / jpy_usd
        )

        # Calculate reward as the change in portfolio value
        reward = portfolio_value - self.balance
        self.balance = portfolio_value
        # TODO done flag check again
        done = self._offset >= len(self._prices) - 1
        return reward, done
    

class MT5Env(gym.Env):
    """A custom environment for forex trading (EUR/USD, JPY/USD, and YEN/EURO)."""
    
    def __init__(self, initial_balance=1000, bars_count=4):
        """
        Initialize the environment.
        
        :param df: DataFrame with forex data (including exchange rates).
        :param initial_balance: Starting balance for the portfolio in USD.
        :param bars_count: Number of historical bars to include in the state.
        """
        super(MT5Env, self).__init__()
        
        self.initial_balance = initial_balance 
        self.bars_count = bars_count  
        

        self.state = State(bars_count=self.bars_count)
        
        # Define the action space (7 actions as described in the State class)
        self.action_space = spaces.Discrete(7)
        
        # Define the observation space based on the shape of the state
        state_shape = self.state.shape
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=state_shape, dtype=np.float32
        )

        self.done = False  # Flag to indicate the end of an episode

    def reset(self):
        """
        Reset the environment at the start of a new episode.
        
        :return: Initial observation.
        """
        # Reset the state
        self.state.reset(initial_balance=self.initial_balance)
        self.done = False
        
        # Return the encoded initial state
        return self.state.encode()

    def step(self, action, trade_percentage=5):
        """
        Execute one step in the environment.
        
        :param action: Action chosen by the agent.
        :return: Tuple (observation, reward, done, info).
        """
        if self.done:
            raise Exception("Cannot call step() on a completed environment. Call reset() first.")
        
        # Perform the action and update the state
        reward, self.done = self.state.step(action, trade_percentage)
        
        # Encode the new state
        observation = self.state.encode()
        
        # Return observation, reward, done flag, and additional info
        return observation, reward, self.done, {}

    def render(self, mode='human'):
        """
        Render the environment.
        
        :param mode: Rendering mode (only 'human' is supported for now).
        """
        if mode != 'human':
            raise NotImplementedError("Only 'human' rendering mode is supported.")
        
        # Print state information
        print(f"Step: {self.state._offset}")
        print(f"Portfolio: {self.state.portfolio}")
        print(f"Balance: {self.state.balance}")



# equity = self.mt5.account_info().equity
# if equity is not None:
#     print(f"Current equity: {equity}")
#     env = MT5Env(initial_balance=equity)
# else:
#     print("Could not retrieve equity.")
#     env = MT5Env()

# state = env.reset()
# done = False

# action = env.action_space.sample() 
# state, reward, done, info = env.step(action)
# env.render()


# symbol = "EURUSD"
# last_tick_time = None

# while True:
#     ticks = self.mt5.copy_ticks_from(symbol, time.time() - 1, 1, self.mt5.COPY_TICKS_ALL)
#     if len(ticks) > 0:
#         tick = ticks[-1] 
#         if last_tick_time is None or tick['time'] > last_tick_time:
#             last_tick_time = tick['time']

#             # Perform an environment step
#             action = env.action_space.sample()  # Replace with your action logic
#             state, reward, done, info = env.step(action)
#             env.render()

#             print(f"Tick received at {tick['time']} - Price: {tick['bid']}/{tick['ask']}")

#             # Exit if the environment is done
#             if done:
#                 break
#     time.sleep(2)
