# 4x-Sight: A Forex Trading Bot

This project involves the development of an automated forex trading bot that uses reinforcement learning (RL) algorithms to optimize profit through multi-currency trading. The bot utilizes PPO (Proximal Policy Optimization) and DQN (Deep Q-Network) to adapt its strategy and maximize the profit over time.

## Features
- Multi-currency trading strategy.
- Optimization of trading decisions using reinforcement learning.
- Integration with MetaTrader 5 for real-time trading.
- Algorithmic learning using PPO and DQN.
- Profit maximization through adaptive decision-making.

## Dependencies
This project relies on several Python packages to function correctly. You can install the dependencies by running:

```bash
pip install -r requirements.txt
```

The required dependencies are:

- `tianshou` (initially used for RL framework)
- `MetaTrader5` (for connecting to MetaTrader 5 platform)
- `tensorflow` (for building and training RL models)
- `python-dotenv` (for managing environment variables)
- `scikit-learn` (for data preprocessing)

Note: The project initially used `tianshou` as the RL framework but was switched to `stable-baselines3` for enhanced performance and support.

## Project Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/forex-trading-bot.git
   cd forex-trading-bot
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables in a `.env` file:
   ```ini
   MT5_LOGIN=your_mt5_login
   MT5_PASSWORD=your_mt5_password
   MT5_SERVER=your_mt5_server
   ```

4. Run the bot:
   ```bash
   python trading_bot.py
   ```

## How It Works

The bot connects to the MetaTrader 5 platform to retrieve market data and execute trades. The RL algorithms (PPO and DQN) are used to train the bot on historical data, enabling it to make optimized trading decisions. The bot adapts to changing market conditions to maximize profits while minimizing risk.

### Algorithm Overview
- **PPO (Proximal Policy Optimization):** A policy gradient method used for training the bot in a stable and efficient manner.
- **DQN (Deep Q-Network):** A Q-learning-based method used to approximate the Q-value function for better decision-making in high-dimensional environments.

## Project Architecture

![Architecture](https://github.com/user-attachments/assets/9a577efc-3b29-416a-8cf8-a8f9b0b2051f)

### How it works
![how_it_works](https://github.com/user-attachments/assets/7a728f2b-8e2a-49f1-b8a5-400b6a5767c2)
Based on the predicted data, algorithm decides on an order that increases the fitness or the value of the currency distribution.
Buy orders do not have a corresponding sell order.
This can be generalized and extended to trade multiple currencies.


## Contributing

Feel free to fork this repository and contribute to the development of this forex trading bot. Pull requests, issues, and feature suggestions are welcome!
