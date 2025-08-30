import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from src.config import LSTM_HIDDEN_SIZE, FEATURES_DIM

class LSTMFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: dict, features_dim=FEATURES_DIM, lstm_hidden_size=LSTM_HIDDEN_SIZE):
        super(LSTMFeatureExtractor, self).__init__(observation_space, features_dim)
        
        self.bars_count, self.num_pairs = observation_space.spaces["past_prices"].shape
        self.portfolio_dim = observation_space.spaces["portfolio"].shape[0]

        self.lstm = nn.LSTM(
            input_size=self.num_pairs,
            hidden_size=lstm_hidden_size,
            batch_first=True
        )

        self.indicator_dim = 2 * self.num_pairs

        self.portfolio_fc = nn.Sequential(
            nn.Linear(self.portfolio_dim, 32),
            nn.ReLU()
        )
        
        self.combined_fc = nn.Sequential(
            nn.Linear(lstm_hidden_size + self.indicator_dim + 32, features_dim),
            nn.ReLU()
        )

    def compute_ema(self, prices: th.Tensor, span: int) -> th.Tensor:
        alpha = 2.0 / (span + 1)
        batch_size, T, num_pairs = prices.shape
        ema = prices[:, 0, :]  # shape: (batch_size, num_pairs)
        for t in range(1, T):
            ema = alpha * prices[:, t, :] + (1 - alpha) * ema
        return ema

    def compute_macd(self, prices: th.Tensor) -> th.Tensor:
        ema_short = self.compute_ema(prices, span=12)
        ema_long = self.compute_ema(prices, span=26)
        macd = ema_short - ema_long
        return macd

    def compute_rsi(self, prices: th.Tensor, period: int = 14) -> th.Tensor:
        diff = prices[:, 1:, :] - prices[:, :-1, :]
        gain = th.clamp(diff, min=0)
        loss = -th.clamp(diff, max=0)
        batch_size, T_minus_1, num_pairs = gain.shape
        period = min(period, T_minus_1)
        avg_gain = gain[:, :period, :].mean(dim=1)
        avg_loss = loss[:, :period, :].mean(dim=1)
        for t in range(period, T_minus_1):
            current_gain = gain[:, t, :]
            current_loss = loss[:, t, :]
            avg_gain = (avg_gain * (period - 1) + current_gain) / period
            avg_loss = (avg_loss * (period - 1) + current_loss) / period
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def forward(self, observations: dict) -> th.Tensor:
        past_prices = observations["past_prices"]
        price_diffs = past_prices[:, 1:, :] - past_prices[:, :-1, :]
        portfolio = observations["portfolio"]

        _, (h_n, _) = self.lstm(price_diffs)
        lstm_last = h_n.squeeze(0)

        rsi = self.compute_rsi(past_prices, period=14)
        macd = self.compute_macd(past_prices)
        indicators = th.cat([rsi, macd], dim=1)

        portfolio_features = self.portfolio_fc(portfolio)

        combined = th.cat([lstm_last, indicators, portfolio_features], dim=1)
        return self.combined_fc(combined)
