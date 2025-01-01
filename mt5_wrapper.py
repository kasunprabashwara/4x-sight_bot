import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

class MT5Wrapper:
    def __init__(self, env_file: str):
        """
        Initializes the MT5 instance using credentials from the .env file.
        
        :param env_file: Path to the .env file containing MT5 credentials.
        """
        # Load environment variables
        load_dotenv(env_file)
        
        self.path = os.getenv("MT5_PATH")
        self.login = int(os.getenv("MT5_LOGIN"))
        self.password = os.getenv("MT5_PASSWORD")
        self.server = os.getenv("MT5_SERVER")
        self.timeout = int(os.getenv("MT5_TIMEOUT", 10000))
        self.portable = os.getenv("MT5_PORTABLE", "True").lower() == "true"

        # Initialize MetaTrader 5
        if not mt5.initialize(path=self.path, login=self.login, password=self.password,
                              server=self.server, timeout=self.timeout, portable=self.portable):
            raise RuntimeError(f"Failed to initialize MT5: {mt5.last_error()}")
        
        print("MetaTrader5 initialized successfully.")

    def get_terminal_info(self):
        """Returns MetaTrader 5 terminal information."""
        return mt5.terminal_info()

    def get_version(self):
        """Returns MetaTrader 5 version information."""
        return mt5.version()

    def get_symbol_price(self, symbol: str):
        """
        Returns the current ask price of the given symbol.

        :param symbol: The symbol to retrieve the price for (e.g., "EURUSD").
        :return: Current ask price or None if retrieval fails.
        """
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            return tick.ask
        else:
            print(f"Failed to get price for {symbol}: {mt5.last_error()}")
            return None

    def place_buy_order(self, symbol: str, volume: float):
        """
        Places a buy order for the given symbol and volume.

        :param symbol: The symbol to trade (e.g., "EURUSD").
        :param volume: The lot size for the trade.
        """
        price = self.get_symbol_price(symbol)
        if price is None:
            print("Buy order aborted: Unable to retrieve price.")
            return

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "Python buy order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to place buy order: {result.comment}")
        else:
            print(f"Buy order placed successfully: {result}")

    def place_sell_order(self, symbol: str, volume: float):
        """
        Places a sell order for the given symbol and volume.

        :param symbol: The symbol to trade (e.g., "EURUSD").
        :param volume: The lot size for the trade.
        """
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            print(f"Sell order aborted: Unable to retrieve price for {symbol}.")
            return

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_SELL,
            "price": tick.bid,
            "deviation": 20,
            "magic": 234000,
            "comment": "Python sell order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to place sell order: {result.comment}")
        else:
            print(f"Sell order placed successfully: {result}")

    def shutdown(self):
        """Shuts down the MetaTrader 5 connection."""
        mt5.shutdown()
        print("MetaTrader5 shut down.")

# Example usage
if __name__ == "__main__":
    mt5_wrapper = MT5Wrapper(env_file=".env")

    print(mt5_wrapper.get_terminal_info())
    print(mt5_wrapper.get_version())

    mt5_wrapper.place_buy_order("EURUSD", 0.01)

    mt5_wrapper.shutdown()
