import MetaTrader5 as mt5
import os
from dotenv import load_dotenv
import pandas as pd

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
    def get_account_info(self):
        """Returns the account information of the connected user."""
        return mt5.account_info()

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
        
    def get_intial_data(self, pairs, bars_count):
        """
        Fetches the initial historical data for the given currency pairs.

        :param pairs: List of currency pair symbols (e.g., ('EUR','USD'), ('GBP','USD'), ('JPY,USD')]).
        :param bars_count: Number of past hourly data points to fetch.
        :return: A DataFrame containing the last `bars_count` rows for the given pairs.
        """

        # Dictionary to store data for each pair
        data = {}

        for pair in pairs:
            # Convert tuple to string format (e.g., ('EUR', 'USD') -> "EURUSD")
            symbol = ''.join(pair)

            # Fetch historical rates for the symbol
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, bars_count)
            if rates is None:
                print(f"Failed to fetch data for {symbol}: {mt5.last_error()}")
                continue

            # Convert to DataFrame
            rates_df = pd.DataFrame(rates)

            # Add the close prices to the data dictionary
            data[symbol] = rates_df['close']

        # Convert the data dictionary to a DataFrame
        df = pd.DataFrame(data)

        return df

    def getOpenLots(self, positions):
        open_lots = 0

        for pos in positions:
            open_lots += pos.volume
        return open_lots
    
    def checkOppositeOpenLots(self, symbol, order_type):
        # if not self.connect_mt5():
        #     return False

        positions = mt5.positions_get()
        if positions is None:
            print("No open positions or failed to retrieve positions.")
            return []

        open_lots = 0

        positions = mt5.positions_get(symbol=symbol)
        if positions is not None:
            if order_type == "buy":
                for pos in positions:
                    if (pos.type == mt5.ORDER_TYPE_SELL):
                        open_lots += pos.volume
                # print(f"Open {symbol} Sell Lots: {open_lots}")
            if order_type == "sell":
                for pos in positions:
                    if (pos.type == mt5.ORDER_TYPE_BUY):
                        open_lots += pos.volume
                # print(f"Open {symbol} Buy Lots: {open_lots}")
        
        return open_lots
    
    def checkAllOpenLots(self):
        # if not self.connect_mt5():
        #     return False

        positions = mt5.positions_get()
        if positions is None:
            print("No open positions or failed to retrieve positions.")
            return []
    
        # Extract unique symbols from open positions
        unique_symbols = list(set(pos.symbol for pos in positions))
        for symbol in unique_symbols:
            open_buy_lots = 0
            open_sell_lots = 0

            positions = mt5.positions_get(symbol=symbol)
            if positions is not None:
                for pos in positions:
                    if (pos.type == mt5.ORDER_TYPE_SELL):
                        open_sell_lots += pos.volume
                    if (pos.type == mt5.ORDER_TYPE_BUY):
                        open_buy_lots += pos.volume
            
            if open_buy_lots > 0:
                print(f"Open {symbol} Buy Lots: {open_buy_lots}")
            if open_sell_lots > 0:
                print(f"Open {symbol} Sell Lots: {open_sell_lots}")

    def getPositions(self, symbol, order_type):
        # if not self.connect_mt5():
        #     return None

        positions = mt5.positions_get(symbol=symbol)

        if positions is None or len(positions) == 0:
            print("No open positions found.")
            return None

        # Filter relevant positions
        if order_type == "buy":
            _positions = [pos for pos in positions if pos.type == mt5.ORDER_TYPE_BUY]
        if order_type == "sell":
            _positions = [pos for pos in positions if pos.type == mt5.ORDER_TYPE_SELL]

        if not _positions:
            print(f"No open {symbol, order_type} positions found.")
            return None

        # for pos in _positions:
        #     print(f"Ticket: {pos.ticket}, Volume: {pos.volume}, Open Price: {pos.price_open}")
        sorted_positions = sorted(_positions, key=lambda pos: pos.volume)
        return sorted_positions

    # Function to get a summary of all open positions
    def get_open_positions(self):
        """
        Retrieves and prints a summary of all currently open positions.
        """
        # if not self.connect_mt5():
        #     return

        positions = mt5.positions_get()
        if positions is None or len(positions) == 0:
            print("No open positions.")
            return

        print("\n📊 Open Positions Summary:")
        print("------------------------------------------------------")
        print(f"{'Ticket':<10} {'Symbol':<8} {'Type':<6} {'Volume':<6} {'Price':<10} {'SL':<10} {'TP':<10} {'Profit':<10}")
        print("------------------------------------------------------")
        for pos in positions:
            order_type = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
            print(f"{pos.ticket:<10} {pos.symbol:<8} {order_type:<6} {pos.volume:<6.2f} {pos.price_open:<10.5f} {pos.sl:<10.5f} {pos.tp:<10.5f} {pos.profit:<10.2f}")
        print("------------------------------------------------------") 

    def open_position(self, symbol, lot, order_type, sl_pips=20, tp_pips=50, deviation=10):
        """
        Opens a market order (Buy or Sell) with valid SL/TP.
        """
        # if not self.connect_mt5():
        #     return False

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"Symbol {symbol} not found")
            return False

        tick = mt5.symbol_info_tick(symbol)

        # Convert pips to actual price distance
        point = symbol_info.point
        min_stop_level = symbol_info.trade_stops_level * point
        sl_distance = max(sl_pips * point, min_stop_level)
        tp_distance = max(tp_pips * point, min_stop_level)

        # Set price, SL, TP based on order type
        if order_type.lower() == "buy":
            price = tick.ask
            sl = price - sl_distance
            tp = price + tp_distance
            trade_type = mt5.ORDER_TYPE_BUY
        elif order_type.lower() == "sell":
            price = tick.bid
            sl = price + sl_distance
            tp = price - tp_distance
            trade_type = mt5.ORDER_TYPE_SELL
        else:
            print("Invalid order type. Use 'buy' or 'sell'.")
            return False

        # Create trade request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": trade_type,
            "price": price,
            "sl": round(sl, symbol_info.digits),
            "tp": round(tp, symbol_info.digits),
            "deviation": deviation,
            "magic": 123456,
            "comment": "Opened by Python",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Send trade request
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to open position: {result.comment}")
            return False
        else:
            print(f"Order placed successfully! Order ID: {result.order} (Lot: {lot})")
            return True
        
    def close_position(self, ticket, symbol, volume_to_close, order_type):
        # if not self.connect_mt5():
        #     return False

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"Symbol {symbol} not found")
            return False

        close_price = tick.bid if order_type == mt5.ORDER_TYPE_BUY else tick.ask

        # Validate symbol
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None or not symbol_info.trade_mode:
            print(f"Symbol {symbol} is not tradable.")
            return False

        # Validate lot size
        if volume_to_close <= 0:
            print(f"Invalid lot size: {volume_to_close}")
            return False

        # Validate price
        if close_price <= 0:
            print(f"Invalid closing price: {close_price}")
            return False

        # Check if AutoTrading is enabled
        if not mt5.terminal_info().trade_allowed:
            print("AutoTrading is disabled. Enable it in MT5 settings.")
            return False

        # Create trade request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume_to_close,
            "type": mt5.ORDER_TYPE_SELL if order_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": close_price,
            "deviation": 10,
            "magic": 123456,
            "comment": f"Partially closed ({volume_to_close})",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        # Check if result is None
        if result is None:
            print(f"Order send failed. Check lot size, connection, or logs in MT5.")
            
            # Print MT5 error logs
            error_code, error_desc = mt5.last_error()
            print(f"Last MT5 Error: {error_code} - {error_desc}")
            return False

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to close position {ticket}: {result.comment}")
            return False

        print(f"Closed {volume_to_close} lot(s) of position {ticket} for {symbol}")
        return True
    def is_symbol_available(self, symbol):
        # Ensure MetaTrader 5 is initialized; if not, you may need to call your init function.
        if not mt5.initialize():
            print("MetaTrader5 is not initialized.")
            return False

        # Retrieve all available symbols from MT5.
        all_symbols = mt5.symbols_get()
        if all_symbols is None:
            print("Failed to retrieve symbols from MetaTrader 5.")
            return False

        # Build a set of available symbol names for efficient lookup.
        available_symbols = {s.name for s in all_symbols}

        return symbol in available_symbols
        
    def check_and_trade(self, currency_A, currency_B, trade_amount, sl_pips=20, tp_pips=50):
        """
        Executes a trade by selling currency B to buy currency A.
        Determines the correct MT5 symbol and order type based on available pairs.
        
        Parameters:
            currency_A (str): The currency you want to buy.
            currency_B (str): The currency you are selling.
            trade_amount (float): The amount of currency B to use for the trade.
            sl_pips (int): Stop loss in pips.
            tp_pips (int): Take profit in pips.
        """

        # Establish connection to MT5
        # if not self.connect_mt5():
        #     return False

        LOT_SIZE = 100000  # Define the lot size for the trade

        # Determine the symbol and the corresponding order type.
        # Try the direct pair (A/B) first.
        symbol_direct = f"{currency_A}{currency_B}"
        symbol_inverse = f"{currency_B}{currency_A}"
        
        if self.is_symbol_available(symbol_direct):
            symbol = symbol_direct
            order_type = "buy"  # With direct symbol, a 'buy' means buying A using B.
        elif self.is_symbol_available(symbol_inverse):
            symbol = symbol_inverse
            order_type = "sell"  # With inverse symbol, selling the base (B) will get you A.
        else:
            print(f"Symbol not available in MT5 for pair: {currency_A} and {currency_B}")
            return False

        # Calculate the lot size.
        # Assumes LOT_SIZE is defined elsewhere in your code.
        lot = float(trade_amount / LOT_SIZE)


        # Check if there is an opposite position that needs to be closed.
        if order_type == "buy":
            positions = self.getPositions(symbol, order_type="sell")
        else:
            positions = self.getPositions(symbol, order_type="buy")

        if positions is not None:
            openLotVolume = self.getOpenLots(positions)
            if openLotVolume <= lot:
                lotsToOpen = lot - openLotVolume
                for pos in positions:
                    print(f"🔄 Full close: Closing ticket {pos.ticket}")
                    self.close_position(pos.ticket, pos.symbol, pos.volume, pos.type)
                self.open_position(symbol, round(lotsToOpen, 4), order_type, sl_pips, tp_pips)
                return
            if openLotVolume > lot:
                remainingLotsToClose = lot
                for pos in positions:
                    if pos.volume <= remainingLotsToClose:
                        print(f"🔄 Full close: Closing ticket {pos.ticket}")
                        self.close_position(pos.ticket, pos.symbol, pos.volume, pos.type)
                        remainingLotsToClose -= pos.volume
                        if remainingLotsToClose == 0:
                            return
                    else:
                        print(f"🔄 Partial close: Closing {round(remainingLotsToClose, 4)} lot(s) from ticket {pos.ticket}")
                        return self.close_position(pos.ticket, pos.symbol, round(remainingLotsToClose, 4), pos.type)

        # If no opposite position exists, open a new position.
        return self.open_position(symbol, lot, order_type, sl_pips, tp_pips)

    
    def shutdown(self):
        """Shuts down the MetaTrader 5 connection."""
        mt5.shutdown()
        print("MetaTrader5 shut down.")

# # Example usage
# if __name__ == "__main__":
#     mt5_wrapper = MT5Wrapper(env_file=".env")

#     print(mt5_wrapper.get_terminal_info())
#     print(mt5_wrapper.get_version())

#     mt5_wrapper.place_buy_order("EURUSD", 0.01)

#     mt5_wrapper.shutdown()
