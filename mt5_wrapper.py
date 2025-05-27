import MetaTrader5 as mt5
import os
from dotenv import load_dotenv
import pandas as pd
import math

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

    def _validate_and_adjust_lot(self, symbol: str, input_lot: float):
        original_input_lot = input_lot

        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            print(f"Symbol {symbol}: Failed to get symbol info in _validate_and_adjust_lot.")
            return None

        volume_min = symbol_info.volume_min
        volume_max = symbol_info.volume_max
        volume_step = symbol_info.volume_step
        
        current_lot = input_lot

        if current_lot <= 0:
            if volume_min > 0: 
                print(f"Symbol {symbol}: Input lot {current_lot:.5f} is zero or negative, but min volume is {volume_min:.5f}. Cannot trade.")
                return None
            elif volume_min == 0 and current_lot < 0:
                print(f"Symbol {symbol}: Input lot {current_lot:.5f} is negative, min volume is 0. Cannot trade negative lot.")
                return None

        if current_lot < volume_min:
            print(f"Symbol {symbol}: Input lot {current_lot:.5f} is less than min_volume {volume_min:.5f}. Adjusting to min_volume.")
            current_lot = volume_min
        
        if volume_max > 0 and current_lot > volume_max:
            print(f"Symbol {symbol}: Lot {current_lot:.5f} exceeds max_volume {volume_max:.5f}. Clamping to max_volume.")
            current_lot = volume_max
            if current_lot < volume_min:
                print(f"Symbol {symbol}: Clamped lot {current_lot:.5f} (max) is now less than min_volume {volume_min:.5f}. Invalid setup.")
                return None
        
        if volume_step > 0:
            adjusted_lot = math.floor(current_lot / volume_step) * volume_step
            if adjusted_lot < volume_min and volume_min > 0:
                print(f"Symbol {symbol}: Lot {current_lot:.5f} after step adjustment became {adjusted_lot:.5f}, which is < min_volume {volume_min:.5f}.")
                smallest_stepped_min_lot = math.ceil(volume_min / volume_step) * volume_step
                if current_lot >= smallest_stepped_min_lot:
                    current_lot = adjusted_lot
                    if current_lot < volume_min :
                         print(f"Symbol {symbol}: Adjusted lot {current_lot:.5f} still below min {volume_min:.5f}. Cannot trade this volume.")
                         return None
                else:
                    print(f"Symbol {symbol}: Original lot {current_lot:.5f} too small for smallest valid step {smallest_stepped_min_lot:.5f} >= min {volume_min:.5f}.")
                    return None
            else:
                current_lot = adjusted_lot

        if volume_step > 0:
            str_step = str(volume_step)
            if '.' in str_step:
                decimals = len(str_step.split('.')[1])
                current_lot = round(current_lot, decimals)
            else:
                current_lot = round(current_lot, 0)
        else:
            current_lot = round(current_lot, 2) 

        if current_lot < volume_min and volume_min > 0:
            print(f"Symbol {symbol}: Final lot {current_lot:.5f} is below min_volume {volume_min:.5f} after all adjustments.")
            return None
        
        if volume_max > 0 and current_lot > volume_max:
            print(f"Symbol {symbol}: Final lot {current_lot:.5f} is above max_volume {volume_max:.5f} (Error in logic).")
            return None

        if current_lot == 0:
            print(f"Symbol {symbol}: Final lot is 0. This is not tradable.")
            return None

        print(f"Symbol {symbol}: Input Lot: {original_input_lot:.5f}, Validated & Adjusted Lot: {current_lot:.5f} (Min: {volume_min:.5f}, Max: {volume_max:.5f}, Step: {volume_step:.5f})")
        return current_lot

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

        # Validate and adjust lot size
        final_lot = self._validate_and_adjust_lot(symbol, lot)
        if final_lot is None:
            print(f"Symbol {symbol}: Lot validation failed for requested lot {lot:.5f}. Cannot open position.")
            return False

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"Symbol {symbol}: Could not retrieve tick data. Cannot open position.")
            return False

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
            "volume": final_lot,
            "type": trade_type,
            "price": price,
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
            print(f"Order placed successfully! Order ID: {result.order} (Lot: {final_lot})")
            return True
        
    def close_position(self, ticket, symbol, volume_to_close, order_type):
        # if not self.connect_mt5():
        #     return False

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"Symbol {symbol} not found in close_position for validation")
            return False
        
        # Validate and adjust volume_to_close
        final_volume_to_close = self._validate_and_adjust_lot(symbol, volume_to_close)
        if final_volume_to_close is None:
            print(f"Symbol {symbol}: Volume validation failed for closing {volume_to_close:.5f} lots. Cannot close position.")
            return False

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"Symbol {symbol} not found or tick data unavailable for closing.")
            return False

        close_price = tick.bid if order_type == mt5.ORDER_TYPE_BUY else tick.ask

        # Validate symbol
        if symbol_info is None or not symbol_info.trade_mode:
            print(f"Symbol {symbol} is not tradable.")
            return False

        # Check if AutoTrading is enabled
        if not mt5.terminal_info().trade_allowed:
            print("AutoTrading is disabled. Enable it in MT5 settings.")
            return False

        # Create trade request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": final_volume_to_close,
            "type": mt5.ORDER_TYPE_SELL if order_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": close_price,
            "deviation": 10,
            "magic": 123456,
            "comment": f"Partially closed ({final_volume_to_close})",
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

        print(f"Closed {final_volume_to_close} lot(s) of position {ticket} for {symbol}")
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
            trade_amount (float): The amount of currency B to use for the trade. Assumed to be in units of currency_B.
            sl_pips (int): Stop loss in pips.
            tp_pips (int): Take profit in pips.
        """

        # Establish connection to MT5
        # if not self.connect_mt5():
        #     return False

        # Determine the symbol and the corresponding order type.
        symbol_direct = f"{currency_A}{currency_B}"
        symbol_inverse = f"{currency_B}{currency_A}"
        order_type = None
        symbol = None
        
        if self.is_symbol_available(symbol_direct):
            symbol = symbol_direct
            order_type = "buy"
        elif self.is_symbol_available(symbol_inverse):
            symbol = symbol_inverse
            order_type = "sell"
        else:
            print(f"Symbol not available in MT5 for pair: {currency_A} and {currency_B}")
            return False

        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            print(f"Could not get symbol info for {symbol} in check_and_trade")
            return False
        
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            print(f"Could not get tick for {symbol} in check_and_trade")
            return False

        # Calculate the lot size.
        # Assumes trade_amount is the amount of currency_B to use for the trade.
        calculated_lot = 0.0
        contract_size = symbol_info.trade_contract_size
        if contract_size == 0:
            print(f"Symbol {symbol}: Contract size is zero. Cannot calculate lot.")
            return False

        if order_type == "buy":  # Buying currency_A with currency_B. Symbol is A/B (e.g., EURUSD). trade_amount is in B (USD).
            price = tick.ask
            if price == 0:
                print(f"Symbol {symbol}: Ask price is zero. Cannot calculate lot.")
                return False
            calculated_lot = trade_amount / (price * contract_size)
        elif order_type == "sell":  # Selling currency_B for currency_A. Symbol is B/A (e.g., USDEUR). trade_amount is in B (USD).
            # Price of B/A is A per B. Contract size is in B.
            calculated_lot = trade_amount / contract_size
        
        if calculated_lot <= 0: # If trade_amount is too small or results in zero/negative lot
            print(f"Symbol {symbol}: Calculated lot is {calculated_lot:.5f} from trade_amount {trade_amount:.2f} {currency_B}. Skipping trade.")
            return False
            
        # Lot validation will occur within open_position and close_position.

        # Check if there is an opposite position that needs to be closed.
        if order_type == "buy":
            positions = self.getPositions(symbol, order_type="sell")
        else:
            positions = self.getPositions(symbol, order_type="buy")

        if positions is not None:
            openLotVolume = self.getOpenLots(positions)
            if openLotVolume < calculated_lot: # Net effect is to open more or a new position
                lotsToOpen = calculated_lot - openLotVolume
                for pos in positions:
                    print(f"🔄 Full close (before opening new/increasing): Closing ticket {pos.ticket} vol {pos.volume}")
                    self.close_position(pos.ticket, pos.symbol, pos.volume, pos.type) 
                
                print(f"Attempting to open new/additional position for {symbol} with calculated_lot: {lotsToOpen:.5f}")
                return self.open_position(symbol, lotsToOpen, order_type, sl_pips, tp_pips)
            
            elif openLotVolume > calculated_lot: # Net effect is to reduce existing opposite position
                lots_to_effectively_close = calculated_lot 

                for pos in positions:
                    if lots_to_effectively_close <= 1e-8: break # All necessary volume closed (using small epsilon for float comparison)

                    volume_this_pos_can_close = min(pos.volume, lots_to_effectively_close)
                    if volume_this_pos_can_close > 1e-8: # Only close if significant
                        print(f"🔄 Partial/Full close (reducing opposite): Closing {volume_this_pos_can_close:.5f} lot(s) from ticket {pos.ticket}")
                        closed_successfully = self.close_position(pos.ticket, pos.symbol, volume_this_pos_can_close, pos.type)
                        if closed_successfully:
                             lots_to_effectively_close -= volume_this_pos_can_close 
                        else:
                             print(f"Failed to close {volume_this_pos_can_close:.5f} from ticket {pos.ticket}. Aborting further closes for this trade action.")
                             return False 
                return True 
            
            else: # openLotVolume is very close to calculated_lot. Close all opposite.
                for pos in positions:
                    print(f"🔄 Full close (exact match): Closing ticket {pos.ticket} vol {pos.volume}")
                    self.close_position(pos.ticket, pos.symbol, pos.volume, pos.type)
                return True

        # If no opposite position exists, open a new position.
        print(f"No opposite positions. Attempting to open new position for {symbol} with calculated_lot: {calculated_lot:.5f}")
        return self.open_position(symbol, calculated_lot, order_type, sl_pips, tp_pips)

    
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
