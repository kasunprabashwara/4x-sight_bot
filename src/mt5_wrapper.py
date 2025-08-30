import MetaTrader5 as mt5
import os
from dotenv import load_dotenv
import pandas as pd
import math
import time

class MT5Wrapper:
    # --- __init__ is unchanged from the previous version ---
    def __init__(self, env_file: str = ".env"):
        load_dotenv(env_file)
        self.path = os.getenv("MT5_PATH")
        self.login = int(os.getenv("MT5_LOGIN"))
        self.password = os.getenv("MT5_PASSWORD")
        self.server = os.getenv("MT5_SERVER")
        self.timeout = int(os.getenv("MT5_TIMEOUT", 10000))
        self.portable = os.getenv("MT5_PORTABLE", "True").lower() == "true"
        self.base_currency = None

        if not mt5.initialize(path=self.path, login=self.login, password=self.password,
                              server=self.server, timeout=self.timeout, portable=self.portable):
            raise RuntimeError(f"Failed to initialize MT5: {mt5.last_error()}")
        
        self.base_currency = mt5.account_info().currency
        print(f"MetaTrader5 initialized successfully. Account base currency: {self.base_currency}")

    # --- NEW HELPER METHOD ---
    def _get_filling_mode(self, symbol: str) -> int:
        """
        Intelligently selects a supported filling mode for the given symbol.
        Prefers IOC or FOK, which are standard for market orders.
        """
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            print(f"Warning: Could not get symbol info for {symbol}. Defaulting to ORDER_FILLING_IOC.")
            return mt5.ORDER_FILLING_IOC

        # The 'filling_mode' attribute is a bitmask of all supported modes.
        # We check for the preferred modes using a bitwise AND.
        supported_modes = symbol_info.filling_mode
        
        # Prefer Immediate-Or-Cancel, as it's the most common and flexible.
        if supported_modes & mt5.ORDER_FILLING_IOC:
            return mt5.ORDER_FILLING_IOC
        # Fallback to Fill-Or-Kill if IOC is not supported.
        elif supported_modes & mt5.ORDER_FILLING_FOK:
            return mt5.ORDER_FILLING_FOK
        else:
            # This is a rare case. If neither is supported, the trade will likely fail anyway.
            # We log a warning and return IOC as a last-ditch effort.
            print(f"CRITICAL WARNING: Symbol {symbol} supports neither IOC nor FOK filling modes. Trade may fail.")
            # Let's try to get the first available from the tuple `filling_modes` if it exists
            try:
                # Newer MT5 API versions provide a tuple of supported modes
                return symbol_info.filling_modes[0]
            except (AttributeError, IndexError):
                 # Fallback for older API or empty tuple
                return mt5.ORDER_FILLING_IOC

    # --- METHODS WITH THE FIX APPLIED ---

    def open_position(self, symbol, lot, order_type, sl_pips=20, tp_pips=50, deviation=10):
        final_lot = self._validate_and_adjust_lot(symbol, lot)
        if final_lot is None: return False
        
        # Get the correct filling mode for this symbol
        filling_mode = self._get_filling_mode(symbol) # <-- FIX
        
        price = self.get_symbol_price(symbol, 'ask' if order_type == "buy" else 'bid')
        if price is None: return False
        
        trade_type = mt5.ORDER_TYPE_BUY if order_type == "buy" else mt5.ORDER_TYPE_SELL
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": final_lot,
            "type": trade_type,
            "price": price,
            "deviation": deviation,
            "magic": 123456,
            "comment": "Python Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to open position for {symbol}: {result.comment}")
            return False
        
        print(f"Successfully opened {order_type} position for {symbol} with lot {final_lot}.")
        return True
        
    def close_position(self, position, volume_to_close):
        symbol, ticket, order_type = position.symbol, position.ticket, position.type
        final_volume = self._validate_and_adjust_lot(symbol, volume_to_close)
        if final_volume is None: return False

        # Get the correct filling mode for this symbol
        filling_mode = self._get_filling_mode(symbol) # <-- FIX
        
        close_price = self.get_symbol_price(symbol, 'bid' if order_type == mt5.ORDER_TYPE_BUY else 'ask')
        if close_price is None: return False
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": final_volume,
            "type": mt5.ORDER_TYPE_SELL if order_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": close_price,
            "deviation": 10,
            "magic": 123456,
            "comment": "Close by Python Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode, # <-- FIX
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to close position {ticket}: {result.comment}")
            return False
            
        print(f"Successfully closed {final_volume} lot(s) of position {ticket} for {symbol}")
        return True

    # --- The rest of the file is unchanged from the previous correct version ---
    # (For brevity, only the changed methods and the new helper are shown above,
    # but the full, correct content of the unchanged methods is included below for completeness.)

    def get_price_of(self, currency: str) -> float | None:
        if currency == self.base_currency: return 1.0
        direct_symbol = f"{currency}{self.base_currency}"
        if self.is_symbol_available(direct_symbol): return self.get_symbol_price(direct_symbol, 'ask')
        inverse_symbol = f"{self.base_currency}{currency}"
        if self.is_symbol_available(inverse_symbol):
            price = self.get_symbol_price(inverse_symbol, 'bid')
            return 1.0 / price if price and price != 0 else None
        print(f"Price Error: Could not find a symbol to determine the price of {currency} in {self.base_currency}.")
        return None

    def check_and_trade(self, currency_A, currency_B, trade_value_in_base_currency, sl_pips=20, tp_pips=50):
        symbol_direct, symbol_inverse = f"{currency_A}{currency_B}", f"{currency_B}{currency_A}"
        if self.is_symbol_available(symbol_direct): symbol, order_type = symbol_direct, "buy"
        elif self.is_symbol_available(symbol_inverse): symbol, order_type = symbol_inverse, "sell"
        else: print(f"Execution Error: No tradable symbol for {currency_A}/{currency_B}"); return

        symbol_info = mt5.symbol_info(symbol); contract_size = symbol_info.trade_contract_size
        if not symbol_info or contract_size == 0: return

        symbol_base_currency = symbol_info.currency_base
        price_of_symbol_base_in_account_base = self.get_price_of(symbol_base_currency)
        if price_of_symbol_base_in_account_base is None:
            print(f"Lot Calc Error: Could not get price of '{symbol_base_currency}'."); return

        volume_in_symbol_base = trade_value_in_base_currency / price_of_symbol_base_in_account_base
        calculated_lot = volume_in_symbol_base / contract_size

        opposite_order_type = "sell" if order_type == "buy" else "buy"
        opposite_positions = self.get_open_positions(symbol, opposite_order_type)
        if opposite_positions:
            print(f"Found {len(opposite_positions)} opposing trade(s) for {symbol}. Closing them.")
            for pos in opposite_positions: self.close_position(pos, pos.volume)

        print(f"Executing trade: {order_type.upper()} {symbol} with calculated lot: {calculated_lot:.4f}")
        self.open_position(symbol, calculated_lot, order_type, sl_pips, tp_pips)

    def _validate_and_adjust_lot(self, symbol: str, input_lot: float):
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            print(f"Symbol {symbol}: Failed to get symbol info.")
            return None
        volume_min, volume_max, volume_step = symbol_info.volume_min, symbol_info.volume_max, symbol_info.volume_step
        if input_lot < volume_min:
            print(f"Symbol {symbol}: Lot {input_lot:.5f} < min_volume {volume_min:.5f}. Aborted.")
            return None
        lot = min(input_lot, volume_max) if volume_max > 0 else input_lot
        if volume_step > 0:
            lot = round(lot / volume_step) * volume_step
        decimals = 0
        if '.' in str(volume_step):
            decimals = len(str(volume_step).split('.')[1])
        lot = round(lot, decimals)
        if lot < volume_min:
            print(f"Symbol {symbol}: Adjusted lot {lot:.5f} < min_volume {volume_min:.5f}. Aborted.")
            return None
        if lot <= 0:
            print(f"Symbol {symbol}: Final lot is zero or negative.")
            return None
        print(f"Symbol {symbol}: Input Lot: {input_lot:.5f}, Validated & Adjusted Lot: {lot:.5f}")
        return lot

    def get_open_positions(self, symbol: str, order_type: str):
        positions = mt5.positions_get(symbol=symbol)
        if positions is None: return []
        mt5_order_type = mt5.ORDER_TYPE_BUY if order_type == "buy" else mt5.ORDER_TYPE_SELL
        return [p for p in positions if p.type == mt5_order_type]

    def get_account_info(self): return mt5.account_info()
    def get_symbol_price(self, symbol: str, price_type='ask'):
        tick = mt5.symbol_info_tick(symbol)
        if not tick: return None
        return tick.ask if price_type == 'ask' else tick.bid
    def get_initial_data(self, symbols: list, bars_count: int, retries=3, delay=2):
        data = {}
        for symbol in symbols:
            for i in range(retries):
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, bars_count)
                if rates is not None and len(rates) > 0: data[symbol] = pd.DataFrame(rates)['close']; break
                else: time.sleep(delay)
            else: print(f"FATAL: Failed to fetch data for {symbol}."); return None
        return pd.DataFrame(data)
    def is_symbol_available(self, symbol):
        info = mt5.symbol_info(symbol)
        return info is not None and mt5.symbol_select(symbol, True)
    def shutdown(self):
        mt5.shutdown()
        print("MetaTrader5 shut down.")