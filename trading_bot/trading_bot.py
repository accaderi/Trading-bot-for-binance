import websocket
import json
import datetime
import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
import os
from threading import Thread
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
import threading
import time
import logging
import traceback
from datetime import datetime, timedelta, timezone
from retrying import retry

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TradingBot(Thread):
    def __init__(self, symbol, interval, ema_slow, ema_fast, adx_period, adx_threshold, chop_period, chop_threshold, initial_trade_percentage, api_key, api_secret):
        super().__init__()
        self.stop_event = threading.Event()
        self.symbol = symbol
        self.interval = interval
        self.ema_slow = ema_slow
        self.ema_fast = ema_fast
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.chop_period = chop_period
        self.chop_threshold = chop_threshold
        self.initial_trade_percentage = initial_trade_percentage
        self.initial_quote_balance = None
        self.money_to_keep = None
        self.last_buy_quantity = None
        self.last_trade_side = None  # New variable to track the last trade side
        self.client = Client(api_key, api_secret) # testnet=True to be added if testing on testnet
        self.base_asset, self.quote_asset = self.get_assets_from_symbol(symbol)
        self.df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.df_orders = pd.DataFrame(columns=[
            'index', 'timestamp', 'side', 'symbol', 'quantity', 'price',
            f'Total_gain_loss_{self.quote_asset}', 'Total_gain_loss_percent',
            f'Last_trade_gain_loss_{self.quote_asset}', 'Last_trade_gain_loss_percent',
            f'{self.quote_asset}_balance', f'{self.base_asset}_balance',
            'status', 'buy/sell', 'remark'
        ])
        self.previous_df_orders = pd.DataFrame()
        self.ws = None
        self.trade_index = 0
        # New instance variables for tracking gains and prices
        self.total_gain_loss = 0
        self.previous_price = 0
        self.previous_quantity = 0
        self.setup()
        
    def setup(self):
        """
        Sets up the bot by creating a directory for output files and
        setting up the min, max, and step size for quantity of the
        trading symbol.
        """
        
        current_dir = os.getcwd()
        self.data_dir = os.path.join(current_dir, 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        self.filename = os.path.join(self.data_dir, f'orders_{self.symbol.replace("/", "_")}_{self.interval}_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.csv')

        symbol_filters = self.get_symbol_info(self.symbol)
        lot_size_filter = symbol_filters['LOT_SIZE']
        self.min_qty = float(lot_size_filter['minQty'])
        self.max_qty = float(lot_size_filter['maxQty'])
        self.step_size = float(lot_size_filter['stepSize'])

        # Initialize initial balance and money to keep
        account_info = self.client.get_account()
        self.initial_quote_balance = float(next(asset['free'] for asset in account_info['balances'] if asset['asset'] == self.quote_asset))
        self.initial_trade_amount = self.initial_quote_balance * (self.initial_trade_percentage / 100)
        self.money_to_keep = self.initial_quote_balance * (1 - self.initial_trade_percentage / 100)

    def run(self):
        while not self.stop_event.is_set():
            try:
                socket = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@kline_{self.interval}"
                logger.info(f"Connecting to WebSocket: {socket}")
                self.ws = websocket.WebSocketApp(socket,
                                                 on_message=self.on_message,
                                                 on_error=self.on_error,
                                                 on_close=self.on_close,
                                                 on_open=self.on_open)
                logger.debug(f"WebSocket object created: {self.ws}")
                self.ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                logger.error(f"Exception in run method: {str(e)}")
                logger.error(traceback.format_exc())
                self.send_update(f"WebSocket error: {str(e)}")
                if not self.stop_event.is_set():
                    time.sleep(5)  # Wait before reconnecting
                else:
                    break  # Exit the loop if stop_event is set
        logger.info("TradingBot run method exited")

    def get_symbol_info(self, symbol):
        info = self.client.get_symbol_info(symbol)
        filters = {item['filterType']: item for item in info['filters']}
        return filters

    def get_assets_from_symbol(self, symbol):
        info = self.client.get_symbol_info(symbol)
        return info['baseAsset'], info['quoteAsset']

    def round_step_size(self, quantity, step_size):
        return float(Decimal(str(quantity)).quantize(Decimal(str(step_size)), rounding=ROUND_DOWN))

    def round_price(self, price):
        return float(Decimal(str(price)).quantize(Decimal('0.10'), rounding=ROUND_HALF_UP))

    def calculate_ema(self, current_price, previous_ema, span):
        smoothing = 2 / (span + 1)
        return (current_price * smoothing) + (previous_ema * (1 - smoothing))
    
    def calculate_adx(self, adx_period):
        """Calculates ADX indicators."""
        if len(self.df) < 2 * adx_period:
            return

        # Check if ADX column exists
        if 'ADX' not in self.df.columns:
            # Initial calculation for the entire DataFrame
            # Calculate True Range (TR)
            self.df['TR'] = np.maximum(
                self.df['high'] - self.df['low'],
                np.maximum(
                    abs(self.df['high'] - self.df['close'].shift(1)),
                    abs(self.df['low'] - self.df['close'].shift(1))
                )
            )

            # Calculate Directional Movement (DM)
            self.df['+DM'] = np.where(
                (self.df['high'] - self.df['high'].shift(1)) > (self.df['low'].shift(1) - self.df['low']),
                np.maximum(self.df['high'] - self.df['high'].shift(1), 0),
                0
            )
            self.df['-DM'] = np.where(
                (self.df['low'].shift(1) - self.df['low']) > (self.df['high'] - self.df['high'].shift(1)),
                np.maximum(self.df['low'].shift(1) - self.df['low'], 0),
                0
            )

            # Calculate smoothed TR and DM
            self.df['ATR'] = self.df['TR'].ewm(span=adx_period, adjust=False).mean()
            self.df['+DMI'] = self.df['+DM'].ewm(span=adx_period, adjust=False).mean()
            self.df['-DMI'] = self.df['-DM'].ewm(span=adx_period, adjust=False).mean()

            # Calculate DI+ and DI-
            self.df['+DI'] = 100 * self.df['+DMI'] / self.df['ATR']
            self.df['-DI'] = 100 * self.df['-DMI'] / self.df['ATR']

            # Calculate DX
            self.df['DX'] = 100 * abs(self.df['+DI'] - self.df['-DI']) / (self.df['+DI'] + self.df['-DI'])

            # Calculate ADX
            self.df['ADX'] = self.df['DX'].ewm(span=adx_period, adjust=False).mean()

            # Handle inf and NaN values
            self.df['ADX'] = self.df['ADX'].replace([np.inf, -np.inf], np.nan)
            self.df['ADX'] = self.df['ADX'].fillna(method='ffill')  # Forward fill NaN values

            # Only keep ADX values where we have enough data
            self.df.loc[:2*adx_period-1, 'ADX'] = np.nan

            logger.info("Initial ADX calculation completed")
        else:
            # Calculate only for the last row
            last_index = len(self.df) - 1

            # Calculate TR for the last row
            self.df.loc[last_index, 'TR'] = max(
                self.df.loc[last_index, 'high'] - self.df.loc[last_index, 'low'],
                abs(self.df.loc[last_index, 'high'] - self.df.loc[last_index-1, 'close']),
                abs(self.df.loc[last_index, 'low'] - self.df.loc[last_index-1, 'close'])
            )

            # Calculate DM for the last row
            plus_dm = max(self.df.loc[last_index, 'high'] - self.df.loc[last_index-1, 'high'], 0) if self.df.loc[last_index, 'high'] - self.df.loc[last_index-1, 'high'] > self.df.loc[last_index-1, 'low'] - self.df.loc[last_index, 'low'] else 0
            minus_dm = max(self.df.loc[last_index-1, 'low'] - self.df.loc[last_index, 'low'], 0) if self.df.loc[last_index-1, 'low'] - self.df.loc[last_index, 'low'] > self.df.loc[last_index, 'high'] - self.df.loc[last_index-1, 'high'] else 0

            # Update smoothed values
            alpha = 1 / adx_period
            self.df.loc[last_index, 'ATR'] = (1 - alpha) * self.df.loc[last_index-1, 'ATR'] + alpha * self.df.loc[last_index, 'TR']
            self.df.loc[last_index, '+DMI'] = (1 - alpha) * self.df.loc[last_index-1, '+DMI'] + alpha * plus_dm
            self.df.loc[last_index, '-DMI'] = (1 - alpha) * self.df.loc[last_index-1, '-DMI'] + alpha * minus_dm

            # Calculate DI+ and DI- for the last row
            self.df.loc[last_index, '+DI'] = 100 * self.df.loc[last_index, '+DMI'] / self.df.loc[last_index, 'ATR']
            self.df.loc[last_index, '-DI'] = 100 * self.df.loc[last_index, '-DMI'] / self.df.loc[last_index, 'ATR']

            # Calculate DX for the last row
            self.df.loc[last_index, 'DX'] = 100 * abs(self.df.loc[last_index, '+DI'] - self.df.loc[last_index, '-DI']) / (self.df.loc[last_index, '+DI'] + self.df.loc[last_index, '-DI'])

            # Update ADX for the last row
            self.df.loc[last_index, 'ADX'] = (1 - alpha) * self.df.loc[last_index-1, 'ADX'] + alpha * self.df.loc[last_index, 'DX']

            # Handle inf and NaN values for the last row
            if np.isinf(self.df.loc[last_index, 'ADX']) or np.isnan(self.df.loc[last_index, 'ADX']):
                self.df.loc[last_index, 'ADX'] = self.df.loc[last_index-1, 'ADX']

            # logger.info("ADX calculation updated for the last row")

        # logger.debug(f"Last ADX value: {self.df['ADX'].iloc[-1]}")

    def calculate_choppiness(self, chop_period):
        """Calculates Choppiness Index."""
        if len(self.df) < chop_period:
            logger.warning(f"Not enough data to calculate Choppiness. Need at least {chop_period} rows, have {len(self.df)}.")
            return

        # Check if Choppiness column exists
        if 'Choppiness' not in self.df.columns:
            # Initial calculation for the entire DataFrame
            # Ensure TR is calculated
            if 'TR' not in self.df.columns:
                self.df['TR'] = np.maximum(
                    self.df['high'] - self.df['low'],
                    np.maximum(
                        abs(self.df['high'] - self.df['close'].shift(1)),
                        abs(self.df['low'] - self.df['close'].shift(1))
                    )
                )

            # Calculate TR sum for the chop_period
            tr_sum = self.df['TR'].rolling(window=chop_period).sum()

            # Calculate Highest High and Lowest Low over the chop_period
            highest_high = self.df['high'].rolling(window=chop_period).max()
            lowest_low = self.df['low'].rolling(window=chop_period).min()

            # Calculate Choppiness Index
            range_high_low = highest_high - lowest_low
            self.df['Choppiness'] = 100 * np.log10(tr_sum / range_high_low) / np.log10(chop_period)

            # Handle inf and NaN values
            self.df['Choppiness'].replace([np.inf, -np.inf], np.nan, inplace=True)
            self.df['Choppiness'].fillna(method='ffill', inplace=True)

            # Only keep Choppiness values where we have enough data
            self.df.loc[:chop_period-1, 'Choppiness'] = np.nan

            logger.info("Initial Choppiness calculation completed")
        else:
            # Calculate only for the last row
            last_index = len(self.df) - 1

            # Ensure TR is calculated for the last row
            if 'TR' not in self.df.columns:
                self.df.loc[last_index, 'TR'] = max(
                    self.df.loc[last_index, 'high'] - self.df.loc[last_index, 'low'],
                    abs(self.df.loc[last_index, 'high'] - self.df.loc[last_index-1, 'close']),
                    abs(self.df.loc[last_index, 'low'] - self.df.loc[last_index-1, 'close'])
                )

            # Calculate TR sum for the last chop_period
            tr_sum = self.df['TR'].iloc[-chop_period:].sum()

            # Calculate Highest High and Lowest Low over the last chop_period
            highest_high = self.df['high'].iloc[-chop_period:].max()
            lowest_low = self.df['low'].iloc[-chop_period:].min()

            # Calculate Choppiness Index for the last row
            range_high_low = highest_high - lowest_low
            if range_high_low != 0:
                choppiness = 100 * np.log10(tr_sum / range_high_low) / np.log10(chop_period)
            else:
                choppiness = np.nan

            self.df.loc[last_index, 'Choppiness'] = choppiness

            # Handle inf and NaN values for the last row
            if np.isinf(choppiness) or np.isnan(choppiness):
                self.df.loc[last_index, 'Choppiness'] = self.df.loc[last_index-1, 'Choppiness']

            # logger.info("Choppiness calculation updated for the last row")

        # logger.debug(f"Last Choppiness value: {self.df['Choppiness'].iloc[-1]}")
    

    def create_order_df(self, timestamp, side, symbol, quantity, price, quote_balance, base_balance, status, buy_sell, remark):
        self.trade_index += 1

        last_trade_gain_loss = 0
        last_trade_gain_loss_percent = 0

        if status == 'FILLED':
            if self.trade_index == 1:
                if side == 'SELL':
                    self.initial_quote_balance = quote_balance
            else:
                if side == 'BUY':
                    last_trade_gain_loss = (self.previous_price - price) * quantity
                else:  # SELL
                    last_trade_gain_loss = (price - self.previous_price) * self.previous_quantity
                
                last_trade_gain_loss_percent = (last_trade_gain_loss / (self.previous_price * self.previous_quantity)) * 100 if self.previous_price and self.previous_quantity else 0

                self.total_gain_loss += last_trade_gain_loss

            # Update previous price and quantity for next trade
            self.previous_price = price
            self.previous_quantity = quantity
            self.last_trade_side = side

        total_gain_loss_percent = (self.total_gain_loss / self.initial_trade_amount) * 100 if self.initial_trade_amount else 0

        new_order = pd.DataFrame({
            'index': [self.trade_index],
            'timestamp': [timestamp],
            'side': [side],
            'symbol': [symbol],
            'quantity': [quantity],
            'price': [price],
            f'{self.quote_asset}_balance': [quote_balance],
            f'{self.base_asset}_balance': [base_balance],
            'status': [status],
            'buy/sell': [buy_sell],
            f'Total_gain_loss_{self.quote_asset}': [self.total_gain_loss],
            'Total_gain_loss_percent': [total_gain_loss_percent],
            f'Last_trade_gain_loss_{self.quote_asset}': [last_trade_gain_loss],
            'Last_trade_gain_loss_percent': [last_trade_gain_loss_percent],
            'remark': [remark]
        })

        self.df_orders = pd.concat([self.df_orders, new_order], ignore_index=True)
        self.save_df_orders()
        return new_order

    def save_df_orders(self):
        self.df_orders.to_csv(self.filename, index=False)
        self.send_update(f"Orders saved to {self.filename}")

    def execute_trade(self, side, symbol, price):
        # Ensure we're not executing the same type of trade twice in a row
        if side == self.last_trade_side:
            self.send_update(f"Skipping {side} trade as it's the same as the last trade.")
            return

        try:
            account_info = self.client.get_account()
            quote_balance = float(next(asset['free'] for asset in account_info['balances'] if asset['asset'] == self.quote_asset))
            base_balance = float(next(asset['free'] for asset in account_info['balances'] if asset['asset'] == self.base_asset))

            if self.last_trade_side is None:  # First trade
                if side == 'BUY':
                    trade_amount = self.initial_quote_balance * (self.initial_trade_percentage / 100)
                    quantity = trade_amount / price
                else:  # SELL
                    trade_amount = self.initial_quote_balance * (self.initial_trade_percentage / 100)
                    quantity = trade_amount / price
                    if quantity > base_balance:
                        quantity = base_balance
            else:  # Subsequent trades
                if side == 'SELL':
                    quantity = self.last_buy_quantity
                    if quantity > base_balance:
                        quantity = base_balance
                else:  # BUY
                    trade_amount = quote_balance - self.money_to_keep
                    if trade_amount <= 0:
                        self.send_update("Insufficient funds to continue trading. Stopping bot.")
                        self.stop()
                        return
                    quantity = trade_amount / price

            rounded_quantity = self.round_step_size(quantity, self.step_size)
            if rounded_quantity < self.min_qty:
                self.send_update(f"Quantity {rounded_quantity} is less than minimum {self.min_qty}. Order not placed.")
                return
            if rounded_quantity > self.max_qty:
                rounded_quantity = self.max_qty
                self.send_update(f"Quantity adjusted to maximum {self.max_qty}.")
            
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type=Client.ORDER_TYPE_MARKET,
                quantity=rounded_quantity
            )
            self.send_update(f'{side} order: {rounded_quantity}')
            
            # Update balances after trade
            account_info = self.client.get_account()
            quote_balance = float(next(asset['free'] for asset in account_info['balances'] if asset['asset'] == self.quote_asset))
            base_balance = float(next(asset['free'] for asset in account_info['balances'] if asset['asset'] == self.base_asset))
            
            if order['status'] == 'FILLED':
                if side == 'BUY':
                    self.last_buy_quantity = rounded_quantity

            new_order = self.create_order_df(
                pd.Timestamp.now(),
                side,
                symbol,
                rounded_quantity,
                price,
                quote_balance,
                base_balance,
                order['status'],
                f"{side} {Client.ORDER_TYPE_MARKET}",
                ''
            )
            self.send_update(f"{side} {Client.ORDER_TYPE_MARKET} order executed: {order}")

        except BinanceAPIException as e:
            self.send_update(f"An API error occurred: {e}")
            new_order = self.create_order_df(
                pd.Timestamp.now(),
                side,
                symbol,
                rounded_quantity,
                price,
                quote_balance,
                base_balance,
                'FAILED',
                f"{side} {Client.ORDER_TYPE_MARKET}",
                f"Insufficient balance. Balance: {quote_balance if side == 'BUY' else base_balance}, Order amount: {rounded_quantity * price if side == 'BUY' else rounded_quantity}"
            )
        except Exception as e:
            self.send_update(f"An error occurred: {e}")

    def on_message(self, ws, message):
        if self.stop_event.is_set():
            return
        # logger.info(f"Received message: {message[:100]}...")  # Log first 100 chars of the message
        
        data = json.loads(message)
        candle = data['k']
        
        is_candle_closed = candle['x']
        if is_candle_closed:
            new_row = pd.DataFrame({
                'timestamp': [pd.to_datetime(candle['t'], unit='ms', utc=True)],
                'open': [float(candle['o'])],
                'high': [float(candle['h'])],
                'low': [float(candle['l'])],
                'close': [float(candle['c'])],
                'volume': [float(candle['v'])]
            })
            
            # Append new row to the DataFrame
            self.df = pd.concat([self.df, new_row], ignore_index=True)
            
            # Recalculate indicators
            self.calculate_indicators()
            
            # Keep only the necessary amount of historical data
            max_period = max(self.ema_slow, self.ema_fast, self.adx_period * 2, self.chop_period)
            self.df = self.df.tail(max_period)
            
            current_price = float(candle['c'])
            
            # Calculate buy/sell signals
            ema_diff = self.df['ema_fast'].iloc[-1] - self.df['ema_slow'].iloc[-1]
            ema_diff_prev = self.df['ema_fast'].iloc[-2] - self.df['ema_slow'].iloc[-2]
            
            ema_buy_signal = (ema_diff > 0) and (ema_diff_prev <= 0)
            ema_sell_signal = (ema_diff < 0) and (ema_diff_prev >= 0)

            if self.adx_period and self.adx_threshold:
                adx_condition = (self.df.loc[self.df.index[-1], 'ADX'] > self.adx_threshold)
            else:
                adx_condition = True

            if self.chop_period and self.chop_threshold:
                chop_condition = (self.df.loc[self.df.index[-1], 'Choppiness'] < self.chop_threshold)
            else:
                chop_condition = True

            buy_signal = ema_buy_signal and adx_condition and chop_condition
            sell_signal = ema_sell_signal and adx_condition and chop_condition

            if buy_signal or sell_signal:
                side = 'BUY' if buy_signal else 'SELL'
                self.execute_trade(side, self.symbol, current_price)
                # Record the action in the DataFrame
                self.df.at[self.df.index[-1], 'action'] = side

        self.previous_df_orders = self.df_orders.copy()
        self.send_update()

    def send_update(self, message=None):
        if self.stop_event.is_set():
            return  # Don't send updates if we're shutting down
        channel_layer = get_channel_layer()
        trades = self.df_orders.tail(10).to_dict('records')
        
        # Convert Timestamp objects to ISO format strings
        for trade in trades:
            if isinstance(trade.get('timestamp'), pd.Timestamp):
                trade['timestamp'] = trade['timestamp'].isoformat()

        self.df = self.df.fillna(value=0)

        # Include all OHLC data with EMAs
        ohlc_data = self.df.to_dict('records')

        for ohlc in ohlc_data:
            if isinstance(ohlc.get('timestamp'), pd.Timestamp):
                ohlc['timestamp'] = ohlc['timestamp'].isoformat()
            ohlc['ema_slow'] = float(ohlc['ema_slow']) if pd.notna(ohlc.get('ema_slow')) else None
            ohlc['ema_fast'] = float(ohlc['ema_fast']) if pd.notna(ohlc.get('ema_fast')) else None
            ohlc['ADX'] = float(ohlc['ADX']) if pd.notna(ohlc.get('ADX')) else None
            ohlc['Choppiness'] = float(ohlc['Choppiness']) if pd.notna(ohlc.get('Choppiness')) else None
            ohlc['action'] = ohlc.get('action')  # Include the action

        last_update = self.df_orders.iloc[-1]['timestamp'] if not self.df_orders.empty else None
        if isinstance(last_update, pd.Timestamp):
            last_update = last_update.isoformat()

        nan_values = self.df.isna().sum()
        if nan_values.any():
            print(f"DataFrame has NaN values in the following columns: {nan_values[nan_values > 0].index.tolist()}")
        # else:
            # print("DataFrame does not have NaN values")

        

        update_data = {
            "status": "success",
            "trades": trades,
            "ohlc_data": ohlc_data,
            "last_update": last_update,
            "adx_threshold": self.adx_threshold,
            "chop_threshold": self.chop_threshold
        }
        
        if message:
            update_data["message"] = message

        async_to_sync(channel_layer.group_send)(
            "bot_updates",
            {
                "type": "bot_update",
                "message": update_data
            }
        )

    def on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")
        logger.error(traceback.format_exc())
        if not self.stop_event.is_set():
            self.send_update(f"Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        logger.info(f"WebSocket closed. Status code: {close_status_code}, Message: {close_msg}")
        if not self.stop_event.is_set():
            self.send_update("WebSocket connection closed")

    def on_open(self, ws):
        logger.info("WebSocket connection opened")
        if not self.stop_event.is_set():
            self.send_update("WebSocket connection opened")
        try:
            logger.info(f"Fetching historical data for {self.symbol} with interval {self.interval}")
            
            required_data_length = self.calculate_required_data_length()
            logger.info(f"Calculated required data length: {required_data_length}")
            
            # Convert interval to timedelta
            interval_td = self.interval_to_timedelta(self.interval)
            
            # Calculate the time range for historical data
            time_range = required_data_length * interval_td
            logger.info(f"Calculated time range: {time_range}")
            
            # Format the time range for the API call
            end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
            start_time = end_time - int(time_range.total_seconds() * 1000)
            
            # Fetch historical data
            klines = self.client.get_historical_klines(self.symbol, self.interval, start_time, end_time)
            
            logger.info(f"Received {len(klines)} historical candles")
            
            if len(klines) < required_data_length:
                logger.warning(f"Received fewer candles ({len(klines)}) than required ({required_data_length})")
                # Implement logic to fetch more data if needed
                while len(klines) < required_data_length:
                    logger.info(f"Attempting to fetch more candles. Current count: {len(klines)}")
                    new_start_time = start_time - int(time_range.total_seconds() * 1000)
                    additional_klines = self.client.get_historical_klines(self.symbol, self.interval, new_start_time, start_time)
                    if not additional_klines:
                        logger.warning("No more historical data available")
                        break
                    klines = additional_klines + klines
                    start_time = new_start_time
                    logger.info(f"Fetched {len(additional_klines)} additional candles. Total: {len(klines)}")
            
            logger.info(f"Final number of historical candles: {len(klines)}")
            
            # Process historical data
            self.process_historical_data(klines)
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error while fetching historical data: {e}")
            self.send_update(f"An API error occurred while fetching historical data: {e}")
        except Exception as e:
            logger.error(f"Unexpected error while fetching historical data: {e}", exc_info=True)
            self.send_update(f"An error occurred while fetching historical data: {e}")

    def interval_to_timedelta(self, interval):
        unit = interval[-1]
        value = int(interval[:-1])
        if unit == 's':
            return timedelta(seconds=value)
        elif unit == 'm':
            return timedelta(minutes=value)
        elif unit == 'h':
            return timedelta(hours=value)
        elif unit == 'd':
            return timedelta(days=value)
        elif unit == 'w':
            return timedelta(weeks=value)
        else:
            raise ValueError(f"Unsupported interval: {interval}")

    def calculate_initial_emas(self):
        if len(self.df) >= max(self.ema_slow, self.ema_fast):
            self.df['ema_slow'] = self.df['close'].ewm(span=self.ema_slow, adjust=False).mean()
            self.df['ema_fast'] = self.df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        else:
            # If we don't have enough data, initialize with NaN values
            self.df['ema_slow'] = float('nan')
            self.df['ema_fast'] = float('nan')

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def fetch_historical_klines(self, symbol, interval, start_time):
        try:
            return self.client.get_historical_klines(symbol, interval, start_time)
        except BinanceAPIException as e:
            if "Unsupported interval" in str(e):
                logger.warning(f"Interval {interval} not supported, trying with '1m'")
                return self.client.get_historical_klines(symbol, '1m', start_time)
            raise

    def stop(self):
        logger.info("Stopping TradingBot")
        self.stop_event.set()
        if self.ws:
            logger.debug(f"Closing WebSocket: {self.ws}")
            self.ws.close()
        logger.info("TradingBot stopped")

    def calculate_required_data_length(self):
        max_period = max(self.ema_slow, self.ema_fast, self.adx_period * 2, self.chop_period)
        additional_data = max(self.adx_period * 2, self.chop_period)
        return max_period + additional_data + 1

    def process_historical_data(self, klines):
        historical_data = []
        for kline in klines:
            historical_data.append({
                'timestamp': pd.to_datetime(kline[0], unit='ms', utc=True),
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5])
            })
        self.df = pd.DataFrame(historical_data)
        
        # Calculate all indicators
        self.calculate_indicators()
        
        # Take the tail of the DataFrame
        max_period = max(self.ema_slow, self.ema_fast, self.adx_period * 2, self.chop_period)
        self.df = self.df.tail(max_period)
        
        self.send_update(f"Initialized with {len(self.df)} historical candles")

    def calculate_indicators(self):
        if len(self.df) == 0:
            return

        # Check if we're dealing with historical data or real-time updates
        if 'ema_slow' not in self.df.columns:
            # Calculate EMAs for historical data
            self.df['ema_slow'] = self.df['close'].ewm(span=self.ema_slow, adjust=False).mean()
            self.df['ema_fast'] = self.df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        else:
            # Update only the last EMA values for real-time data
            current_close = self.df['close'].iloc[-1]
            previous_ema_slow = self.df['ema_slow'].iloc[-2]
            previous_ema_fast = self.df['ema_fast'].iloc[-2]

            self.df.loc[self.df.index[-1], 'ema_slow'] = self.calculate_ema(current_close, previous_ema_slow, self.ema_slow)
            self.df.loc[self.df.index[-1], 'ema_fast'] = self.calculate_ema(current_close, previous_ema_fast, self.ema_fast)

        # Calculate ADX
        if self.adx_period and self.adx_threshold:
            self.calculate_adx(self.adx_period)

        # Calculate Choppiness
        if self.chop_period and self.chop_threshold:
            self.calculate_choppiness(self.chop_period)

        # logger.info(f"Indicators recalculated. DataFrame length: {len(self.df)}")
        # logger.debug(f"Last row of DataFrame:\n{self.df.iloc[-1]}")