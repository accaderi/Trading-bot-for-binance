import pandas as pd
from trading_bot import TradingBot
from binance.client import Client
from tqdm.auto import tqdm
from datetime import datetime, timezone
import pickle
import os
import json

class BacktestTradingBot(TradingBot):
    def __init__(self, csv_file, symbol, timeframe, ema_slow, ema_fast, adx_period, adx_threshold, chop_period, chop_threshold, initial_balance):
        super().__init__(symbol, timeframe, ema_slow, ema_fast, adx_period, adx_threshold, chop_period, chop_threshold, initial_balance, 'dummy_api_key', 'dummy_api_secret')
        self.df = pd.read_csv(csv_file, parse_dates=['timestamp'])
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0
        self.trades = []

    def run_backtest(self):
        self.calculate_indicators()
        self.simulate_trades()
        self.print_results()

    def calculate_indicators(self):
        # Calculate EMAs
        self.df['ema_slow'] = self.df['close'].ewm(span=self.ema_slow, adjust=False).mean()
        self.df['ema_fast'] = self.df['close'].ewm(span=self.ema_fast, adjust=False).mean()

        # Calculate ADX
        if self.adx_period and self.adx_threshold:
            self.calculate_adx(self.adx_period)

        # Calculate Choppiness
        if self.chop_period and self.chop_threshold:
            self.calculate_choppiness(self.chop_period)

    def simulate_trades(self):
        for i in range(1, len(self.df)):
            current_price = self.df.loc[i, 'close']
            
            ema_diff = self.df.loc[i, 'ema_fast'] - self.df.loc[i, 'ema_slow']
            ema_diff_prev = self.df.loc[i-1, 'ema_fast'] - self.df.loc[i-1, 'ema_slow']
            
            ema_buy_signal = (ema_diff > 0) and (ema_diff_prev <= 0)
            ema_sell_signal = (ema_diff < 0) and (ema_diff_prev >= 0)

            adx_condition = (self.df.loc[i, 'ADX'] > self.adx_threshold) if self.adx_period and self.adx_threshold else True
            chop_condition = (self.df.loc[i, 'Choppiness'] < self.chop_threshold) if self.chop_period and self.chop_threshold else True

            buy_signal = ema_buy_signal and adx_condition and chop_condition
            sell_signal = ema_sell_signal and adx_condition and chop_condition

            if buy_signal and self.position <= 0:
                self.execute_trade('BUY', current_price)
            elif sell_signal and self.position >= 0:
                self.execute_trade('SELL', current_price)

    def execute_trade(self, side, price):
        if side == 'BUY':
            quantity = self.balance / price
            self.balance -= quantity * price
            self.position += quantity
        else:  # SELL
            self.balance += self.position * price
            self.position = 0

        self.trades.append({
            'timestamp': self.df.loc[len(self.trades), 'timestamp'],
            'side': side,
            'price': price,
            'quantity': quantity if side == 'BUY' else self.position,
            'balance': self.balance
        })

    def print_results(self):
        total_trades = len(self.trades)
        profitable_trades = sum(1 for i in range(1, len(self.trades)) if self.trades[i]['balance'] > self.trades[i-1]['balance'])
        final_balance = self.balance + self.position * self.df.iloc[-1]['close']
        
        print(f"Initial Balance: ${self.initial_balance:.2f}")
        print(f"Final Balance: ${final_balance:.2f}")
        print(f"Total Return: {((final_balance - self.initial_balance) / self.initial_balance) * 100:.2f}%")
        print(f"Total Trades: {total_trades}")
        print(f"Profitable Trades: {profitable_trades}")
        print(f"Win Rate: {(profitable_trades / total_trades) * 100:.2f}%")

def test_parameters(csv_file, symbol, timeframe, short_ema_periods, long_ema_periods, adx_periods, adx_thresholds, 
                    chop_periods, chop_thresholds, nowtime, use_ema=True, use_adx=True, use_chop=True, initial_balance=1000):
    """
    Test different combinations of parameters using BacktestTradingBot and return the results.
    Also saves results as CSV and pickle files.
    """
    results = []
    
    # Calculate total combinations
    total_combinations = 1
    if use_ema:
        total_combinations *= len(short_ema_periods) * len(long_ema_periods)
    if use_adx:
        total_combinations *= len(adx_periods) * len(adx_thresholds)
    if use_chop:
        total_combinations *= len(chop_periods) * len(chop_thresholds)
    
    print("Total combinations:", total_combinations)

    # Initialize progress bar
    pbar = tqdm(total=total_combinations, desc="Testing combinations")
    
    # Loop through all combinations
    for short_ema in (short_ema_periods if use_ema else [None]):
        for long_ema in (long_ema_periods if use_ema else [None]):
            if use_ema and short_ema >= long_ema:
                pbar.update(1)
                continue
            
            for adx_period in (adx_periods if use_adx else [None]):
                for adx_threshold in (adx_thresholds if use_adx else [None]):
                    for chop_period in (chop_periods if use_chop else [None]):
                        for chop_threshold in (chop_thresholds if use_chop else [None]):
                            
                            # Create BacktestTradingBot instance
                            backtest_bot = BacktestTradingBot(
                                csv_file, symbol, timeframe, long_ema, short_ema, 
                                adx_period, adx_threshold, chop_period, chop_threshold, 
                                initial_balance
                            )
                            
                            # Run backtest
                            backtest_bot.run_backtest()
                            
                            # Store the result
                            result = {
                                'short_ema': short_ema,
                                'long_ema': long_ema,
                                'adx_period': adx_period,
                                'adx_threshold': adx_threshold,
                                'chop_period': chop_period,
                                'chop_threshold': chop_threshold,
                                'final_balance': backtest_bot.balance + backtest_bot.position * backtest_bot.df.iloc[-1]['close'],
                                'total_trades': len(backtest_bot.trades),
                                'profitable_trades': sum(1 for i in range(1, len(backtest_bot.trades)) if backtest_bot.trades[i]['balance'] > backtest_bot.trades[i-1]['balance'])
                            }
                            results.append(result)
                            pbar.update(1)

    # Close the progress bar when done
    pbar.close()

    # Convert results to DataFrame
    df_results = pd.DataFrame(results)

    # Save to CSV and pickle
    csv_filename = f"{csv_file[:-41]}backtest_parameters_{nowtime}.csv"
    df_results.to_csv(csv_filename, index=False)
    print(f"Saved results to {csv_filename}")

    pkl_filename = f"{csv_file[:-41]}backtest_parameters_{nowtime}.pkl"
    with open(pkl_filename, 'wb') as f:
        pickle.dump(df_results, f)
    print(f"Saved results to {pkl_filename}")
    
    # Sort results by final balance
    df_results_sorted = df_results.sort_values('final_balance', ascending=False)

    # Display 3 best performing combinations
    print("\n3 Best Performing Combinations:")
    print(df_results_sorted.head(3).to_string(index=False))

    # Display 3 worst performing combinations
    print("\n3 Worst Performing Combinations:")
    print(df_results_sorted.tail(3).to_string(index=False))
    
    return df_results

# Usage example
if __name__ == "__main__":
    start_time = datetime(2023,1,1).strftime('%Y-%m-%dT%H:%M:%SZ')
    end_time = datetime(2023,12,31).strftime('%Y-%m-%dT%H:%M:%SZ')
    symbol = 'ETHUSDT'
    timeframe = '3m'
    # Get the current project directory
    current_dir = os.getcwd()
    # Create a 'data' subdirectory if it doesn't exist
    data_dir = os.path.join(current_dir, '20241004_trading/data')
    os.makedirs(data_dir, exist_ok=True)
    nowtime = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H_%M_%S')
    parameters = {
    "csv_file": os.path.join(data_dir, f'{symbol.replace("/", "_")}_{start_time[:10]}_{end_time[:10]}_{timeframe}_data.csv'),
    "symbol": symbol,
    "timeframe": timeframe,
    "short_ema_periods": [9, 12, 15, 20],
    "long_ema_periods": [21, 26, 30, 50, 100],
    "adx_periods": [10, 14, 20],
    "adx_thresholds": [20, 25, 30, 50],
    "chop_periods": [14, 20, 28],
    "chop_thresholds": [38, 50, 62],
    "nowtime": nowtime,
    "use_ema": True,
    "use_adx": False,
    "use_chop": False,
    "initial_balance": 1000
    }

    with open(f'{parameters["csv_file"][:-41]}parameters_{nowtime}.json', 'w') as f:
        json.dump(parameters, f, indent=4)

    results = test_parameters(**parameters)